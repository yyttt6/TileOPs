"""Ascend convolution kernels with tile-local im2col and a direct fallback."""

from functools import lru_cache
import math

import tilelang
import tilelang.language as T
import torch

from .common import UB_BUDGET_BYTES, grid_repeat_count, launch_block_count
from .gemm import _pipelined_plan as _gemm_pipelined_plan
from .gemm import build_gemm_kernel


_BLOCK_OC = 16
_BLOCK_SPATIAL = 64
_BLOCK_K = 64
_VEC = 2
_DIRECT_TILE = 64


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    raise TypeError(
        "Conv2dFwdOp supports float16, bfloat16, and float32, " f"got {dtype}"
    )


@lru_cache(maxsize=128)
def _compile_cube_conv2d(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    c_out: int,
    c_in_g: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
    out_h: int,
    out_w: int,
    dtype_name: str,
    has_bias: bool,
):
    if has_bias:
        raise ValueError("Cube Conv2d does not support bias")

    c_out_g = c_out // groups
    out_hw = out_h * out_w
    k_total = c_in_g * kernel_h * kernel_w
    oc_tiles = math.ceil(c_out_g / _BLOCK_OC)
    spatial_tiles = math.ceil(out_hw / _BLOCK_SPATIAL)
    k_tiles = math.ceil(k_total / _BLOCK_K)
    logical_blocks = n * groups * oc_tiles * spatial_tiles
    launch_blocks = launch_block_count(logical_blocks)
    grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)

    @tilelang.jit(
        out_idx=[3],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
            tilelang.PassConfigKey.TL_ASCEND_PTO_USE_PIPE_IN_CV_COPY: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def factory(dtype: str = dtype_name):
        @T.macro
        def compute(x, weight, workspace, output):
            x_flat = T.Tensor((n * c_in * h_in * w_in,), dtype, x.data)
            weight_flat = T.Tensor((c_out, k_total), dtype, weight.data)
            output_flat = T.Tensor((n, c_out, out_hw), dtype, output.data)

            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                im2col_ub = T.alloc_ub((_BLOCK_K // _VEC, _BLOCK_SPATIAL), dtype)
                data_l1 = T.alloc_L1((_BLOCK_K, _BLOCK_SPATIAL), dtype)
                weight_l1 = T.alloc_L1((_BLOCK_OC, _BLOCK_K), dtype)
                accum_l0 = T.alloc_L0C((_BLOCK_OC, _BLOCK_SPATIAL), "float")

                for grid_repeat in T.serial(grid_repeats):
                    logical_cid = cid + grid_repeat * launch_blocks
                    if logical_cid < logical_blocks:
                        spatial_tile = logical_cid % spatial_tiles
                        remaining = logical_cid // spatial_tiles
                        oc_tile = remaining % oc_tiles
                        remaining = remaining // oc_tiles
                        group_id = remaining % groups
                        batch_id = remaining // groups
                        spatial_base = spatial_tile * _BLOCK_SPATIAL
                        oc_base = group_id * c_out_g + oc_tile * _BLOCK_OC
                        ci_base = group_id * c_in_g

                        with T.Scope("C"):
                            for k_tile in T.serial(k_tiles):
                                k_base = k_tile * _BLOCK_K
                                T.wait_cross_flag(0)
                                T.copy(
                                    workspace[
                                        logical_cid,
                                        k_tile,
                                        0:_BLOCK_K,
                                        0:_BLOCK_SPATIAL,
                                    ],
                                    data_l1,
                                )
                                T.copy(
                                    weight_flat[
                                        oc_base : oc_base + _BLOCK_OC,
                                        k_base : k_base + _BLOCK_K,
                                    ],
                                    weight_l1,
                                )
                                T.barrier_all()
                                T.gemm_v0(
                                    weight_l1,
                                    data_l1,
                                    accum_l0,
                                    init=(k_tile == 0),
                                    kL0Size=_BLOCK_K,
                                )
                            T.copy(
                                accum_l0,
                                output_flat[
                                    batch_id,
                                    oc_base : oc_base + _BLOCK_OC,
                                    spatial_base : spatial_base + _BLOCK_SPATIAL,
                                ],
                            )

                        with T.Scope("V"):
                            for k_tile in T.serial(k_tiles):
                                k_base = k_tile * _BLOCK_K
                                for k_lane, spatial_lane in T.Parallel(
                                    _BLOCK_K // _VEC, _BLOCK_SPATIAL
                                ):
                                    k_index = k_base + vid * (_BLOCK_K // _VEC) + k_lane
                                    spatial_index = spatial_base + spatial_lane
                                    ci_local = k_index // (kernel_h * kernel_w)
                                    kernel_index = k_index % (kernel_h * kernel_w)
                                    kh = kernel_index // kernel_w
                                    kw = kernel_index % kernel_w
                                    oh = spatial_index // out_w
                                    ow = spatial_index % out_w
                                    ih = oh * stride_h - pad_h + kh * dilation_h
                                    iw = ow * stride_w - pad_w + kw * dilation_w
                                    valid = (
                                        (k_index < k_total)
                                        & (spatial_index < out_hw)
                                        & (ih >= 0)
                                        & (ih < h_in)
                                        & (iw >= 0)
                                        & (iw < w_in)
                                    )
                                    input_offset = (
                                        (batch_id * c_in + ci_base + ci_local) * h_in
                                        + ih
                                    ) * w_in + iw
                                    im2col_ub[k_lane, spatial_lane] = T.if_then_else(
                                        valid,
                                        x_flat[input_offset],
                                        T.cast(0.0, dtype),
                                    )

                                T.copy(
                                    im2col_ub,
                                    workspace[
                                        logical_cid,
                                        k_tile,
                                        vid * (_BLOCK_K // _VEC) : (vid + 1)
                                        * (_BLOCK_K // _VEC),
                                        0:_BLOCK_SPATIAL,
                                    ],
                                )
                                T.set_cross_flag("MTE3", 0)

        @T.prim_func
        def main(
            x: T.Tensor((n, c_in, h_in, w_in), dtype),
            weight: T.Tensor((c_out, c_in_g, kernel_h, kernel_w), dtype),
            workspace: T.Tensor(
                (logical_blocks, k_tiles, _BLOCK_K, _BLOCK_SPATIAL), dtype
            ),
            output: T.Tensor((n, c_out, out_h, out_w), dtype),
        ):
            compute(x, weight, workspace, output)

        return main

    return factory(dtype_name)


@lru_cache(maxsize=128)
def _compile_direct_conv2d(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    c_out: int,
    c_in_g: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
    out_h: int,
    out_w: int,
    dtype_name: str,
    has_bias: bool,
):
    c_out_g = c_out // groups
    out_hw = out_h * out_w
    total = n * c_out * out_hw
    logical_blocks = math.ceil(total / (_VEC * _DIRECT_TILE))
    launch_blocks = launch_block_count(logical_blocks)
    grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def factory(dtype: str = dtype_name):
        @T.macro
        def compute(x, weight, bias, output):
            x_flat = T.Tensor((n * c_in * h_in * w_in,), dtype, x.data)
            weight_flat = T.Tensor(
                (c_out * c_in_g * kernel_h * kernel_w,), dtype, weight.data
            )
            output_flat = T.Tensor((total,), dtype, output.data)

            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                result_fp32 = T.alloc_ub((_DIRECT_TILE,), "float32")
                result = T.alloc_ub((_DIRECT_TILE,), dtype)
                # dav-2201 has no scalar bf16 cast instruction in either
                # direction, so nothing may cross the dtype boundary one element
                # at a time.  Values are gathered with plain scalar moves into a
                # tile of the public dtype and widened a whole tile at a time.
                gather_dt = T.alloc_ub((_DIRECT_TILE,), dtype)
                gather_fp32 = T.alloc_ub((_DIRECT_TILE,), "float32")
                weight_dt = T.alloc_ub((_DIRECT_TILE,), dtype)
                weight_fp32 = T.alloc_ub((_DIRECT_TILE,), "float32")
                prod_fp32 = T.alloc_ub((_DIRECT_TILE,), "float32")
                # Output coordinates are decoded once per tile and reused by
                # every window position instead of being recomputed per window.
                oh_ub = T.alloc_ub((_DIRECT_TILE,), "int32")
                ow_ub = T.alloc_ub((_DIRECT_TILE,), "int32")
                x_base_ub = T.alloc_ub((_DIRECT_TILE,), "int32")
                w_base_ub = T.alloc_ub((_DIRECT_TILE,), "int32")

                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < logical_blocks:
                            start = (
                                logical_cid * (_VEC * _DIRECT_TILE) + vid * _DIRECT_TILE
                            )
                            T.tile.fill(result_fp32, 0.0)
                            T.tile.fill(gather_dt, 0.0)
                            # Lanes past the end never write weight_dt; zero it
                            # once so their product stays a finite zero.
                            T.tile.fill(weight_dt, 0.0)
                            for lane in T.serial(_DIRECT_TILE):
                                output_index = start + lane
                                if output_index < total:
                                    spatial_index = output_index % out_hw
                                    channel_batch = output_index // out_hw
                                    oc = channel_batch % c_out
                                    batch_id = channel_batch // c_out
                                    group_id = oc // c_out_g
                                    oh_ub[lane] = spatial_index // out_w
                                    ow_ub[lane] = spatial_index % out_w
                                    x_base_ub[lane] = (
                                        batch_id * c_in + group_id * c_in_g
                                    ) * (h_in * w_in)
                                    w_base_ub[lane] = (
                                        oc * c_in_g * kernel_h * kernel_w
                                    )
                                    if has_bias:
                                        gather_dt[lane] = bias[oc]
                            if has_bias:
                                if dtype == "float32":
                                    T.copy(gather_dt, gather_fp32)
                                else:
                                    T.tile.cast(
                                        gather_fp32,
                                        gather_dt,
                                        "CAST_NONE",
                                        _DIRECT_TILE,
                                    )
                                T.tile.add(result_fp32, result_fp32, gather_fp32)
                            for ci_local in T.serial(c_in_g):
                                for kh in T.serial(kernel_h):
                                    for kw in T.serial(kernel_w):
                                        weight_offset = (
                                            ci_local * kernel_h + kh
                                        ) * kernel_w + kw
                                        T.tile.fill(gather_dt, 0.0)
                                        for lane in T.serial(_DIRECT_TILE):
                                            if start + lane < total:
                                                weight_dt[lane] = weight_flat[
                                                    w_base_ub[lane] + weight_offset
                                                ]
                                                ih = (
                                                    oh_ub[lane] * stride_h
                                                    - pad_h
                                                    + kh * dilation_h
                                                )
                                                iw = (
                                                    ow_ub[lane] * stride_w
                                                    - pad_w
                                                    + kw * dilation_w
                                                )
                                                if (
                                                    ih >= 0
                                                    and ih < h_in
                                                    and iw >= 0
                                                    and iw < w_in
                                                ):
                                                    gather_dt[lane] = x_flat[
                                                        x_base_ub[lane]
                                                        + ci_local * (h_in * w_in)
                                                        + ih * w_in
                                                        + iw
                                                    ]
                                        if dtype == "float32":
                                            T.copy(gather_dt, gather_fp32)
                                            T.copy(weight_dt, weight_fp32)
                                        else:
                                            T.tile.cast(
                                                gather_fp32,
                                                gather_dt,
                                                "CAST_NONE",
                                                _DIRECT_TILE,
                                            )
                                            T.tile.cast(
                                                weight_fp32,
                                                weight_dt,
                                                "CAST_NONE",
                                                _DIRECT_TILE,
                                            )
                                        T.tile.mul(
                                            prod_fp32, gather_fp32, weight_fp32
                                        )
                                        T.tile.add(
                                            result_fp32, result_fp32, prod_fp32
                                        )

                            if dtype == "float32":
                                T.copy(
                                    result_fp32,
                                    output_flat[start : start + _DIRECT_TILE],
                                )
                            else:
                                T.tile.cast(
                                    result,
                                    result_fp32,
                                    "CAST_RINT",
                                    _DIRECT_TILE,
                                )
                                T.copy(
                                    result,
                                    output_flat[start : start + _DIRECT_TILE],
                                )

        if has_bias:

            @T.prim_func
            def main_bias(
                x: T.Tensor((n, c_in, h_in, w_in), dtype),
                weight: T.Tensor((c_out, c_in_g, kernel_h, kernel_w), dtype),
                bias: T.Tensor((c_out,), dtype),
                output: T.Tensor((n, c_out, out_h, out_w), dtype),
            ):
                compute(x, weight, bias, output)

            return main_bias

        @T.prim_func
        def main(
            x: T.Tensor((n, c_in, h_in, w_in), dtype),
            weight: T.Tensor((c_out, c_in_g, kernel_h, kernel_w), dtype),
            output: T.Tensor((n, c_out, out_h, out_w), dtype),
        ):
            compute(x, weight, None, output)

        return main

    return factory(dtype_name)


#: Lane count for the channel-bias epilogue.  Bounded so the three UB buffers
#: (dtype input, fp32 input, dtype output) stay well inside the arena.
_BIAS_TILE = 2048


def _bias_tile(out_spatial: int) -> int:
    """Lanes per vector op for the channel-bias epilogue.

    A block owns one whole ``(batch, channel)`` plane, so the bias is ONE fp32
    scalar for the entire block and the add is a full-tile vector op against a
    scalar -- never a lane-indexed read of ``bias``.  That is the same
    distinction R263 measured on ``bias_add``: the per-lane gather ran at
    6.3-18.4 GB/s where the vector path ran at 355-1010 GB/s, and fixing it
    moved that op from 0.0068 to 0.976 (R263-data/08-bias-add-row-broadcast.txt).
    """
    return min(_BIAS_TILE, out_spatial)


@lru_cache(maxsize=128)
def _compile_channel_bias_add(
    n: int, c_out: int, out_spatial: int, dtype_name: str
):
    """``out[b, c, s] = y[b, c, s] + bias[c]`` for a contiguous NC* tensor.

    One logical block per ``(batch, channel)`` plane.  ``bias[c]`` is read once
    per block as a scalar out of a UB fp32 staging buffer -- staged with one
    ``T.copy`` and widened with one ``T.tile.cast``, because dav-2201 has no
    scalar bf16 cast instruction (the same constraint the direct window kernel
    documents).  The add itself is ``T.tile.add(dst, src, scalar)``.
    """
    tile = _bias_tile(out_spatial)
    full_chunks = out_spatial // tile
    tail = out_spatial - full_chunks * tile
    logical_blocks = n * c_out
    launch_blocks = launch_block_count(logical_blocks)
    grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def factory(dtype: str = dtype_name):
        @T.prim_func
        def main(
            y: T.Tensor((n * c_out * out_spatial,), dtype),
            bias: T.Tensor((c_out,), dtype),
            output: T.Tensor((n * c_out * out_spatial,), dtype),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                bias_dt = T.alloc_ub((c_out,), dtype)
                bias_f32 = T.alloc_ub((c_out,), "float32")
                src_dt = T.alloc_ub((tile,), dtype)
                src_f32 = T.alloc_ub((tile,), "float32")
                acc_f32 = T.alloc_ub((tile,), "float32")
                out_dt = T.alloc_ub((tile,), dtype)
                with T.Scope("V"):
                    T.copy(bias, bias_dt)
                    if dtype == "float32":
                        T.copy(bias_dt, bias_f32)
                    else:
                        T.tile.cast(bias_f32, bias_dt, "CAST_NONE", c_out)
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < logical_blocks:
                            channel = logical_cid % c_out
                            plane = logical_cid * out_spatial
                            for chunk in T.serial(full_chunks):
                                base = plane + chunk * tile
                                T.copy(y[base : base + tile], src_dt)
                                if dtype == "float32":
                                    T.copy(src_dt, src_f32)
                                else:
                                    T.tile.cast(
                                        src_f32, src_dt, "CAST_NONE", tile
                                    )
                                T.tile.add(acc_f32, src_f32, bias_f32[channel])
                                if dtype == "float32":
                                    T.copy(acc_f32, output[base : base + tile])
                                else:
                                    T.tile.cast(
                                        out_dt, acc_f32, "CAST_RINT", tile
                                    )
                                    T.copy(out_dt, output[base : base + tile])
                            if tail:
                                base = plane + full_chunks * tile
                                T.copy(y[base : base + tail], src_dt[0:tail])
                                if dtype == "float32":
                                    T.copy(src_dt[0:tail], src_f32[0:tail])
                                else:
                                    T.tile.cast(
                                        src_f32[0:tail],
                                        src_dt[0:tail],
                                        "CAST_NONE",
                                        tail,
                                    )
                                T.tile.add(
                                    acc_f32[0:tail],
                                    src_f32[0:tail],
                                    bias_f32[channel],
                                )
                                if dtype == "float32":
                                    T.copy(
                                        acc_f32[0:tail],
                                        output[base : base + tail],
                                    )
                                else:
                                    T.tile.cast(
                                        out_dt[0:tail],
                                        acc_f32[0:tail],
                                        "CAST_RINT",
                                        tail,
                                    )
                                    T.copy(
                                        out_dt[0:tail],
                                        output[base : base + tail],
                                    )

        return main

    return factory(dtype_name)


def _build_bias_epilogue(base_launch, n, c_out, out_shape, dtype):
    """Wrap a bias-free convolution kernel with the channel-bias epilogue.

    WHY THIS EXISTS.  The bias-carrying convolutions were the whole of Conv2d's
    ``ratio_min``: every one of them fell into ``_compile_direct_conv2d``, whose
    gather is a per-lane scalar GM load.  Measured on
    ``Conv2dFwdOp/stride2-bias``/bf16, 20 reps at Level1 ``PipeUtilization``
    (R269-data/logs/02-pipes-stride2bias-before.log):

        main_kernel  Duration 77107.70 us
          aiv_time        76676.44 us   (99.44% of the kernel)
          aiv_scalar_time 75914.99 us   aiv_scalar_ratio 0.990
          aiv_vec_time      523.83 us
          aiv_mte2_time       0.009 us  <- the vector LOAD pipe is idle

    i.e. the kernel reads every input and weight element through the scalar
    pipe and never issues a vector load at all.  A convolution WITHOUT bias of
    the same shape takes the cube path and is ~100x faster, so bias was buying
    a 100x slowdown for one add per output element.  Splitting it into
    (bias-free convolution) + (vector channel-bias add) keeps the fast path and
    pays ~3 extra bytes of traffic per output element.

    The add is done in fp32 after the convolution has rounded to ``dtype``,
    whereas ``torch.nn.functional.conv2d`` adds bias inside the accumulator.
    That is one extra rounding of the pre-bias sum; section 10 of R269 reports
    the measured ``max_abs_err`` for every workload.
    """
    dtype_name = _dtype_name(dtype)
    out_spatial = 1
    for extent in out_shape[2:]:
        out_spatial *= int(extent)
    epilogue = _compile_channel_bias_add(n, c_out, out_spatial, dtype_name)

    def launch(x, weight, bias=None):
        if bias is None:
            raise ValueError("bias epilogue requires a bias tensor")
        product = base_launch(x, weight)
        return epilogue(product.reshape(-1), bias).reshape(out_shape)

    launch.path_kind = f"{base_launch.path_kind}+bias_epilogue"
    launch.output_shape = tuple(out_shape)
    launch.workspace_shape = getattr(base_launch, "workspace_shape", None)
    launch.logical_blocks = getattr(base_launch, "logical_blocks", None)
    launch.launch_blocks = getattr(base_launch, "launch_blocks", None)
    launch.grid_repeats = getattr(base_launch, "grid_repeats", None)
    launch.bias_logical_blocks = n * c_out
    launch.bias_tile = _bias_tile(out_spatial)
    launch.compiled = getattr(base_launch, "compiled", None)
    launch.base_launch = base_launch
    launch.epilogue = epilogue
    return launch


#: Upper bound on lanes an accumulator row of the vectorised window kernel may
#: hold.  Larger tiles mean longer staging copies (fewer, bigger DMAs) but the
#: accumulator is ``oc_tile`` rows deep, so this is the knob the UB budget
#: trades against output-channel reuse.
_VW_TILE_MAX = 2048

#: Most output channels one block accumulates at once.  The point of blocking
#: the output channel axis at all is that all of them share the SAME staged
#: input window, so the staging traffic is divided by this number.
_VW_OC_MAX = 16

#: UB waterline for this kernel.  Below ``common.UB_BUDGET_BYTES`` (180224)
#: because the declared buffers summing under budget is not the same as the
#: allocator accepting them (the warning on ``UB_BUDGET_NORM_BYTES``).
_VW_UB_BUDGET = 155648

#: Constants of the tile cost model, in rough AIV cycles.  ``_VW_DMA_CYCLES``
#: is the issue cost of one ``T.copy``; ``_VW_ISSUE_CYCLES`` that of one vector
#: instruction; a vector instruction additionally costs ``lanes / 32`` cycles.
#: ``_VW_SCALAR_STORE_CYCLES`` is one iteration of a ``T.serial`` zero-store
#: loop into UB, including its address arithmetic -- calibrated in R269
#: section 6 from the deeplabv3-aspp measurement, where those loops are 90% of
#: the kernel.  It is still 4x cheaper than the ~142 cycles R269 section 2
#: measured for a scalar LOAD out of GM, which is the point of this kernel.
_VW_DMA_CYCLES = 120
_VW_ISSUE_CYCLES = 12
_VW_SCALAR_STORE_CYCLES = 34

#: Vector cores on an ascend910b1 (24 AI cores x 2 AIV).
_VW_CORES = 48

#: How much cheaper the vector window must MODEL than the cube path before a
#: shape is moved off the cube path.  See the router in
#: :func:`build_conv2d_kernel` and R269 section 6.
_VW_ROUTE_MARGIN = 4

#: Cycles per scalar GM load issued from the cube kernel's ``T.Parallel`` lane
#: loop, calibrated against R268 section 2's measurement.  See
#: :func:`_cube_window_cost`.
_VW_CUBE_SCALAR_CYCLES = 34.7


def _round_up(value: int, multiple: int) -> int:
    return -(-int(value) // int(multiple)) * int(multiple)


def _vector_window_plan(
    n, groups, spatial_in, spatial_out, kernel_extents, stride, padding,
    dilation, c_in_g, c_out_g, k_total, itemsize,
):
    """Geometry for the vectorised direct-window kernel, or ``None`` to refuse.

    ADMISSION -- why these two conditions and not others.
    ``_compile_direct_conv2d`` / ``_compile_direct_conv_nd`` compute one scalar
    address per lane per window tap and issue one scalar GM load for it.  R269
    section 2 measured what that costs on two independent cases
    (``mobilenetv2-depthwise``/fp16 and ``stride2-bias``/bf16): ``aiv_time`` is
    99.4-99.6% of the kernel, ``aiv_scalar_ratio`` is 0.99, and
    ``aiv_mte2_time`` is 0.009-0.035 us -- the vector LOAD pipe never runs at
    all.  Dividing the measured scalar time by the number of loads the shape
    implies gives ~142 cycles per scalar GM load.

    The fix is to make the staging a ``T.copy``, which needs the lanes of a
    tile to be CONTIGUOUS in the input for a fixed window tap.  Write the input
    element feeding output lane ``p`` (a flat index into the output's spatial
    domain) for tap ``k``.  If every axis has stride 1 and the output extent
    equals the input extent on every axis, then

        flat_in(p, k) = p + delta(k),
        delta(k) = sum_a (k_a * dilation_a - padding_a) * input_stride_a

    with ``delta`` a compile-time constant: the whole tile is one contiguous
    run.  Lanes whose shifted coordinate leaves its axis are still READ (the
    address is inside the tensor, just in the wrong row) and are zeroed
    afterwards, which is correct because a zeroed lane contributes nothing.

    Rank 1 is admitted more widely: with stride 1 and no padding the shift is
    always non-negative and in range, and no lane ever needs zeroing, so the
    output extent does not have to match the input extent.

    Deliberately NOT admitted, and why:
      * any axis with stride > 1 -- the lanes are then strided in the input,
        not contiguous, and ``T.copy`` cannot express that.  R269 section 9
        records this as the largest piece of money left on the table.
      * rank >= 2 whose padding changes the extent -- ``flat_in`` is then not
        ``p + const`` and the whole argument collapses.
      * rank 1 with padding -- admissible in principle, but the lanes needing
        zeroing depend on the (dynamic) chunk index, which the rank >= 2 row
        machinery handles and the rank-1 single-copy path does not.

    THE TILE SEARCH.  Two competing pressures: a wider tile means fewer, longer
    staging copies, but the accumulator is ``oc_tile`` tiles deep and output
    channel reuse is what divides the staging traffic.  Every (tile width,
    output-channel block) pair that fits the UB budget is scored with the cost
    model below and the cheapest wins; see the comment on that loop.
    """
    rank = len(spatial_in)
    if rank not in (1, 2, 3):
        return None
    if any(int(s) != 1 for s in stride):
        return None
    out_spatial = 1
    for extent in spatial_out:
        out_spatial *= int(extent)
    align = 32 // itemsize
    if rank == 1:
        if int(padding[0]) != 0:
            return None
        row_len = None
        group_rows = 1
        unit = align
    else:
        if tuple(int(v) for v in spatial_out) != tuple(int(v) for v in spatial_in):
            return None
        row_len = int(spatial_out[-1])
        plane_rows = out_spatial // row_len
        group_rows = 0
        for candidate in range(1, plane_rows + 1):
            if (candidate * row_len * itemsize) % 32 == 0:
                group_rows = candidate
                break
        if not group_rows or plane_rows % group_rows:
            return None
        unit = group_rows * row_len
    if unit > _VW_TILE_MAX or unit > out_spatial:
        return None
    # Enumerate every (tile width, output-channel block) pair that fits UB and
    # score it with an explicit cost model, because the two knobs pull against
    # each other and which one wins is shape-dependent:
    #   * a wider tile means fewer, longer staging copies, but the accumulator
    #     is `oc_tile` tiles deep, so a wide tile can force oc_tile down;
    #   * `oc_tile` is what divides the staging traffic, because all oc lanes
    #     of a block consume the SAME staged window;
    #   * the weight staging buffers are `oc_tile * k_total` elements, so a
    #     large kernel (encodec-deep: k_total = 1280) makes oc_tile expensive
    #     in UB while a small one (depthwise: k_total = 9) makes it free.
    # Units are rough AIV cycles.  The model is documented, not fitted; R269
    # section 9 reports the alternatives that were measured against it.
    span = min(_VW_TILE_MAX, out_spatial)
    taps = 1
    for extent in kernel_extents:
        taps *= int(extent)
    # Total lanes needing a scalar zero-store, summed over taps, per row.
    band_total = 0
    leading_fraction = 0.0
    if rank > 1:
        for tap in _iter_taps(kernel_extents):
            _, lo, hi, shift = _tap_geometry(
                tap, spatial_in, padding, dilation, rank
            )
            band_total += lo + hi
            share = 0.0
            for axis in range(rank - 1):
                share += abs(shift[axis]) / float(int(spatial_out[axis]))
            leading_fraction += min(1.0, share)
    best = None
    for multiplier in range(span // unit, 0, -1):
        tile = multiplier * unit
        rows = tile // row_len if rank > 1 else 1
        chunks = -(-out_spatial // tile)
        acc_stride = _round_up(tile, 8)
        w_stride = _round_up(taps, align)
        groups_per_tile = tile // unit if rank > 1 else 1
        fixed = tile * (itemsize + 4) + tile * itemsize + rows * 4 * 2 + 2048
        for oc_tile in range(min(_VW_OC_MAX, c_out_g), 0, -1):
            if c_out_g % oc_tile:
                continue
            ub_bytes = (fixed + oc_tile * acc_stride * 4
                        + oc_tile * w_stride * (itemsize + 4)
                        + oc_tile * (itemsize + 4))
            if ub_bytes > _VW_UB_BUDGET:
                continue
            items = n * groups * (c_out_g // oc_tile) * chunks
            waves = -(-items // _VW_CORES)
            lanes = tile / 32.0
            # scalar zero-stores per tap, averaged: the column bands always,
            # plus a whole row for the fraction of rows whose shifted leading
            # coordinate leaves its axis.
            # Scalar zero-stores per tap, averaged over taps: the column
            # bands on every row, plus whole rows for the fraction of rows
            # whose shifted leading coordinate leaves its axis.  That fraction
            # is |shift| / extent, which is what makes a heavily DILATED
            # convolution expensive here: deeplabv3-aspp shifts by 12 rows of
            # 32, so 37% of every tile is zeroed one scalar at a time.
            fix = band_total * rows / taps
            if rank > 1:
                fix += row_len * rows * leading_fraction / taps
            # One contiguous copy per tap; `groups_per_tile` only bounds the
            # boundary-tile fallback, which is rare enough to leave out.
            per_tap = (_VW_DMA_CYCLES + tile * itemsize / 32.0
                       + 2 * (lanes + _VW_ISSUE_CYCLES)
                       + oc_tile * (lanes + _VW_ISSUE_CYCLES)
                       + _VW_SCALAR_STORE_CYCLES * fix)
            per_ci = (taps * per_tap
                      + oc_tile * (_VW_DMA_CYCLES + taps * itemsize / 32.0)
                      + oc_tile * w_stride / 32.0 + _VW_ISSUE_CYCLES)
            per_item = (c_in_g * per_ci
                        + 2 * oc_tile * (_VW_DMA_CYCLES + lanes
                                         + _VW_ISSUE_CYCLES))
            cost = waves * per_item
            candidate = {
                "tile": tile,
                "rows": rows,
                "row_len": row_len if rank > 1 else tile,
                "group_rows": group_rows,
                "group_lanes": unit if rank > 1 else tile,
                "groups_per_tile": groups_per_tile,
                "chunks": chunks,
                "acc_stride": acc_stride,
                "w_stride": w_stride,
                "oc_tile": oc_tile,
                "items": items,
                "ub_bytes": ub_bytes,
                "model_cost": round(cost),
            }
            if best is None or cost < best["model_cost"]:
                best = candidate
    return best


def _cube_window_cost(n, groups, c_out_g, out_hw, k_total):
    """Rough AIV cycles for ``_compile_cube_conv2d``, in the same units as
    ``_vector_window_plan``'s ``model_cost``.

    The cube kernel's cost is NOT its Cube time.  R268 section 2 measured
    ``resnet-3x3``/fp16 at ``aiv_scalar_ratio = 1.000`` and
    ``aiv_time / Duration = 99.602%``, with the five ``aic_*`` pipes summing to
    0.869% of the kernel: what the kernel actually spends its time on is the
    per-lane scalar gather that builds the im2col tile in a GM workspace.  So
    counting those scalar loads IS the cost model.

    Calibration.  That measurement gives 5808.26 us of ``aiv_scalar_time`` for
    ``392 * 9 * 4096`` scalar loads over 48 vector cores at 1.8 GHz, i.e.

        5808.26e-6 * 1.8e9 * 48 / (392 * 9 * 4096) = 34.7 cycles per load.

    ``_VW_CUBE_SCALAR_CYCLES`` is that number.  Note it is FOUR TIMES cheaper
    than the ~142 cycles per scalar load implied by the direct-window kernel's
    own measurement (R269 section 2): the cube kernel issues its loads from a
    ``T.Parallel`` lane loop, which pipelines them, while the direct kernel
    issues them from a ``T.serial`` one, which does not.
    """
    oc_tiles = -(-c_out_g // _BLOCK_OC)
    spatial_tiles = -(-out_hw // _BLOCK_SPATIAL)
    k_tiles = -(-k_total // _BLOCK_K)
    logical_blocks = n * groups * oc_tiles * spatial_tiles
    loads = logical_blocks * k_tiles * _BLOCK_K * _BLOCK_SPATIAL
    return loads * _VW_CUBE_SCALAR_CYCLES / _VW_CORES


def _iter_taps(kernel_extents):
    extents = [int(v) for v in kernel_extents]
    if len(extents) == 1:
        return [(k0,) for k0 in range(extents[0])]
    if len(extents) == 2:
        return [(k0, k1) for k0 in range(extents[0]) for k1 in range(extents[1])]
    return [(k0, k1, k2)
            for k0 in range(extents[0])
            for k1 in range(extents[1])
            for k2 in range(extents[2])]


def _tap_geometry(taps, spatial_in, padding, dilation, rank):
    """Per-tap static geometry: flat input shift and the invalid lane bands.

    ``taps`` is a tuple of kernel indices, one per spatial axis.  Returns
    ``(delta, lo, hi, shift)`` where ``delta`` is the flat input shift,
    ``lo``/``hi`` are how many lanes at the start / end of every row read out of
    the row (and must be zeroed), and ``shift`` is the per-leading-axis
    coordinate shift used for whole-row validity.
    """
    shift = tuple(int(taps[a]) * int(dilation[a]) - int(padding[a])
                  for a in range(rank))
    delta = 0
    stride_in = 1
    for axis in range(rank - 1, -1, -1):
        delta += shift[axis] * stride_in
        stride_in *= int(spatial_in[axis])
    inner = shift[rank - 1]
    return delta, max(0, -inner), max(0, inner), shift


@lru_cache(maxsize=128)
def _compile_vector_conv(
    n: int,
    c_in: int,
    spatial_in: tuple[int, ...],
    c_out: int,
    c_in_g: int,
    kernel: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    spatial_out: tuple[int, ...],
    dtype_name: str,
    has_bias: bool,
    plan_key: tuple,
):
    """Direct convolution whose window staging is a ``T.copy``, not a gather.

    One block owns ``(batch, group, oc_tile output channels, tile output lanes)``.
    Per window tap it stages the input window with ONE contiguous ``T.copy``
    and then folds it into every one of the ``oc_tile`` accumulators with
    ``T.tile.axpy`` against a weight scalar read out of UB.  So per tap the
    kernel issues 1 vector load and ``oc_tile`` vector FMAs, where the scalar
    kernel it replaces issued ``2 * tile`` scalar GM loads.  (``groups_per_tile``
    aligned sub-block copies are issued instead for the rare tile whose source
    range crosses the end of the tensor.)

    Weights and bias are staged into UB and widened to fp32 in one
    ``T.tile.cast`` each, because dav-2201 has no scalar bf16 cast instruction:
    a bf16 weight cannot become an fp32 axpy scalar one element at a time, but
    an fp32 UB element can be read as a scalar directly.
    """
    plan = dict(plan_key)
    rank = len(spatial_in)
    c_out_g = c_out // groups
    in_spatial = 1
    for extent in spatial_in:
        in_spatial *= extent
    out_spatial = 1
    for extent in spatial_out:
        out_spatial *= extent
    k_total = c_in_g
    for extent in kernel:
        k_total *= extent
    numel_x = n * c_in * in_spatial
    tile = plan["tile"]
    rows = plan["rows"]
    row_len = plan["row_len"]
    group_lanes = plan["group_lanes"]
    group_rows = plan["group_rows"]
    groups_per_tile = plan["groups_per_tile"]
    chunks = plan["chunks"]
    acc_stride = plan["acc_stride"]
    w_stride = plan["w_stride"]
    oc_tile = plan["oc_tile"]
    oc_blocks = c_out_g // oc_tile
    items = n * groups * oc_blocks * chunks
    logical_blocks = -(-items // _VEC)
    launch_blocks = launch_block_count(logical_blocks)
    grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)
    last_start = out_spatial - tile
    # Window taps are visited in the SAME order as the scalar kernel visits
    # them (c_in_g outermost, then the kernel axes row-major), so every output
    # element's fp32 accumulation order is unchanged and the two kernels are
    # expected to agree bit for bit where the scalar one is also exact.
    taps = 1
    for extent in kernel:
        taps *= extent

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def factory(dtype: str = dtype_name):
        @T.macro
        def compute(x, weight, bias, output):
            x_flat = T.Tensor((numel_x,), dtype, x.data)
            weight_flat = T.Tensor((c_out * k_total,), dtype, weight.data)
            output_flat = T.Tensor((n * c_out * out_spatial,), dtype, output.data)
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                acc = T.alloc_ub((oc_tile, acc_stride), "float32")
                stage_dt = T.alloc_ub((tile,), dtype)
                stage_f32 = T.alloc_ub((tile,), "float32")
                out_dt = T.alloc_ub((tile,), dtype)
                w_dt = T.alloc_ub((oc_tile, w_stride), dtype)
                w_f32 = T.alloc_ub((oc_tile, w_stride), "float32")
                bias_dt = T.alloc_ub((oc_tile,), dtype)
                bias_f32 = T.alloc_ub((oc_tile,), "float32")
                coord_ub = T.alloc_ub((max(rows, 2) * 2,), "int32")
                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        item = (cid + grid_repeat * launch_blocks) * _VEC + vid
                        if item < items:
                            chunk = item % chunks
                            rest = item // chunks
                            oc_block = rest % oc_blocks
                            rest = rest // oc_blocks
                            group_id = rest % groups
                            batch_id = rest // groups
                            oc_base = group_id * c_out_g + oc_block * oc_tile
                            ci_base = group_id * c_in_g
                            # The last chunk of a plane is CLAMPED, not
                            # truncated: it recomputes the lanes the previous
                            # chunk already produced and stores the same values
                            # over them.  A store is idempotent (unlike an
                            # accumulate), so this removes the need for
                            # `tile` to divide `out_spatial` and keeps every
                            # copy extent a compile-time constant.
                            start = T.if_then_else(
                                chunk * tile < last_start,
                                chunk * tile,
                                last_start,
                            )
                            x_base = (batch_id * c_in + ci_base) * in_spatial
                            T.tile.fill(w_dt, 0.0)
                            if has_bias:
                                T.copy(bias[oc_base : oc_base + oc_tile], bias_dt)
                                if dtype == "float32":
                                    T.copy(bias_dt, bias_f32)
                                else:
                                    T.tile.cast(
                                        bias_f32, bias_dt, "CAST_NONE", oc_tile
                                    )
                            for oc_lane in T.serial(oc_tile):
                                T.tile.fill(acc[oc_lane, 0:tile], 0.0)
                            if rank > 1:
                                # Leading output coordinates of every row this
                                # tile owns, decoded once instead of once per
                                # tap.
                                for r in T.serial(rows):
                                    global_row = start // row_len + r
                                    if rank == 2:
                                        coord_ub[r] = global_row
                                    else:
                                        coord_ub[r] = global_row % spatial_out[1]
                                        coord_ub[rows + r] = (
                                            global_row // spatial_out[1]
                                        )
                            if rank == 1:
                                for ci_local in T.serial(c_in_g):
                                    # Stage the `taps` weights of this input channel for
                                    # every one of the `oc_tile` output channels.  Staging
                                    # the whole k_total per BLOCK instead makes the UB cost
                                    # of oc_tile proportional to k_total, which on
                                    # deeplabv3-aspp (k_total = 18432) forced oc_tile down
                                    # to 1 -- i.e. no output-channel reuse at all, and the
                                    # staged input window paid for by exactly one
                                    # accumulator (R269 section 6).
                                    for oc_lane in T.serial(oc_tile):
                                        T.copy(
                                            weight_flat[
                                                (oc_base + oc_lane) * k_total + ci_local * taps
                                                : (oc_base + oc_lane) * k_total
                                                + ci_local * taps + taps
                                            ],
                                            w_dt[oc_lane, 0:taps],
                                        )
                                    if dtype == "float32":
                                        T.copy(w_dt, w_f32)
                                    else:
                                        T.tile.cast(
                                            w_f32, w_dt, "CAST_NONE", oc_tile * w_stride
                                        )
                                    for tap in T.serial(taps):
                                        # Tap geometry is computed at RUNTIME (as
                                        # scalars) rather than by unrolling the
                                        # window in Python: TVMScript only accepts
                                        # `range` / `T.serial` style loops, and an
                                        # unrolled 27-tap rank-3 window would also
                                        # multiply the emitted code by 27.  Every
                                        # value here is a scalar arithmetic
                                        # expression; the COPY extents stay
                                        # compile-time constants, which is the only
                                        # thing the DMA needs.
                                        k0 = tap
                                        inner = k0 * dilation[0] - padding[0]
                                        delta = inner
                                        src = (
                                            x_base
                                            + ci_local * in_spatial
                                            + start
                                            + delta
                                        )
                                        lo = T.if_then_else(inner < 0, -inner, 0)
                                        hi = T.if_then_else(inner > 0, inner, 0)
                                        T.tile.fill(stage_dt, 0.0)
                                        # ONE copy for the whole tile.  Consecutive output
                                        # rows map to consecutive input rows -- that is what
                                        # the extent-preserving admission buys -- so the
                                        # tile's source is a single contiguous run.  Issuing
                                        # one copy PER ROW instead was measured at 27x
                                        # slower on deeplabv3-aspp (R269 section 6): it
                                        # turned one 2 KiB DMA into 32 of 64 bytes.
                                        if src >= 0 and src + tile <= numel_x:
                                            T.copy(x_flat[src : src + tile], stage_dt)
                                        else:
                                            # Boundary tile: some sub-block of it is outside
                                            # the tensor.  Guard each 32-byte-aligned
                                            # sub-block, and fall back to per-lane moves only
                                            # for the ones that really are out.  Those lanes
                                            # are all zeroed below anyway; this branch exists
                                            # so the kernel never dereferences outside the
                                            # tensor.
                                            for grp in T.serial(groups_per_tile):
                                                gsrc = src + grp * group_lanes
                                                if (gsrc >= 0
                                                        and gsrc + group_lanes <= numel_x):
                                                    T.copy(
                                                        x_flat[gsrc : gsrc + group_lanes],
                                                        stage_dt[
                                                            grp * group_lanes
                                                            : grp * group_lanes
                                                            + group_lanes
                                                        ],
                                                    )
                                                else:
                                                    for lane in T.serial(group_lanes):
                                                        if (gsrc + lane >= 0
                                                                and gsrc + lane < numel_x):
                                                            stage_dt[
                                                                grp * group_lanes + lane
                                                            ] = x_flat[gsrc + lane]
                                        if dtype == "float32":
                                            T.copy(stage_dt, stage_f32)
                                        else:
                                            T.tile.cast(
                                                stage_f32, stage_dt, "CAST_NONE", tile
                                            )
                                        # Rank 1 with zero padding: the shift is always in
                                        # range, so no lane is ever invalid.
                                        pass
                                        for oc_lane in T.serial(oc_tile):
                                            T.tile.axpy(
                                                acc[oc_lane, 0:tile],
                                                stage_f32,
                                                w_f32[oc_lane, tap],
                                            )
                            elif rank == 2:
                                for ci_local in T.serial(c_in_g):
                                    # Stage the `taps` weights of this input channel for
                                    # every one of the `oc_tile` output channels.  Staging
                                    # the whole k_total per BLOCK instead makes the UB cost
                                    # of oc_tile proportional to k_total, which on
                                    # deeplabv3-aspp (k_total = 18432) forced oc_tile down
                                    # to 1 -- i.e. no output-channel reuse at all, and the
                                    # staged input window paid for by exactly one
                                    # accumulator (R269 section 6).
                                    for oc_lane in T.serial(oc_tile):
                                        T.copy(
                                            weight_flat[
                                                (oc_base + oc_lane) * k_total + ci_local * taps
                                                : (oc_base + oc_lane) * k_total
                                                + ci_local * taps + taps
                                            ],
                                            w_dt[oc_lane, 0:taps],
                                        )
                                    if dtype == "float32":
                                        T.copy(w_dt, w_f32)
                                    else:
                                        T.tile.cast(
                                            w_f32, w_dt, "CAST_NONE", oc_tile * w_stride
                                        )
                                    for tap in T.serial(taps):
                                        # Tap geometry is computed at RUNTIME (as
                                        # scalars) rather than by unrolling the
                                        # window in Python: TVMScript only accepts
                                        # `range` / `T.serial` style loops, and an
                                        # unrolled 27-tap rank-3 window would also
                                        # multiply the emitted code by 27.  Every
                                        # value here is a scalar arithmetic
                                        # expression; the COPY extents stay
                                        # compile-time constants, which is the only
                                        # thing the DMA needs.
                                        k0 = tap // kernel[1]
                                        k1 = tap % kernel[1]
                                        outer = k0 * dilation[0] - padding[0]
                                        inner = k1 * dilation[1] - padding[1]
                                        delta = outer * spatial_in[1] + inner
                                        src = (
                                            x_base
                                            + ci_local * in_spatial
                                            + start
                                            + delta
                                        )
                                        lo = T.if_then_else(inner < 0, -inner, 0)
                                        hi = T.if_then_else(inner > 0, inner, 0)
                                        T.tile.fill(stage_dt, 0.0)
                                        # ONE copy for the whole tile.  Consecutive output
                                        # rows map to consecutive input rows -- that is what
                                        # the extent-preserving admission buys -- so the
                                        # tile's source is a single contiguous run.  Issuing
                                        # one copy PER ROW instead was measured at 27x
                                        # slower on deeplabv3-aspp (R269 section 6): it
                                        # turned one 2 KiB DMA into 32 of 64 bytes.
                                        if src >= 0 and src + tile <= numel_x:
                                            T.copy(x_flat[src : src + tile], stage_dt)
                                        else:
                                            # Boundary tile: some sub-block of it is outside
                                            # the tensor.  Guard each 32-byte-aligned
                                            # sub-block, and fall back to per-lane moves only
                                            # for the ones that really are out.  Those lanes
                                            # are all zeroed below anyway; this branch exists
                                            # so the kernel never dereferences outside the
                                            # tensor.
                                            for grp in T.serial(groups_per_tile):
                                                gsrc = src + grp * group_lanes
                                                if (gsrc >= 0
                                                        and gsrc + group_lanes <= numel_x):
                                                    T.copy(
                                                        x_flat[gsrc : gsrc + group_lanes],
                                                        stage_dt[
                                                            grp * group_lanes
                                                            : grp * group_lanes
                                                            + group_lanes
                                                        ],
                                                    )
                                                else:
                                                    for lane in T.serial(group_lanes):
                                                        if (gsrc + lane >= 0
                                                                and gsrc + lane < numel_x):
                                                            stage_dt[
                                                                grp * group_lanes + lane
                                                            ] = x_flat[gsrc + lane]
                                        if dtype == "float32":
                                            T.copy(stage_dt, stage_f32)
                                        else:
                                            T.tile.cast(
                                                stage_f32, stage_dt, "CAST_NONE", tile
                                            )
                                        for r in T.serial(rows):
                                            for col in T.serial(lo):
                                                stage_f32[r * row_len + col] = 0.0
                                            for col in T.serial(hi):
                                                stage_f32[
                                                    r * row_len + row_len - 1 - col
                                                ] = 0.0
                                            if (coord_ub[r] + outer < 0
                                                    or coord_ub[r] + outer
                                                    >= spatial_out[0]):
                                                for col in T.serial(row_len):
                                                    stage_f32[
                                                        r * row_len + col
                                                    ] = 0.0
                                        for oc_lane in T.serial(oc_tile):
                                            T.tile.axpy(
                                                acc[oc_lane, 0:tile],
                                                stage_f32,
                                                w_f32[oc_lane, tap],
                                            )
                            elif rank == 3:
                                for ci_local in T.serial(c_in_g):
                                    # Stage the `taps` weights of this input channel for
                                    # every one of the `oc_tile` output channels.  Staging
                                    # the whole k_total per BLOCK instead makes the UB cost
                                    # of oc_tile proportional to k_total, which on
                                    # deeplabv3-aspp (k_total = 18432) forced oc_tile down
                                    # to 1 -- i.e. no output-channel reuse at all, and the
                                    # staged input window paid for by exactly one
                                    # accumulator (R269 section 6).
                                    for oc_lane in T.serial(oc_tile):
                                        T.copy(
                                            weight_flat[
                                                (oc_base + oc_lane) * k_total + ci_local * taps
                                                : (oc_base + oc_lane) * k_total
                                                + ci_local * taps + taps
                                            ],
                                            w_dt[oc_lane, 0:taps],
                                        )
                                    if dtype == "float32":
                                        T.copy(w_dt, w_f32)
                                    else:
                                        T.tile.cast(
                                            w_f32, w_dt, "CAST_NONE", oc_tile * w_stride
                                        )
                                    for tap in T.serial(taps):
                                        # Tap geometry is computed at RUNTIME (as
                                        # scalars) rather than by unrolling the
                                        # window in Python: TVMScript only accepts
                                        # `range` / `T.serial` style loops, and an
                                        # unrolled 27-tap rank-3 window would also
                                        # multiply the emitted code by 27.  Every
                                        # value here is a scalar arithmetic
                                        # expression; the COPY extents stay
                                        # compile-time constants, which is the only
                                        # thing the DMA needs.
                                        k0 = tap // (kernel[1] * kernel[2])
                                        k1 = (tap // kernel[2]) % kernel[1]
                                        k2 = tap % kernel[2]
                                        deep = k0 * dilation[0] - padding[0]
                                        outer = k1 * dilation[1] - padding[1]
                                        inner = k2 * dilation[2] - padding[2]
                                        delta = (
                                            (deep * spatial_in[1] + outer)
                                            * spatial_in[2]
                                            + inner
                                        )
                                        src = (
                                            x_base
                                            + ci_local * in_spatial
                                            + start
                                            + delta
                                        )
                                        lo = T.if_then_else(inner < 0, -inner, 0)
                                        hi = T.if_then_else(inner > 0, inner, 0)
                                        T.tile.fill(stage_dt, 0.0)
                                        # ONE copy for the whole tile.  Consecutive output
                                        # rows map to consecutive input rows -- that is what
                                        # the extent-preserving admission buys -- so the
                                        # tile's source is a single contiguous run.  Issuing
                                        # one copy PER ROW instead was measured at 27x
                                        # slower on deeplabv3-aspp (R269 section 6): it
                                        # turned one 2 KiB DMA into 32 of 64 bytes.
                                        if src >= 0 and src + tile <= numel_x:
                                            T.copy(x_flat[src : src + tile], stage_dt)
                                        else:
                                            # Boundary tile: some sub-block of it is outside
                                            # the tensor.  Guard each 32-byte-aligned
                                            # sub-block, and fall back to per-lane moves only
                                            # for the ones that really are out.  Those lanes
                                            # are all zeroed below anyway; this branch exists
                                            # so the kernel never dereferences outside the
                                            # tensor.
                                            for grp in T.serial(groups_per_tile):
                                                gsrc = src + grp * group_lanes
                                                if (gsrc >= 0
                                                        and gsrc + group_lanes <= numel_x):
                                                    T.copy(
                                                        x_flat[gsrc : gsrc + group_lanes],
                                                        stage_dt[
                                                            grp * group_lanes
                                                            : grp * group_lanes
                                                            + group_lanes
                                                        ],
                                                    )
                                                else:
                                                    for lane in T.serial(group_lanes):
                                                        if (gsrc + lane >= 0
                                                                and gsrc + lane < numel_x):
                                                            stage_dt[
                                                                grp * group_lanes + lane
                                                            ] = x_flat[gsrc + lane]
                                        if dtype == "float32":
                                            T.copy(stage_dt, stage_f32)
                                        else:
                                            T.tile.cast(
                                                stage_f32, stage_dt, "CAST_NONE", tile
                                            )
                                        for r in T.serial(rows):
                                            for col in T.serial(lo):
                                                stage_f32[r * row_len + col] = 0.0
                                            for col in T.serial(hi):
                                                stage_f32[
                                                    r * row_len + row_len - 1 - col
                                                ] = 0.0
                                            if (coord_ub[r] + outer < 0
                                                    or coord_ub[r] + outer
                                                    >= spatial_out[1]
                                                    or coord_ub[rows + r] + deep < 0
                                                    or coord_ub[rows + r] + deep
                                                    >= spatial_out[0]):
                                                for col in T.serial(row_len):
                                                    stage_f32[
                                                        r * row_len + col
                                                    ] = 0.0
                                        for oc_lane in T.serial(oc_tile):
                                            T.tile.axpy(
                                                acc[oc_lane, 0:tile],
                                                stage_f32,
                                                w_f32[oc_lane, tap],
                                            )
                            for oc_lane in T.serial(oc_tile):
                                if has_bias:
                                    T.tile.add(
                                        acc[oc_lane, 0:tile],
                                        acc[oc_lane, 0:tile],
                                        bias_f32[oc_lane],
                                    )
                                obase = (
                                    (batch_id * c_out + oc_base + oc_lane)
                                    * out_spatial
                                    + start
                                )
                                if dtype == "float32":
                                    T.copy(
                                        acc[oc_lane, 0:tile],
                                        output_flat[obase : obase + tile],
                                    )
                                else:
                                    T.tile.cast(
                                        out_dt, acc[oc_lane, 0:tile],
                                        "CAST_RINT", tile,
                                    )
                                    T.copy(
                                        out_dt, output_flat[obase : obase + tile]
                                    )

        out_shape = (n, c_out, *spatial_out)
        w_shape = (c_out, c_in_g, *kernel)
        x_shape = (n, c_in, *spatial_in)
        if has_bias:

            @T.prim_func
            def main_bias(
                x: T.Tensor(x_shape, dtype),
                weight: T.Tensor(w_shape, dtype),
                bias: T.Tensor((c_out,), dtype),
                output: T.Tensor(out_shape, dtype),
            ):
                compute(x, weight, bias, output)

            return main_bias

        @T.prim_func
        def main(
            x: T.Tensor(x_shape, dtype),
            weight: T.Tensor(w_shape, dtype),
            output: T.Tensor(out_shape, dtype),
        ):
            compute(x, weight, None, output)

        return main

    return factory(dtype_name)


def _try_vector_window(
    n, c_in, spatial_in, c_out, c_in_g, kernel, stride, padding, dilation,
    groups, spatial_out, dtype_name, has_bias, itemsize,
):
    """Build the vectorised window kernel, or return ``None`` if it refuses."""
    k_total = c_in_g
    for extent in kernel:
        k_total *= extent
    plan = _vector_window_plan(
        n, groups, spatial_in, spatial_out, kernel, stride, padding, dilation,
        c_in_g, c_out // groups, k_total, itemsize,
    )
    if plan is None:
        return None
    compiled = _compile_vector_conv(
        n, c_in, spatial_in, c_out, c_in_g, kernel, stride, padding, dilation,
        groups, spatial_out, dtype_name, bool(has_bias),
        tuple(sorted(plan.items())),
    )
    out_spatial = 1
    for extent in spatial_out:
        out_spatial *= extent
    items = (n * groups * ((c_out // groups) // plan["oc_tile"])
             * plan["chunks"])
    logical_blocks = -(-items // _VEC)

    def launch(x, weight, bias=None):
        return compiled(x, weight, bias) if has_bias else compiled(x, weight)

    launch.path_kind = "vector_window"
    launch.output_shape = (n, c_out, *spatial_out)
    launch.workspace_shape = None
    launch.logical_blocks = logical_blocks
    launch.launch_blocks = launch_block_count(logical_blocks)
    launch.grid_repeats = grid_repeat_count(logical_blocks, launch.launch_blocks)
    launch.vector_plan = plan
    launch.compiled = compiled
    return launch


def _is_pointwise_gemm(dtype_name, groups, has_bias, kernel, stride, padding,
                       dilation):
    """Is this Conv2d exactly a matrix product, with no im2col to do at all?

    A 1x1 dense stride-1 unpadded convolution is ``weight[C_out, C_in] @
    x[C_in, H*W]`` per batch element: the implicit-im2col tile IS the input
    tile, so every scalar address the AIV im2col computes is the identity.
    That gather is where the cube path spends its time -- measured, on
    ``resnet-3x3``/fp16, 20 reps at Level1 ``PipeUtilization``
    (R268-data/logs/02-pipes-resnet3x3.log):

        main_kernel  Duration 5832 us
          aiv_scalar_time 5808 us   aiv_scalar_ratio 1.000
          aic_mac_time       6.1 us aic_mac_ratio    0.001
          aic_mte2_time     22.7 us

    i.e. 99.6% of the kernel is the AIV scalar pipe and the Cube is idle.  So
    for the shapes where the gather is provably redundant, route to the dense
    GEMM kernel instead, which since T264 runs T.mma behind an asynchronous
    GM->L1 stage (see kernels/gemm.py and docs/reports/R264.md section 3).
    """
    return (
        dtype_name != "float32"
        and not has_bias
        and groups == 1
        and kernel == (1, 1)
        and stride == (1, 1)
        and padding == (0, 0)
        and dilation == (1, 1)
    )


def _build_pointwise_gemm_conv2d(n, c_in, h_in, w_in, c_out, dtype, out_h, out_w):
    """1x1 dense Conv2d as one dense GEMM per batch element.

    ``weight`` is [C_out, C_in, 1, 1] and each ``x[i]`` is [C_in, H, W]
    contiguous, so both reshapes below are views: no data is moved to set the
    GEMM up.  ``out_h, out_w == h_in, w_in`` holds because the guard in
    :func:`_is_pointwise_gemm` pins stride, padding and dilation.

    The batch is a loop rather than a batched GEMM because the weight operand
    is SHARED across the batch: feeding it to ``build_bmm_kernel`` would need a
    materialised [N, C_out, C_in] copy of it.  N is small in practice (the
    Conv2dFwdOp manifest workloads use 1 or 2) and every launch is counted by
    the harness's interval union, so the loop is not hidden from the timer.
    """
    out_hw = out_h * out_w
    gemm = build_gemm_kernel(
        (c_out, c_in), (c_in, out_hw), dtype, trans_a=False, trans_b=False
    )

    def launch(x, weight, bias=None):
        if bias is not None:
            raise ValueError("pointwise GEMM Conv2d path does not take bias")
        flat_weight = weight.reshape(c_out, c_in)
        if n == 1:
            product = gemm(flat_weight, x.reshape(c_in, out_hw))
            return product.reshape(1, c_out, out_h, out_w)
        return torch.stack(
            [gemm(flat_weight, x[index].reshape(c_in, out_hw)) for index in range(n)]
        ).reshape(n, c_out, out_h, out_w)

    launch.path_kind = "pointwise_gemm"
    launch.output_shape = (n, c_out, out_h, out_w)
    launch.workspace_shape = None
    launch.launches_per_call = n
    launch.gemm_shape = (c_out, out_hw, c_in)
    # The grid is the GEMM kernel's, not this wrapper's; report it so that the
    # same introspection the other two paths expose keeps working.
    plan = _gemm_pipelined_plan(c_out, out_hw, c_in, 2)
    if plan is None:
        launch.logical_blocks = None
    else:
        launch.logical_blocks = -(-c_out // plan[0]) * -(-out_hw // plan[1])
    launch.launch_blocks = (
        None if launch.logical_blocks is None
        else launch_block_count(launch.logical_blocks)
    )
    launch.grid_repeats = (
        None if launch.logical_blocks is None
        else grid_repeat_count(launch.logical_blocks, launch.launch_blocks)
    )
    launch.compiled = gemm
    return launch


def build_conv2d_kernel(
    input_shape,
    weight_shape,
    dtype,
    *,
    stride,
    padding,
    dilation,
    groups,
    has_bias,
):
    """Build a complete-domain NCHW Conv2d callable."""
    n, c_in, h_in, w_in = (int(value) for value in input_shape)
    c_out, c_in_g, kernel_h, kernel_w = (int(value) for value in weight_shape)
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    out_h = (h_in + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (w_in + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    dtype_name = _dtype_name(dtype)
    c_out_g = c_out // groups
    # Bias must not decide which convolution ALGORITHM runs.  Before T268 it
    # did: the pointwise-GEMM and implicit-im2col-cube paths both refuse a bias
    # because neither has an epilogue, so every bias-carrying workload fell
    # into the scalar-gather direct kernel -- measured at ~100x the identical
    # bias-free shape, and the whole of Conv2d's ratio_min (R269 section 2).
    # So: ask what the geometry alone would choose (`has_bias=False` below),
    # and give bias to whichever path can serve it -- natively for the
    # vectorised window, and as one extra vector pass for the other two.
    pointwise = _is_pointwise_gemm(
        dtype_name, groups, False,
        (kernel_h, kernel_w), (stride_h, stride_w), (pad_h, pad_w),
        (dilation_h, dilation_w),
    )
    cube_ok = dtype_name != "float32" and c_out_g % _BLOCK_OC == 0
    if pointwise:
        base = _build_pointwise_gemm_conv2d(
            n, c_in, h_in, w_in, c_out, dtype, out_h, out_w
        )
        if not has_bias:
            return base
        return _build_bias_epilogue(
            base, n, c_out, (n, c_out, out_h, out_w), dtype
        )
    vector = _try_vector_window(
        n, c_in, (h_in, w_in), c_out, c_in_g,
        (kernel_h, kernel_w), (stride_h, stride_w), (pad_h, pad_w),
        (dilation_h, dilation_w), groups, (out_h, out_w), dtype_name,
        has_bias, torch.empty((), dtype=dtype).element_size(),
    )
    if vector is not None:
        if not cube_ok:
            # No cube path for this shape at all (grouped so that
            # c_out / groups is not a multiple of the cube's output-channel
            # tile, or float32).  The alternative is the scalar-gather kernel,
            # which R269 section 2 measured at aiv_scalar_ratio = 0.99 -- there
            # is nothing to weigh up.
            return vector
        cube_cost = _cube_window_cost(
            n, groups, c_out_g, out_h * out_w,
            c_in_g * kernel_h * kernel_w,
        )
        # Both paths exist, so choose -- but only move a shape OFF the cube
        # path when the model predicts a wide win, because the two models are
        # calibrated to different measurements and each is good to a factor of
        # ~2-3 (R269 section 6 tabulates predicted against measured for the
        # four shapes where BOTH paths were measured).  At margin 4 the model
        # reproduces all four measured verdicts, with the nearest case a factor
        # of 2.3 clear of the threshold.
        if vector.vector_plan["model_cost"] * _VW_ROUTE_MARGIN <= cube_cost:
            return vector
    if has_bias and cube_ok:
        base = build_conv2d_kernel(
            input_shape, weight_shape, dtype,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, has_bias=False,
        )
        return _build_bias_epilogue(
            base, n, c_out, (n, c_out, out_h, out_w), dtype
        )
    use_cube = cube_ok and not has_bias
    cube_logical_blocks = (
        n
        * groups
        * math.ceil(c_out_g / _BLOCK_OC)
        * math.ceil((out_h * out_w) / _BLOCK_SPATIAL)
    )
    cube_k_tiles = math.ceil((c_in_g * kernel_h * kernel_w) / _BLOCK_K)
    compiler = _compile_cube_conv2d if use_cube else _compile_direct_conv2d
    compiled = compiler(
        n,
        c_in,
        h_in,
        w_in,
        c_out,
        c_in_g,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        out_h,
        out_w,
        dtype_name,
        bool(has_bias),
    )

    if has_bias:

        def launch(x, weight, bias):
            return compiled(x, weight, bias)

    elif use_cube:

        def launch(x, weight, bias=None):
            workspace = torch.empty(
                (cube_logical_blocks, cube_k_tiles, _BLOCK_K, _BLOCK_SPATIAL),
                dtype=x.dtype,
                device=x.device,
            )
            return compiled(x, weight, workspace)

    else:

        def launch(x, weight, bias=None):
            return compiled(x, weight)

    launch.path_kind = "implicit_im2col_cube" if use_cube else "direct_window"
    launch.output_shape = (n, c_out, out_h, out_w)
    launch.workspace_shape = (
        (cube_logical_blocks, cube_k_tiles, _BLOCK_K, _BLOCK_SPATIAL)
        if use_cube
        else None
    )
    if use_cube:
        logical_blocks = (
            n
            * groups
            * math.ceil(c_out_g / _BLOCK_OC)
            * math.ceil((out_h * out_w) / _BLOCK_SPATIAL)
        )
    else:
        logical_blocks = math.ceil((n * c_out * out_h * out_w) / (_VEC * _DIRECT_TILE))
    launch.logical_blocks = logical_blocks
    launch.launch_blocks = launch_block_count(logical_blocks)
    launch.grid_repeats = grid_repeat_count(logical_blocks, launch.launch_blocks)
    launch.compiled = compiled
    return launch


@lru_cache(maxsize=128)
def _compile_direct_conv_nd(
    n: int,
    c_in: int,
    spatial_in: tuple[int, ...],
    c_out: int,
    c_in_g: int,
    kernel: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    spatial_out: tuple[int, ...],
    dtype_name: str,
    has_bias: bool,
):
    """Generic rank-1/rank-3 direct window kernel.

    The indexing is shared by Conv1d and Conv3d; only the statically-known
    spatial rank changes the tensor ABI.  This keeps Conv1d a one-axis
    degeneration of the same convolution implementation instead of creating
    a second algorithm.
    """
    ndim = len(spatial_in)
    if ndim not in (1, 3):
        raise ValueError("generic convolution supports rank 1 or 3 spatial domains")
    c_out_g = c_out // groups
    in_spatial = math.prod(spatial_in)
    out_spatial = math.prod(spatial_out)
    k_total = c_in_g * math.prod(kernel)
    total = n * c_out * out_spatial
    logical_blocks = math.ceil(total / (_VEC * _DIRECT_TILE))
    launch_blocks = launch_block_count(logical_blocks)
    grid_repeats = grid_repeat_count(logical_blocks, launch_blocks)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def factory(dtype: str = dtype_name):
        @T.macro
        def compute(x, weight, bias, output):
            x_flat = T.Tensor((n * c_in * in_spatial,), dtype, x.data)
            weight_flat = T.Tensor((c_out * k_total,), dtype, weight.data)
            output_flat = T.Tensor((total,), dtype, output.data)
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                result_fp32 = T.alloc_ub((_DIRECT_TILE,), "float32")
                result = T.alloc_ub((_DIRECT_TILE,), dtype)
                # dav-2201 has no scalar bf16 cast instruction in either
                # direction, so nothing may cross the dtype boundary one element
                # at a time.  Values are gathered with plain scalar moves into a
                # tile of the public dtype and widened a whole tile at a time.
                gather_dt = T.alloc_ub((_DIRECT_TILE,), dtype)
                gather_fp32 = T.alloc_ub((_DIRECT_TILE,), "float32")
                weight_dt = T.alloc_ub((_DIRECT_TILE,), dtype)
                weight_fp32 = T.alloc_ub((_DIRECT_TILE,), "float32")
                prod_fp32 = T.alloc_ub((_DIRECT_TILE,), "float32")
                # Output coordinates are decoded once per tile and reused by
                # every window position instead of being recomputed per window.
                # One flat buffer holds axis a of lane l at a * _DIRECT_TILE + l.
                coord_ub = T.alloc_ub((_DIRECT_TILE * ndim,), "int32")
                x_base_ub = T.alloc_ub((_DIRECT_TILE,), "int32")
                w_base_ub = T.alloc_ub((_DIRECT_TILE,), "int32")
                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < logical_blocks:
                            start = (
                                logical_cid * (_VEC * _DIRECT_TILE) + vid * _DIRECT_TILE
                            )
                            T.tile.fill(result_fp32, 0.0)
                            # Lanes past the end never write weight_dt; zero it
                            # once so their product stays a finite zero.
                            T.tile.fill(weight_dt, 0.0)
                            for lane in T.serial(_DIRECT_TILE):
                                output_index = start + lane
                                if output_index < total:
                                    spatial_index = output_index % out_spatial
                                    channel_batch = output_index // out_spatial
                                    oc = channel_batch % c_out
                                    batch_id = channel_batch // c_out
                                    group_id = oc // c_out_g
                                    x_base_ub[lane] = (
                                        batch_id * c_in + group_id * c_in_g
                                    ) * in_spatial
                                    w_base_ub[lane] = oc * k_total
                                    # Decode output coordinates.
                                    if ndim == 1:
                                        coord_ub[lane] = spatial_index
                                    else:
                                        coord_ub[2 * _DIRECT_TILE + lane] = (
                                            spatial_index % spatial_out[2]
                                        )
                                        tmp = spatial_index // spatial_out[2]
                                        coord_ub[_DIRECT_TILE + lane] = (
                                            tmp % spatial_out[1]
                                        )
                                        coord_ub[lane] = tmp // spatial_out[1]
                            # Iterate the kernel window; one lane-parallel tile
                            # per window position.
                            if ndim == 1:
                                for ci_local in T.serial(c_in_g):
                                    for k0 in T.serial(kernel[0]):
                                        weight_offset = ci_local * kernel[0] + k0
                                        T.tile.fill(gather_dt, 0.0)
                                        for lane in T.serial(_DIRECT_TILE):
                                            if start + lane < total:
                                                weight_dt[lane] = weight_flat[
                                                    w_base_ub[lane] + weight_offset
                                                ]
                                                i0 = (
                                                    coord_ub[lane] * stride[0]
                                                    - padding[0]
                                                    + k0 * dilation[0]
                                                )
                                                if i0 >= 0 and i0 < spatial_in[0]:
                                                    gather_dt[lane] = x_flat[
                                                        x_base_ub[lane]
                                                        + ci_local * in_spatial
                                                        + i0
                                                    ]
                                        if dtype == "float32":
                                            T.copy(gather_dt, gather_fp32)
                                            T.copy(weight_dt, weight_fp32)
                                        else:
                                            T.tile.cast(
                                                gather_fp32,
                                                gather_dt,
                                                "CAST_NONE",
                                                _DIRECT_TILE,
                                            )
                                            T.tile.cast(
                                                weight_fp32,
                                                weight_dt,
                                                "CAST_NONE",
                                                _DIRECT_TILE,
                                            )
                                        T.tile.mul(
                                            prod_fp32, gather_fp32, weight_fp32
                                        )
                                        T.tile.add(
                                            result_fp32, result_fp32, prod_fp32
                                        )
                            else:
                                for ci_local in T.serial(c_in_g):
                                    for k0 in T.serial(kernel[0]):
                                        for k1 in T.serial(kernel[1]):
                                            for k2 in T.serial(kernel[2]):
                                                weight_offset = (
                                                    (ci_local * kernel[0] + k0)
                                                    * kernel[1]
                                                    + k1
                                                ) * kernel[2] + k2
                                                T.tile.fill(gather_dt, 0.0)
                                                for lane in T.serial(_DIRECT_TILE):
                                                    if start + lane < total:
                                                        weight_dt[lane] = weight_flat[
                                                            w_base_ub[lane]
                                                            + weight_offset
                                                        ]
                                                        i0 = (
                                                            coord_ub[lane] * stride[0]
                                                            - padding[0]
                                                            + k0 * dilation[0]
                                                        )
                                                        i1 = (
                                                            coord_ub[
                                                                _DIRECT_TILE + lane
                                                            ]
                                                            * stride[1]
                                                            - padding[1]
                                                            + k1 * dilation[1]
                                                        )
                                                        i2 = (
                                                            coord_ub[
                                                                2 * _DIRECT_TILE + lane
                                                            ]
                                                            * stride[2]
                                                            - padding[2]
                                                            + k2 * dilation[2]
                                                        )
                                                        if (
                                                            i0 >= 0
                                                            and i0 < spatial_in[0]
                                                            and i1 >= 0
                                                            and i1 < spatial_in[1]
                                                            and i2 >= 0
                                                            and i2 < spatial_in[2]
                                                        ):
                                                            gather_dt[lane] = x_flat[
                                                                x_base_ub[lane]
                                                                + ci_local * in_spatial
                                                                + (
                                                                    i0 * spatial_in[1]
                                                                    + i1
                                                                )
                                                                * spatial_in[2]
                                                                + i2
                                                            ]
                                                if dtype == "float32":
                                                    T.copy(gather_dt, gather_fp32)
                                                    T.copy(weight_dt, weight_fp32)
                                                else:
                                                    T.tile.cast(
                                                        gather_fp32,
                                                        gather_dt,
                                                        "CAST_NONE",
                                                        _DIRECT_TILE,
                                                    )
                                                    T.tile.cast(
                                                        weight_fp32,
                                                        weight_dt,
                                                        "CAST_NONE",
                                                        _DIRECT_TILE,
                                                    )
                                                T.tile.mul(
                                                    prod_fp32,
                                                    gather_fp32,
                                                    weight_fp32,
                                                )
                                                T.tile.add(
                                                    result_fp32,
                                                    result_fp32,
                                                    prod_fp32,
                                                )
                            if has_bias:
                                # Bias stays the last term of the accumulation,
                                # matching the scalar implementation it replaces.
                                T.tile.fill(gather_dt, 0.0)
                                for lane in T.serial(_DIRECT_TILE):
                                    output_index = start + lane
                                    if output_index < total:
                                        gather_dt[lane] = bias[
                                            (output_index // out_spatial) % c_out
                                        ]
                                if dtype == "float32":
                                    T.copy(gather_dt, gather_fp32)
                                else:
                                    T.tile.cast(
                                        gather_fp32,
                                        gather_dt,
                                        "CAST_NONE",
                                        _DIRECT_TILE,
                                    )
                                T.tile.add(result_fp32, result_fp32, gather_fp32)
                            if dtype == "float32":
                                T.copy(
                                    result_fp32,
                                    output_flat[start : start + _DIRECT_TILE],
                                )
                            else:
                                T.tile.cast(
                                    result, result_fp32, "CAST_RINT", _DIRECT_TILE
                                )
                                T.copy(
                                    result, output_flat[start : start + _DIRECT_TILE]
                                )

        if ndim == 1:
            if has_bias:

                @T.prim_func
                def main_bias(
                    x: T.Tensor((n, c_in, spatial_in[0]), dtype),
                    weight: T.Tensor((c_out, c_in_g, kernel[0]), dtype),
                    bias: T.Tensor((c_out,), dtype),
                    output: T.Tensor((n, c_out, spatial_out[0]), dtype),
                ):
                    compute(x, weight, bias, output)

                return main_bias

            @T.prim_func
            def main(
                x: T.Tensor((n, c_in, spatial_in[0]), dtype),
                weight: T.Tensor((c_out, c_in_g, kernel[0]), dtype),
                output: T.Tensor((n, c_out, spatial_out[0]), dtype),
            ):
                compute(x, weight, None, output)

            return main
        if has_bias:

            @T.prim_func
            def main_bias(
                x: T.Tensor(
                    (n, c_in, spatial_in[0], spatial_in[1], spatial_in[2]), dtype
                ),
                weight: T.Tensor(
                    (c_out, c_in_g, kernel[0], kernel[1], kernel[2]), dtype
                ),
                bias: T.Tensor((c_out,), dtype),
                output: T.Tensor(
                    (n, c_out, spatial_out[0], spatial_out[1], spatial_out[2]), dtype
                ),
            ):
                compute(x, weight, bias, output)

            return main_bias

        @T.prim_func
        def main(
            x: T.Tensor((n, c_in, spatial_in[0], spatial_in[1], spatial_in[2]), dtype),
            weight: T.Tensor((c_out, c_in_g, kernel[0], kernel[1], kernel[2]), dtype),
            output: T.Tensor(
                (n, c_out, spatial_out[0], spatial_out[1], spatial_out[2]), dtype
            ),
        ):
            compute(x, weight, None, output)

        return main

    return factory(dtype_name)


def build_conv_nd_kernel(
    input_shape, weight_shape, dtype, *, stride, padding, dilation, groups, has_bias
):
    """Build Conv1d/Conv3d using the shared direct-window implementation."""
    n, c_in, *spatial_in = (int(value) for value in input_shape)
    c_out, c_in_g, *kernel = (int(value) for value in weight_shape)
    spatial_out = tuple(
        (size + 2 * pad - dil * (ks - 1) - 1) // step + 1
        for size, ks, step, pad, dil in zip(
            spatial_in, kernel, stride, padding, dilation, strict=True
        )
    )
    dtype_name = _dtype_name(dtype)
    vector = _try_vector_window(
        n, c_in, tuple(spatial_in), c_out, c_in_g, tuple(kernel),
        tuple(stride), tuple(padding), tuple(dilation), groups, spatial_out,
        dtype_name, has_bias, torch.empty((), dtype=dtype).element_size(),
    )
    if vector is not None:
        return vector
    compiled = _compile_direct_conv_nd(
        n,
        c_in,
        tuple(spatial_in),
        c_out,
        c_in_g,
        tuple(kernel),
        tuple(stride),
        tuple(padding),
        tuple(dilation),
        groups,
        spatial_out,
        dtype_name,
        bool(has_bias),
    )

    def launch(x, weight, bias=None):
        return compiled(x, weight, bias) if has_bias else compiled(x, weight)

    launch.path_kind = "direct_window"
    launch.output_shape = (n, c_out, *spatial_out)
    launch.logical_blocks = math.ceil(
        (n * c_out * math.prod(spatial_out)) / (_VEC * _DIRECT_TILE)
    )
    launch.launch_blocks = launch_block_count(launch.logical_blocks)
    launch.grid_repeats = grid_repeat_count(launch.logical_blocks, launch.launch_blocks)
    launch.compiled = compiled
    return launch


__all__ = ["build_conv2d_kernel", "build_conv_nd_kernel"]
