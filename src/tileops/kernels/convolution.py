"""Ascend convolution kernels with tile-local im2col and a direct fallback."""

from functools import lru_cache
import math

import tilelang
import tilelang.language as T
import torch

from .common import grid_repeat_count, launch_block_count


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
    use_cube = dtype_name != "float32" and c_out_g % _BLOCK_OC == 0 and not has_bias
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
