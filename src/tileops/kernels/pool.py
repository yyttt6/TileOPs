"""Ascend AIV max-pool window-gather template.

R197 added a fast path in front of both original paths: when the working set
fits in UB, `MaxPool2dFwdOp` is dispatched to `pool_avg`'s vector-gather
template (a fixed 2D max pool is the ndim=2 case of it), which removes the
per-lane scalar loop that had this operator at 1.75-6.2 GB/s.  The two original
paths below stay as the fallback.


The common NCHW path assigns one complete output row to each vector context.
It loads the row's horizontal receptive field into UB once and reuses those
values for overlapping windows.  A flattened fallback preserves the complete
manifest parameter domain when a row tile would exceed UB capacity.

Both paths write fixed-width vectors to GM.  This avoids scalar GM stores,
which auto-CV-combine can silently place under an impossible AIC-and-AIV guard
on dav-2201.
"""

from functools import lru_cache
import math

import tilelang
import tilelang.language as T
import torch

from .common import grid_repeat_count, launch_block_count


_VEC = 2
_STORE_TILE = 64
_MAX_REUSE_ELEMENTS = 16_384


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    raise TypeError(
        "MaxPool2dFwdOp supports float16, bfloat16, and float32, " f"got {dtype}"
    )


def _pair(name: str, value) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        result = (value, value)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        result = (value[0], value[1])
    else:
        raise TypeError(
            f"{name} must be an int or a length-2 tuple/list, got {value!r}"
        )
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in result):
        raise TypeError(f"{name} entries must be ints, got {value!r}")
    return int(result[0]), int(result[1])


def _output_dim(
    input_size: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool,
) -> int:
    effective = dilation * (kernel - 1) + 1
    numerator = input_size + 2 * padding - effective
    if ceil_mode:
        numerator += stride - 1
    output = numerator // stride + 1
    if ceil_mode and output > 0 and (output - 1) * stride >= input_size + padding:
        output -= 1
    return max(output, 0)


def _normalize_params(kernel_size, stride, padding, dilation, ceil_mode):
    kernel_h, kernel_w = _pair("kernel_size", kernel_size)
    stride_h, stride_w = (
        (kernel_h, kernel_w) if stride is None else _pair("stride", stride)
    )
    pad_h, pad_w = _pair("padding", padding)
    dilation_h, dilation_w = _pair("dilation", dilation)
    if kernel_h <= 0 or kernel_w <= 0:
        raise ValueError("kernel_size entries must be positive")
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError("stride entries must be positive")
    if dilation_h <= 0 or dilation_w <= 0:
        raise ValueError("dilation entries must be positive")
    if not (0 <= pad_h <= kernel_h // 2 and 0 <= pad_w <= kernel_w // 2):
        raise ValueError(
            "padding must be non-negative and at most half the kernel size"
        )
    if not isinstance(ceil_mode, bool):
        raise TypeError(f"ceil_mode must be bool, got {type(ceil_mode).__name__}")
    return (
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    )


@lru_cache(maxsize=128)
def _compile_row_reuse(
    n: int,
    c: int,
    h_in: int,
    w_in: int,
    out_h: int,
    out_w: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    dtype_name: str,
):
    rows = n * c * out_h
    blocks = math.ceil(rows / _VEC)
    launch_blocks = launch_block_count(blocks)
    grid_repeats = grid_repeat_count(blocks, launch_blocks)
    source_width = math.ceil(w_in / 16) * 16
    source_elems = kernel_h * source_width

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((n, c, h_in, w_in), dtype_name),
            C: T.Tensor((rows, out_w), dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                source = T.alloc_ub((kernel_h, source_width), dtype_name)
                source_fp32 = T.alloc_ub((kernel_h, source_width), "float32")
                result_fp32 = T.alloc_ub((_STORE_TILE,), "float32")
                result = T.alloc_ub((_STORE_TILE,), dtype_name)

                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < blocks:
                            # Required ownership formula uses both logical_cid and vid.
                            row = logical_cid * _VEC + vid
                            T.tile.fill(source, 0)
                            T.tile.fill(result_fp32, 0.0)
                            if row < rows:
                                oh = row % out_h
                                channel_batch = row // out_h
                                channel = channel_batch % c
                                batch = channel_batch // c

                                # Each valid input row is loaded once. Horizontally
                                # overlapping output windows reuse this UB stripe.
                                for kh in T.serial(kernel_h):
                                    ih = oh * stride_h - pad_h + kh * dilation_h
                                    if ih >= 0 and ih < h_in:
                                        T.copy(
                                            A[batch, channel, ih, :], source[kh, 0:w_in]
                                        )

                                if dtype_name == "float32":
                                    T.copy(source, source_fp32)
                                else:
                                    T.tile.cast(
                                        source_fp32, source, "CAST_NONE", source_elems
                                    )

                                for ow in T.serial(out_w):
                                    best = T.alloc_var("float32")
                                    has_nan = T.alloc_var("bool")
                                    best = -T.infinity("float32")
                                    has_nan = False
                                    for kh in T.serial(kernel_h):
                                        ih = oh * stride_h - pad_h + kh * dilation_h
                                        for kw in T.serial(kernel_w):
                                            iw = ow * stride_w - pad_w + kw * dilation_w
                                            if (
                                                ih >= 0
                                                and ih < h_in
                                                and iw >= 0
                                                and iw < w_in
                                            ):
                                                value = source_fp32[kh, iw]
                                                if T.isnan(value):
                                                    has_nan = True
                                                    best = value
                                                elif not has_nan:
                                                    best = T.max(best, value)
                                    result_fp32[ow] = best

                                if dtype_name == "float32":
                                    T.copy(result_fp32[0:out_w], C[row, :])
                                else:
                                    T.tile.cast(
                                        result, result_fp32, "CAST_RINT", _STORE_TILE
                                    )
                                    T.copy(result[0:out_w], C[row, :])

        return main

    return factory()


@lru_cache(maxsize=128)
def _compile_generic(
    n: int,
    c: int,
    h_in: int,
    w_in: int,
    out_h: int,
    out_w: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    dtype_name: str,
):
    total = n * c * out_h * out_w
    blocks = math.ceil(total / (_VEC * _STORE_TILE))
    launch_blocks = launch_block_count(blocks)
    grid_repeats = grid_repeat_count(blocks, launch_blocks)
    padded_total = blocks * _VEC * _STORE_TILE

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
    )
    def factory():
        @T.prim_func
        def main(
            A: T.Tensor((n, c, h_in, w_in), dtype_name),
            C: T.Tensor((padded_total,), dtype_name),
        ):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                result_fp32 = T.alloc_ub((_STORE_TILE,), "float32")
                result = T.alloc_ub((_STORE_TILE,), dtype_name)

                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < blocks:
                            # Required ownership formula uses both logical_cid and vid.
                            start = (
                                logical_cid * (_VEC * _STORE_TILE) + vid * _STORE_TILE
                            )
                            T.tile.fill(result_fp32, 0.0)
                            for lane in T.serial(_STORE_TILE):
                                output_index = start + lane
                                if output_index < total:
                                    ow = output_index % out_w
                                    spatial = output_index // out_w
                                    oh = spatial % out_h
                                    channel_batch = spatial // out_h
                                    channel = channel_batch % c
                                    batch = channel_batch // c
                                    best = T.alloc_var("float32")
                                    has_nan = T.alloc_var("bool")
                                    best = -T.infinity("float32")
                                    has_nan = False
                                    for kh in T.serial(kernel_h):
                                        for kw in T.serial(kernel_w):
                                            ih = oh * stride_h - pad_h + kh * dilation_h
                                            iw = ow * stride_w - pad_w + kw * dilation_w
                                            if (
                                                ih >= 0
                                                and ih < h_in
                                                and iw >= 0
                                                and iw < w_in
                                            ):
                                                value = T.cast(
                                                    A[batch, channel, ih, iw], "float32"
                                                )
                                                if T.isnan(value):
                                                    has_nan = True
                                                    best = value
                                                elif not has_nan:
                                                    best = T.max(best, value)
                                    result_fp32[lane] = best

                            if dtype_name == "float32":
                                T.copy(result_fp32, C[start : start + _STORE_TILE])
                            else:
                                T.tile.cast(
                                    result, result_fp32, "CAST_RINT", _STORE_TILE
                                )
                                T.copy(result, C[start : start + _STORE_TILE])

        return main

    return factory()


def build_max_pool2d_kernel(
    input_shape,
    dtype,
    *,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    """Return a callable implementing the manifest's value-only max-pool2d."""
    if len(input_shape) != 4:
        raise ValueError(f"MaxPool2dFwdOp expects a 4D NCHW input, got {input_shape}")
    n, c, h_in, w_in = (int(value) for value in input_shape)
    if min(n, c, h_in, w_in) <= 0:
        raise ValueError(
            f"MaxPool2dFwdOp requires non-empty dimensions, got {input_shape}"
        )
    (
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = _normalize_params(kernel_size, stride, padding, dilation, ceil_mode)
    out_h = _output_dim(h_in, kernel_h, stride_h, pad_h, dilation_h, ceil_mode)
    out_w = _output_dim(w_in, kernel_w, stride_w, pad_w, dilation_w, ceil_mode)
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            "MaxPool2dFwdOp calculated output size must be greater than zero, "
            f"got {(out_h, out_w)}"
        )

    dtype_name = _dtype_name(dtype)

    # --- fast path: one vector gather per window tap (R197) -------------------
    # MaxPool2d is a fixed-window max pool, i.e. exactly what pool_avg's gather
    # template already computes with ndim=2; it is imported lazily because
    # pool_avg imports `_dtype_name`/`_output_dim` from this module.
    from .pool_avg import _GatherGeometry, _compile_gather_pool

    in3 = (1, h_in, w_in)
    out3 = (1, out_h, out_w)
    k3 = (1, kernel_h, kernel_w)
    s3 = (1, stride_h, stride_w)
    p3 = (0, pad_h, pad_w)
    d3 = (1, dilation_h, dilation_w)
    geometry = _GatherGeometry(
        in3, out3, k3, s3, p3, d3, n * c, dtype_name, "max", False, None
    )
    if geometry.fits:
        compiled = _compile_gather_pool(
            in3, out3, k3, s3, p3, d3, n * c, dtype_name, "max", False, None
        )
        units, owp = geometry.UNITS, geometry.OWP

        def launch(x):
            raw = compiled(x.reshape(-1))
            return raw.view(units, out_h, owp)[:, :, :out_w].view(n, c, out_h, out_w)

        launch.path_kind = "max2d_gather"
        launch.output_shape = (n, c, out_h, out_w)
        launch.geometry = geometry
        launch.compiled = compiled
        return launch

    use_row_reuse = (
        16 <= out_w <= _STORE_TILE
        and kernel_h * math.ceil(w_in / 16) * 16 <= _MAX_REUSE_ELEMENTS
    )
    if use_row_reuse:
        compiled = _compile_row_reuse(
            n,
            c,
            h_in,
            w_in,
            out_h,
            out_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            dtype_name,
        )

        def launch(x):
            return compiled(x).view(n, c, out_h, out_w)

        launch.path_kind = "row_reuse"
    else:
        compiled = _compile_generic(
            n,
            c,
            h_in,
            w_in,
            out_h,
            out_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            dtype_name,
        )
        total = n * c * out_h * out_w

        def launch(x):
            return compiled(x)[:total].view(n, c, out_h, out_w)

        launch.path_kind = "generic"

    launch.output_shape = (n, c, out_h, out_w)
    launch.compiled = compiled
    return launch


__all__ = ["build_max_pool2d_kernel"]
