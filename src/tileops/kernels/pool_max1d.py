"""Ascend AIV value-only MaxPool1d kernel."""

from functools import lru_cache
import math

import tilelang
import tilelang.language as T

from .common import grid_repeat_count, launch_block_count
from .pool import _dtype_name, _output_dim


_VEC = 2
_TILE = 64


def _single(name, value):
    if isinstance(value, int) and not isinstance(value, bool):
        result = value
    elif isinstance(value, (tuple, list)) and len(value) == 1:
        result = value[0]
    else:
        raise TypeError(f"{name} must be an int or length-1 tuple/list, got {value!r}")
    if not isinstance(result, int) or isinstance(result, bool):
        raise TypeError(f"{name} must contain an int, got {value!r}")
    return int(result)


@lru_cache(maxsize=128)
def _compile_max_pool1d(n, c, length, out_l, kernel, stride, padding, dilation, dtype_name):
    total = n * c * out_l
    logical_blocks = math.ceil(total / (_VEC * _TILE))
    launch_blocks = launch_block_count(logical_blocks)
    repeats = grid_repeat_count(logical_blocks, launch_blocks)
    padded_total = logical_blocks * _VEC * _TILE

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
    )
    def factory():
        @T.prim_func
        def main(A: T.Tensor((n, c, length), dtype_name), C: T.Tensor((padded_total,), dtype_name)):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                gathered = T.alloc_ub((_TILE,), dtype_name)
                gathered_fp32 = T.alloc_ub((_TILE,), "float32")
                result_fp32 = T.alloc_ub((_TILE,), "float32")
                result = T.alloc_ub((_TILE,), dtype_name)
                with T.Scope("V"):
                    for repeat in T.serial(repeats):
                        logical_cid = cid + repeat * launch_blocks
                        if logical_cid < logical_blocks:
                            start = logical_cid * (_VEC * _TILE) + vid * _TILE
                            T.tile.fill(result_fp32, -T.infinity("float32"))
                            for kw in T.serial(kernel):
                                T.tile.fill(gathered, 0)
                                for lane in T.serial(_TILE):
                                    output_index = start + lane
                                    if output_index < total:
                                        ow = output_index % out_l
                                        channel_batch = output_index // out_l
                                        channel = channel_batch % c
                                        batch = channel_batch // c
                                        iw = ow * stride - padding + kw * dilation
                                        if iw >= 0 and iw < length:
                                            gathered[lane] = A[batch, channel, iw]
                                if dtype_name == "float32":
                                    T.copy(gathered, gathered_fp32)
                                else:
                                    T.tile.cast(gathered_fp32, gathered, "CAST_NONE", _TILE)
                                for lane in T.serial(_TILE):
                                    output_index = start + lane
                                    if output_index < total:
                                        ow = output_index % out_l
                                        iw = ow * stride - padding + kw * dilation
                                        if iw >= 0 and iw < length:
                                            value = gathered_fp32[lane]
                                            if T.isnan(value):
                                                result_fp32[lane] = value
                                            elif not T.isnan(result_fp32[lane]):
                                                result_fp32[lane] = T.max(result_fp32[lane], value)
                            if dtype_name == "float32":
                                T.copy(result_fp32, C[start : start + _TILE])
                            else:
                                T.tile.cast(result, result_fp32, "CAST_RINT", _TILE)
                                T.copy(result, C[start : start + _TILE])
        return main
    return factory()


def build_max_pool1d_kernel(input_shape, dtype, *, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    if len(input_shape) != 3:
        raise ValueError(f"MaxPool1dFwdOp expects a 3D NCL input, got {input_shape}")
    n, c, length = (int(v) for v in input_shape)
    if min(n, c, length) <= 0:
        raise ValueError(f"MaxPool1dFwdOp requires non-empty dimensions, got {input_shape}")
    kernel = _single("kernel_size", kernel_size)
    stride = kernel if stride is None else _single("stride", stride)
    padding = _single("padding", padding)
    dilation = _single("dilation", dilation)
    if kernel <= 0 or stride <= 0 or dilation <= 0:
        raise ValueError("kernel_size, stride, and dilation must be positive")
    if not 0 <= padding <= kernel // 2:
        raise ValueError("padding must be non-negative and at most half the kernel size")
    if not isinstance(ceil_mode, bool):
        raise TypeError("ceil_mode must be bool")
    out_l = _output_dim(length, kernel, stride, padding, dilation, ceil_mode)
    if out_l <= 0:
        raise ValueError("MaxPool1dFwdOp calculated output size must be greater than zero")
    dtype_name = _dtype_name(dtype)
    compiled = _compile_max_pool1d(n, c, length, out_l, kernel, stride, padding, dilation, dtype_name)
    total = n * c * out_l

    def launch(x):
        return compiled(x)[:total].view(n, c, out_l)

    launch.path_kind = "max1d_generic"
    launch.output_shape = (n, c, out_l)
    launch.compiled = compiled
    return launch


__all__ = ["build_max_pool1d_kernel"]
