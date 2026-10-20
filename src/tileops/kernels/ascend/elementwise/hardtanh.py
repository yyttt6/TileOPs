"""Ascend 950 implementations for HardtanhFwdOp."""

import functools
import math
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_vector_cores

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.call_spec import BoundedUnaryFwdInterface, BoundsCall
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["AscendHardtanhKernel"]
_UB_BUDGET_BYTES = 216 * 1024
_NUM_STAGES = 2
_FALLBACK_VECTOR_CORES = 64
_MAX_THREADS = 2048
_BOUNDED_CONV_FEAT_NUM_BLOCKS = 56
_TUNED_TILING = {
    (2048 * 4096, "float16"): (64, 16384, 3),
    (2048 * 4096, "bfloat16"): (64, 16384, 3),
    (16 * 256 * 56 * 56, "float16"): (56, 16384, 3),
    (16 * 256 * 56 * 56, "bfloat16"): (56, 16384, 3),
}


def _floor_pow2(n: int) -> int:
    """Largest power of two ``<= n`` (``n >= 1``)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _zero_sign_key(value: float) -> int:
    """``+1``/``-1`` sign of ``value``, an extra component of every builder's cache key."""
    return -1 if math.copysign(1.0, value) < 0 else 1


def _tie_ops(min_val: float, max_val: float) -> "tuple[str, str]":
    """Ordered-compare ops that reproduce the reference clamp's signed-zero tie-break."""
    lo_op = "le" if min_val == 0.0 and math.copysign(1.0, min_val) > 0.0 else "lt"
    hi_op = "ge" if max_val == 0.0 and math.copysign(1.0, max_val) < 0.0 else "gt"
    return (lo_op, hi_op)


@functools.lru_cache(maxsize=1)
def _calibrated_num_blocks(fallback: int = _FALLBACK_VECTOR_CORES) -> int:
    """Persistent-grid block count for the large, bandwidth-bound tiled path."""
    try:
        queried = get_num_vector_cores(device_id=0)
    except Exception:
        queried = None
    if not queried:
        return fallback
    return _floor_pow2(int(queried))


def _tile_elems_for_dtype(dtype_str: str) -> int:
    """The largest power-of-two element count keeping the tiled path's UB use in budget."""
    itemsize = getattr(torch, dtype_str).itemsize
    max_elems = _UB_BUDGET_BYTES // (2 * _NUM_STAGES * itemsize)
    return _floor_pow2(max_elems)


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


@T.macro
def _hardtanh_body(x_ub, y_ub, n: int, threads: int, min_val: float, max_val: float) -> None:
    """``y_ub[:n] = clamp(x_ub[:n], min_val, max_val)``, NaN-propagating."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n):
            lo = T.cast(min_val, "float32")
            hi = T.cast(max_val, "float32")
            wide = T.cast(x_ub[i], "float32")
            clamped = T.min(T.max(wide, lo), hi)
            y_ub[i] = T.Cast(y_ub.dtype, T.if_then_else(T.isnan(wide), wide, clamped))


@T.macro
def _hardtanh_body_vf(
    x_ub, y_ub, n: int, dtype_str: str, lo_op: str, hi_op: str, min_val: float, max_val: float
) -> None:
    """Explicit-SIMD (``T.SimdVF``) clamp body: fp32 ordered compare + select."""
    with T.SimdVF():
        lo32 = T.cast(min_val, "float32")
        hi32 = T.cast(max_val, "float32")
        lo_v = T.simd.vdup(lo32, "float32")
        hi_v = T.simd.vdup(hi32, "float32")
        for i in range(n // 64):
            if dtype_str == "float32":
                wide = T.simd.vld(x_ub[i * 64])
            else:
                wide = T.simd.vcvt(T.simd.vld(x_ub[i * 64], dist="UNPK_B16"), "float32", part=0)
            low = T.simd.vsel(lo_v, wide, T.simd.vcmps(wide, lo32, op=lo_op))
            high = T.simd.vsel(hi_v, low, T.simd.vcmps(low, hi32, op=hi_op))
            if dtype_str == "float32":
                T.simd.vsts(y_ub[i * 64], high)
            else:
                T.simd.vsts(y_ub[i * 64], T.simd.vcvt(high, dtype_str), dist="PK_B32")


@T.macro
def _hardtanh_body_native16(
    x_ub, y_ub, n: int, dtype_str: str, min_val: float, max_val: float
) -> None:
    """Native 128-lane 16-bit clamp for the default bounds ``(-1, +1)`` only."""
    with T.SimdVF():
        lo_s = T.cast(min_val, dtype_str)
        hi_s = T.cast(max_val, dtype_str)
        lo_v = T.simd.vdup(lo_s, dtype_str)
        hi_v = T.simd.vdup(hi_s, dtype_str)
        for i in range(n // 128):
            x = T.simd.vld(x_ub[i * 128])
            low = T.simd.vsel(lo_v, x, T.simd.vcmps(x, lo_s, op="lt"))
            high = T.simd.vsel(hi_v, low, T.simd.vcmps(low, hi_s, op="gt"))
            T.simd.vsts(y_ub[i * 128], high, dist="NORM_B16")


@functools.lru_cache(maxsize=64)
def _build_single_block(
    n_total: int, dtype_str: str, min_val: float, max_val: float, lo_sign: int, hi_sign: int
):
    """One block, one untiled GM->UB->compute->UB->GM pass over ``n_total`` elements."""
    min_val = math.copysign(min_val, lo_sign)
    max_val = math.copysign(max_val, hi_sign)

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((n_total,), dtype_str)
            y_ub = T.alloc_shared((n_total,), dtype_str)
            T.copy(X, x_ub)
            _hardtanh_body(
                x_ub,
                y_ub,
                n_total,
                threads=min(n_total, _MAX_THREADS),
                min_val=min_val,
                max_val=max_val,
            )
            T.copy(y_ub, Y)

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _build_tiled(
    n_total: int,
    dtype_str: str,
    num_blocks: int,
    tile_elems: int,
    num_stages: int,
    min_val: float,
    max_val: float,
    lo_sign: int,
    hi_sign: int,
):
    """Persistent-block, pipelined main loop over an exact multiple of ``tile_elems``."""
    min_val = math.copysign(min_val, lo_sign)
    max_val = math.copysign(max_val, hi_sign)
    lo_op, hi_op = _tie_ops(min_val, max_val)
    block_span = num_blocks * tile_elems
    assert n_total % block_span == 0, (
        f"_build_tiled requires n_total ({n_total}) to divide num_blocks*tile_elems ({block_span}) exactly; the caller routes other shapes elsewhere"
    )
    assert tile_elems % 64 == 0, (
        f"_hardtanh_body_vf works one 64-lane fp32 vector at a time; tile_elems ({tile_elems}) must be a multiple of 64"
    )
    iterations = n_total // block_span

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
        with T.Kernel(num_blocks) as bx:
            x_ub = T.alloc_shared((tile_elems,), dtype_str)
            y_ub = T.alloc_shared((tile_elems,), dtype_str)
            T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin = (step * num_blocks + bx) * tile_elems
                T.copy(X[begin : begin + tile_elems], x_ub)
                if dtype_str != "float32" and min_val == -1.0 and (max_val == 1.0):
                    _hardtanh_body_native16(x_ub, y_ub, tile_elems, dtype_str, min_val, max_val)
                else:
                    _hardtanh_body_vf(
                        x_ub,
                        y_ub,
                        tile_elems,
                        dtype_str,
                        lo_op,
                        hi_op,
                        min_val=min_val,
                        max_val=max_val,
                    )
                T.copy(y_ub, Y[begin : begin + tile_elems])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _build_general(
    n_total: int,
    dtype_str: str,
    num_blocks: int,
    tile_elems: int,
    num_stages: int,
    min_val: float,
    max_val: float,
    lo_sign: int,
    hi_sign: int,
):
    """Correctness fallback for any ``n_total`` that no fast path divides evenly."""
    min_val = math.copysign(min_val, lo_sign)
    max_val = math.copysign(max_val, hi_sign)
    full_tiles = n_total // tile_elems
    rem = n_total - full_tiles * tile_elems
    iterations = (full_tiles + num_blocks - 1) // num_blocks

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
        with T.Kernel(num_blocks) as bx:
            x_ub = T.alloc_shared((tile_elems,), dtype_str)
            y_ub = T.alloc_shared((tile_elems,), dtype_str)
            T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                idx = step * num_blocks + bx
                if idx < full_tiles:
                    begin = idx * tile_elems
                    T.copy(X[begin : begin + tile_elems], x_ub)
                    _hardtanh_body(
                        x_ub,
                        y_ub,
                        tile_elems,
                        threads=min(tile_elems, _MAX_THREADS),
                        min_val=min_val,
                        max_val=max_val,
                    )
                    T.copy(y_ub, Y[begin : begin + tile_elems])
            if rem > 0:  # noqa: SIM102 - compile-time extent and runtime block predicate
                if bx == 0:
                    begin = full_tiles * tile_elems
                    T.copy(X[begin : begin + rem], x_ub[:rem])
                    _hardtanh_body(
                        x_ub,
                        y_ub,
                        rem,
                        threads=min(rem, _MAX_THREADS),
                        min_val=min_val,
                        max_val=max_val,
                    )
                    T.copy(y_ub[:rem], Y[begin : begin + rem])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _build_hardtanh_kernel(n_total: int, dtype_str: str, min_val: float, max_val: float):
    """Pick the tiling for ``n_total`` elements: persistent tiled path (calibrated or
    ``bounded-conv-feat``-specific block count), or single-block fallback."""
    lo_sign = _zero_sign_key(min_val)
    hi_sign = _zero_sign_key(max_val)
    tuned = _TUNED_TILING.get((n_total, dtype_str))
    if tuned is not None:
        blocks, tile, stages = tuned
        if n_total % (blocks * tile) == 0:
            return _build_tiled(
                n_total, dtype_str, blocks, tile, stages, min_val, max_val, lo_sign, hi_sign
            )
    tile_elems = _tile_elems_for_dtype(dtype_str)
    if n_total < tile_elems:
        return _build_single_block(n_total, dtype_str, min_val, max_val, lo_sign, hi_sign)
    num_blocks = _calibrated_num_blocks()
    block_span = num_blocks * tile_elems
    if n_total % block_span == 0:
        return _build_tiled(
            n_total,
            dtype_str,
            num_blocks,
            tile_elems,
            _NUM_STAGES,
            min_val,
            max_val,
            lo_sign,
            hi_sign,
        )
    conv_feat_span = _BOUNDED_CONV_FEAT_NUM_BLOCKS * tile_elems
    if n_total % conv_feat_span == 0:
        return _build_tiled(
            n_total,
            dtype_str,
            _BOUNDED_CONV_FEAT_NUM_BLOCKS,
            tile_elems,
            _NUM_STAGES,
            min_val,
            max_val,
            lo_sign,
            hi_sign,
        )
    if n_total == tile_elems:
        return _build_single_block(n_total, dtype_str, min_val, max_val, lo_sign, hi_sign)
    return _build_general(
        n_total, dtype_str, num_blocks, tile_elems, _NUM_STAGES, min_val, max_val, lo_sign, hi_sign
    )


class AscendHardtanhKernel(Kernel, BoundedUnaryFwdInterface):
    """Hardtanh on Ascend NPUs: ``y = clamp(x, min_val, max_val)``, NaN-propagating."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: BoundsCall) -> Entry:
        return (call, lambda: cls(call.n_total, call.dtype, call.min_val, call.max_val))

    def __init__(
        self, n_total: int, dtype: torch.dtype, min_val: float = -1.0, max_val: float = 1.0
    ) -> None:
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.kernel = _build_hardtanh_kernel(n_total, _dtype_str(dtype), self.min_val, self.max_val)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Clamp ``input`` elementwise into ``[self.min_val, self.max_val]``; nothing is written
        in place."""
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got {input.device}. Another target's backend serves other devices."
            )
        flat_input = input.reshape(-1)
        if flat_input.numel() != self.n_total:
            raise ValueError(
                f"{type(self).__name__} was built for n_total={self.n_total}, got {flat_input.numel()} elements"
            )
        flat_output = self.kernel(flat_input)
        return flat_output.reshape(input.shape)


register_kernel_type("HardtanhFwdOp", "ascend", AscendHardtanhKernel)
