"""Ascend 950 implementations for LeakyReluFwdOp."""

import functools
import math
import struct

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_vector_cores

import tileops.backend
from tileops.kernels.elementwise.call_spec import LeakyReluCall, LeakyReluFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel


def _floor_pow2(n: int) -> int:
    """Largest power of two ``<= n`` (``n >= 1``)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _vlrelu_slope_eligible(slope) -> bool:
    """True only for a positive, finite, normal-fp32 scalar slope."""
    try:
        s = float(slope)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(s) or s <= 0.0:
        return False
    try:
        f32 = struct.unpack("<f", struct.pack("<f", s))[0]
    except OverflowError:
        return False
    return math.isfinite(f32) and f32 >= 2.0 ** (-126)


def _slope_cache_key(negative_slope) -> bytes:
    """Exact-bit cache key for ``negative_slope``."""
    try:
        return struct.pack("<d", float(negative_slope))
    except (TypeError, ValueError, struct.error):
        return repr(negative_slope).encode()


@functools.lru_cache(maxsize=1)
def _calibrated_num_blocks(fallback: int = 64) -> int:
    """Persistent-grid block count for the large, bandwidth-bound path."""
    try:
        queried = get_num_vector_cores(device_id=0)
    except Exception:
        queried = None
    if not queried:
        return fallback
    return _floor_pow2(int(queried))


_TILE_16BIT = 16384
_TILE_FP32 = 8192
_NUM_STAGES = 3


def _tile_for(dtype: str) -> int:
    """Persistent-path tile width for ``dtype`` (16-bit vs fp32 UB budget; see above)."""
    return _TILE_FP32 if dtype in ("float32", "float") else _TILE_16BIT


_TILE_SMALL_FALLBACK = 2048


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


@T.macro
def _leaky_relu_body_promote(x_ub, y_ub, n: int, threads: int, negative_slope: float) -> None:
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n):
            zero_f32 = T.cast(0, "float32")
            slope_f32 = T.cast(negative_slope, "float32")
            wide = T.cast(x_ub[i], "float32")
            neg = T.cast(slope_f32 * wide, y_ub.dtype)
            y_ub[i] = T.if_then_else(wide > zero_f32, x_ub[i], neg)


def _make_zero_slope_body(zero_sign_bits: int):
    """Build the IEEE ``0*x`` generic/tail body with the slope sign as an integer."""

    @T.macro
    def body(x_ub, y_ub, n: int, threads: int, negative_slope: float) -> None:
        """IEEE ``0*x`` fallback for the generic/tail path when ``negative_slope`` is +-0.0."""
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                wide = T.cast(x_ub[i], "float32")
                sign = T.reinterpret(wide, "uint32") & T.uint32(2147483648) ^ T.uint32(
                    zero_sign_bits
                )
                signed_zero = T.reinterpret(sign, "float32")
                neg = T.if_then_else(
                    T.isnan(wide),
                    wide,
                    T.if_then_else(T.isinf(wide), T.cast(float("nan"), "float32"), signed_zero),
                )
                y_ub[i] = T.if_then_else(
                    wide > T.cast(0, "float32"), x_ub[i], T.cast(neg, y_ub.dtype)
                )

    return body


@T.macro
def _leaky_relu_body_native(x_ub, y_ub, n: int, threads: int, negative_slope: float) -> None:
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n):
            zero_native = T.cast(0, x_ub.dtype)
            slope_native = T.cast(negative_slope, x_ub.dtype)
            y_ub[i] = T.if_then_else(x_ub[i] > zero_native, x_ub[i], slope_native * x_ub[i])


@T.macro
def _leaky_relu_body_vf16(x_ub, y_ub, n: int, threads: int, negative_slope: float) -> None:
    """Explicit VF-A body for 16-bit storage dtypes (fp16/bf16), fp32 arithmetic."""
    with T.SimdVF():
        for g in T.serial(n // 64 // 4):
            for s in T.unroll(4, explicit=True):
                base = (g * 4 + s) * 64
                x16 = T.simd.vld(x_ub[base], dist="UNPK_B16")
                xf = T.simd.vcvt(x16, "float32", part=0)
                m = T.simd.vcmps(xf, T.cast(0, "float32"), op="gt")
                neg = T.simd.vmuls(xf, T.cast(negative_slope, "float32"))
                yf = T.simd.vsel(xf, neg, m)
                y16 = T.simd.vcvt(yf, y_ub.dtype, sat=True)
                T.simd.vsts(y_ub[base], y16, dist="PK_B32")


@T.macro
def _leaky_relu_body_vf_f32(x_ub, y_ub, n: int, threads: int, negative_slope: float) -> None:
    with T.SimdVF():
        for g in T.serial(n // 64 // 4):
            for s in T.unroll(4, explicit=True):
                base = (g * 4 + s) * 64
                xf = T.simd.vld(x_ub[base])
                m = T.simd.vcmps(xf, T.cast(0, "float32"), op="gt")
                neg = T.simd.vmuls(xf, T.cast(negative_slope, "float32"))
                yf = T.simd.vsel(xf, neg, m)
                T.simd.vsts(y_ub[base], yf)


@T.macro
def _leaky_relu_body_vlrelu16(x_ub, y_ub, n: int, threads: int, negative_slope: float) -> None:
    """VF-B for 16-bit storage: storage -> fp32 -> vlrelu(fp32 alpha) -> storage."""
    with T.SimdVF():
        for g in T.serial(n // 64 // 4):
            for s in T.unroll(4, explicit=True):
                base = (g * 4 + s) * 64
                x16 = T.simd.vld(x_ub[base], dist="UNPK_B16")
                xf = T.simd.vcvt(x16, "float32", part=0)
                yf = T.simd.vlrelu(xf, T.cast(negative_slope, "float32"))
                y16 = T.simd.vcvt(yf, y_ub.dtype, sat=True)
                T.simd.vsts(y_ub[base], y16, dist="PK_B32")


@T.macro
def _leaky_relu_body_vlrelu_f32(x_ub, y_ub, n: int, threads: int, negative_slope: float) -> None:
    """VF-B for float32 storage: vld -> vlrelu(fp32 alpha) -> vsts."""
    with T.SimdVF():
        for g in T.serial(n // 64 // 4):
            for s in T.unroll(4, explicit=True):
                base = (g * 4 + s) * 64
                xf = T.simd.vld(x_ub[base])
                yf = T.simd.vlrelu(xf, T.cast(negative_slope, "float32"))
                T.simd.vsts(y_ub[base], yf)


def _body_for(variant: str):
    if variant == "promote":
        return _leaky_relu_body_promote
    if variant == "native":
        return _leaky_relu_body_native
    raise ValueError(f"unknown leaky_relu body variant {variant!r}")


def _generic_body_for(variant: str, negative_slope: float, zero_sign_bits: int):
    """Generic/tail body: the IEEE zero-slope form for a +-0.0 slope, else the usual body."""
    if variant == "promote" and float(negative_slope) == 0.0:
        return _make_zero_slope_body(zero_sign_bits)
    return _body_for(variant)


def _persistent_body_for(dtype: str, variant: str, negative_slope: float):
    """Persistent-path body selection (see ``_vlrelu_slope_eligible``)."""
    if variant != "promote":
        return _body_for(variant)
    if _vlrelu_slope_eligible(negative_slope):
        return _leaky_relu_body_vlrelu_f32 if dtype == "float32" else _leaky_relu_body_vlrelu16
    return _leaky_relu_body_vf_f32 if dtype == "float32" else _leaky_relu_body_vf16


@functools.lru_cache(maxsize=64)
def _make_leaky_relu_persistent(
    n_total: int,
    dtype: str,
    num_blocks: int,
    tile: int,
    num_stages: int,
    negative_slope: float,
    slope_key: bytes,
    zero_sign_bits: int,
    variant: str = "promote",
):
    block_span = num_blocks * tile
    assert n_total % block_span == 0, (
        f"_make_leaky_relu_persistent requires n_total ({n_total}) to divide num_blocks*tile ({block_span}) exactly; the caller routes other shapes elsewhere"
    )
    iterations = n_total // block_span
    assert tile % 256 == 0, f"VF-A body needs tile divisible by 256, got {tile}"
    body = _persistent_body_for(dtype, variant, negative_slope)

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
        with T.Kernel(num_blocks) as bx:
            x_ub = T.alloc_shared((tile,), dtype)
            y_ub = T.alloc_shared((tile,), dtype)
            T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin = (step * num_blocks + bx) * tile
                T.copy(X[begin : begin + tile], x_ub)
                body(x_ub, y_ub, tile, threads=min(2048, tile), negative_slope=negative_slope)
                T.copy(y_ub, Y[begin : begin + tile])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _make_leaky_relu_generic_single_block(
    n_total: int,
    dtype: str,
    tile_small: int,
    negative_slope: float,
    slope_key: bytes,
    zero_sign_bits: int,
    variant: str = "promote",
):
    full_tiles, remainder = divmod(n_total, tile_small)
    body = _generic_body_for(variant, negative_slope, zero_sign_bits)

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
        with T.Kernel(1):
            if full_tiles:
                x_ub = T.alloc_shared((tile_small,), dtype)
                y_ub = T.alloc_shared((tile_small,), dtype)
                T.annotate_buffer_versions({x_ub: 2, y_ub: 2})
                for step in T.Pipelined(full_tiles, num_stages=2):
                    begin = step * tile_small
                    T.copy(X[begin : begin + tile_small], x_ub)
                    body(
                        x_ub,
                        y_ub,
                        tile_small,
                        threads=min(2048, tile_small),
                        negative_slope=negative_slope,
                    )
                    T.copy(y_ub, Y[begin : begin + tile_small])
            if remainder:
                xt_ub = T.alloc_shared((remainder,), dtype)
                yt_ub = T.alloc_shared((remainder,), dtype)
                begin = full_tiles * tile_small
                T.copy(X[begin : begin + remainder], xt_ub)
                body(
                    xt_ub,
                    yt_ub,
                    remainder,
                    threads=min(128, remainder),
                    negative_slope=negative_slope,
                )
                T.copy(yt_ub, Y[begin : begin + remainder])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _make_leaky_relu_kernel(
    n_total: int, dtype: str, negative_slope: float, variant: str = "promote"
):
    """Pick the tiling for ``n_total`` elements: persistent grid, or generic fallback."""
    num_blocks = _calibrated_num_blocks()
    tile = _tile_for(dtype)
    block_span = num_blocks * tile
    slope_key = _slope_cache_key(negative_slope)
    zero_sign_bits = 2147483648 if math.copysign(1.0, float(negative_slope)) < 0.0 else 0
    if n_total % block_span == 0:
        return _make_leaky_relu_persistent(
            n_total,
            dtype,
            num_blocks,
            tile,
            _NUM_STAGES,
            negative_slope,
            slope_key,
            zero_sign_bits,
            variant,
        )
    return _make_leaky_relu_generic_single_block(
        n_total, dtype, _TILE_SMALL_FALLBACK, negative_slope, slope_key, zero_sign_bits, variant
    )


class AscendLeakyReluKernel(Kernel, LeakyReluFwdInterface):
    """Leaky ReLU on Ascend NPUs: ``y = x if x > 0 else negative_slope * x``."""

    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: LeakyReluCall) -> Entry:
        return (call, lambda: cls(call.n_total, call.dtype, call.negative_slope))

    def __init__(self, n_total: int, dtype: torch.dtype, negative_slope: float = 0.01):
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.negative_slope = negative_slope
        self._compiled_fn = _make_leaky_relu_kernel(
            n_total, _dtype_str(dtype), negative_slope, variant="promote"
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        flat = input.reshape(-1)
        result = self._compiled_fn(flat)
        return result.reshape(input.shape)


tileops.backend.register_kernel_type("LeakyReluFwdOp", "ascend", AscendLeakyReluKernel)
