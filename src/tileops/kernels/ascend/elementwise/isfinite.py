"""Ascend 950 implementations for IsfiniteFwdOp."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_vector_cores

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryPredicateFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["AscendIsfiniteKernel"]
FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
INT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
ALWAYS_TRUE_DTYPES = INT_DTYPES + (torch.bool,)
_TILE_FP16_BF16 = 32768
_TILE_FP32 = 16384
_NUM_STAGES = 2
_PACK_CONFIG = {"float16": (64, 8192, 2, 1024), "bfloat16": (64, 8192, 2, 1024)}
_MASK_BITS = {
    "float16": ("int16", 31744),
    "bfloat16": ("int16", 32640),
    "float32": ("int32", 2139095040),
}
_WORD32_MASKS = {"float16": (31744, 2080374784), "bfloat16": (32640, 2139095040)}
_FALLBACK_VECTOR_CORES = 72
_MAX_THREADS = 2048


def _floor_pow2(n: int) -> int:
    """Largest power of two ``<= n`` (``n >= 1``)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


@functools.lru_cache(maxsize=1)
def _calibrated_num_blocks() -> int:
    """Persistent-grid block count for the pipelined, large-``n`` path."""
    try:
        queried = get_num_vector_cores(device_id=0)
    except Exception:
        queried = None
    if not queried:
        return _FALLBACK_VECTOR_CORES
    return _floor_pow2(int(queried))


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


@T.macro
def _isfinite_body(x_ub, y_ub, n: int, threads: int) -> None:
    """``y_ub[:n] = isfinite(x_ub[:n])`` as bool, compiler-vectorized over ``threads``."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n):
            y_ub[i] = T.isfinite(T.cast(x_ub[i], "float32"))


@T.macro
def _isfinite_pack4_body(
    x_ub, y_ub, n_packed: int, bits_dtype: str, exp_mask: int, threads: int
) -> None:
    """``y_ub[i] = pack4(isfinite(x_ub[4*i : 4*i+4]))``; lane k -> byte k, little endian."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n_packed):
            b0 = T.cast(
                T.bitwise_and(
                    T.reinterpret(x_ub[4 * i + 0], bits_dtype), T.cast(exp_mask, bits_dtype)
                )
                != T.cast(exp_mask, bits_dtype),
                "int32",
            )
            b1 = T.cast(
                T.bitwise_and(
                    T.reinterpret(x_ub[4 * i + 1], bits_dtype), T.cast(exp_mask, bits_dtype)
                )
                != T.cast(exp_mask, bits_dtype),
                "int32",
            )
            b2 = T.cast(
                T.bitwise_and(
                    T.reinterpret(x_ub[4 * i + 2], bits_dtype), T.cast(exp_mask, bits_dtype)
                )
                != T.cast(exp_mask, bits_dtype),
                "int32",
            )
            b3 = T.cast(
                T.bitwise_and(
                    T.reinterpret(x_ub[4 * i + 3], bits_dtype), T.cast(exp_mask, bits_dtype)
                )
                != T.cast(exp_mask, bits_dtype),
                "int32",
            )
            lo = T.bitwise_or(b0, T.shift_left(b1, T.cast(8, "int32")))
            hi = T.bitwise_or(
                T.shift_left(b2, T.cast(16, "int32")), T.shift_left(b3, T.cast(24, "int32"))
            )
            y_ub[i] = T.bitwise_or(lo, hi)


@T.macro
def _isfinite_word32_body(
    a_ub, c_ub, n_packed: int, low_mask: int, high_mask: int, threads: int
) -> None:
    """Two int32 input words -> four 16-bit exponent predicates -> one packed word."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n_packed):
            w0 = a_ub[2 * i]
            w1 = a_ub[2 * i + 1]
            b0 = T.cast(
                T.bitwise_and(w0, T.cast(low_mask, "int32")) != T.cast(low_mask, "int32"), "int32"
            )
            b1 = T.cast(
                T.bitwise_and(w0, T.cast(high_mask, "int32")) != T.cast(high_mask, "int32"), "int32"
            )
            b2 = T.cast(
                T.bitwise_and(w1, T.cast(low_mask, "int32")) != T.cast(low_mask, "int32"), "int32"
            )
            b3 = T.cast(
                T.bitwise_and(w1, T.cast(high_mask, "int32")) != T.cast(high_mask, "int32"), "int32"
            )
            lo = T.bitwise_or(b0, T.shift_left(b1, T.cast(8, "int32")))
            hi = T.bitwise_or(
                T.shift_left(b2, T.cast(16, "int32")), T.shift_left(b3, T.cast(24, "int32"))
            )
            c_ub[i] = T.bitwise_or(lo, hi)


@functools.lru_cache(maxsize=64)
def _make_isfinite_persistent(
    n_total: int, dtype: str, num_blocks: int, tile: int, num_stages: int
):
    """Persistent multi-block grid with ``T.Pipelined`` staging (module docstring)."""
    block_span = num_blocks * tile
    assert n_total % block_span == 0, (
        f"_make_isfinite_persistent requires n_total ({n_total}) to divide num_blocks*tile ({block_span}) exactly; the caller routes other shapes elsewhere"
    )
    iterations = n_total // block_span

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), "bool")):
        with T.Kernel(num_blocks) as bx:
            x_ub = T.alloc_shared((tile,), dtype)
            y_ub = T.alloc_shared((tile,), "bool")
            T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin = (step * num_blocks + bx) * tile
                T.copy(X[begin : begin + tile], x_ub)
                _isfinite_body(x_ub, y_ub, tile, threads=min(_MAX_THREADS, tile))
                T.copy(y_ub, Y[begin : begin + tile])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _make_isfinite_packed(
    n_total: int, dtype: str, num_blocks: int, tile_packed: int, num_stages: int, threads: int
):
    """Persistent packed grid: four input values per int32 output word."""
    tile_in = 4 * tile_packed
    block_span = num_blocks * tile_in
    assert n_total % block_span == 0, (
        f"_make_isfinite_packed requires n_total ({n_total}) to divide num_blocks*tile_in ({block_span}) exactly; routing keeps other shapes on the original implementation"
    )
    iterations = n_total // block_span
    bits_dtype, exp_mask = _MASK_BITS[dtype]

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total // 4,), "int32")):
        with T.Kernel(num_blocks) as bx:
            x_ub = T.alloc_shared((tile_in,), dtype)
            y_ub = T.alloc_shared((tile_packed,), "int32")
            T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin_in = (step * num_blocks + bx) * tile_in
                begin_out = (step * num_blocks + bx) * tile_packed
                T.copy(X[begin_in : begin_in + tile_in], x_ub)
                _isfinite_pack4_body(x_ub, y_ub, tile_packed, bits_dtype, exp_mask, threads=threads)
                T.copy(y_ub, Y[begin_out : begin_out + tile_packed])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _make_isfinite_word32(
    n_total: int, dtype: str, num_blocks: int, tile_packed: int, num_stages: int, threads: int
):
    """Persistent word32-input pack4 builder for the two 16-bit dtypes."""
    tile_words = 2 * tile_packed
    input_words = n_total // 2
    assert input_words % (num_blocks * tile_words) == 0, (
        f"_make_isfinite_word32 requires input_words ({input_words}) to divide num_blocks*tile_words ({num_blocks * tile_words}) exactly"
    )
    iterations = input_words // (num_blocks * tile_words)
    low_mask, high_mask = _WORD32_MASKS[dtype]

    @T.prim_func
    def main(A: T.Tensor((input_words,), "int32"), C: T.Tensor((n_total // 4,), "int32")):
        with T.Kernel(num_blocks) as bx:
            a_ub = T.alloc_shared((tile_words,), "int32")
            c_ub = T.alloc_shared((tile_packed,), "int32")
            T.annotate_buffer_versions({a_ub: num_stages, c_ub: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin_in = (step * num_blocks + bx) * tile_words
                begin_out = (step * num_blocks + bx) * tile_packed
                T.copy(A[begin_in : begin_in + tile_words], a_ub)
                _isfinite_word32_body(a_ub, c_ub, tile_packed, low_mask, high_mask, threads=threads)
                T.copy(c_ub, C[begin_out : begin_out + tile_packed])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _packed_shape(n_total: int, dtype: str) -> bool:
    """True only when the packed builder divides this exact shape and dtype."""
    if dtype not in _PACK_CONFIG:
        return False
    blocks, tile_packed, _stages, _threads = _PACK_CONFIG[dtype]
    return n_total % (4 * blocks * tile_packed) == 0


@functools.lru_cache(maxsize=64)
def _make_isfinite_generic_single_block(n_total: int, dtype: str, tile: int):
    full_tiles, remainder = divmod(n_total, tile)

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), "bool")):
        with T.Kernel(1):
            if full_tiles:
                x_ub = T.alloc_shared((tile,), dtype)
                y_ub = T.alloc_shared((tile,), "bool")
                T.annotate_buffer_versions({x_ub: 2, y_ub: 2})
                for step in T.Pipelined(full_tiles, num_stages=2):
                    begin = step * tile
                    T.copy(X[begin : begin + tile], x_ub)
                    _isfinite_body(x_ub, y_ub, tile, threads=min(_MAX_THREADS, tile))
                    T.copy(y_ub, Y[begin : begin + tile])
            if remainder:
                xt_ub = T.alloc_shared((remainder,), dtype)
                yt_ub = T.alloc_shared((remainder,), "bool")
                begin = full_tiles * tile
                T.copy(X[begin : begin + remainder], xt_ub)
                _isfinite_body(xt_ub, yt_ub, remainder, threads=min(128, remainder))
                T.copy(yt_ub, Y[begin : begin + remainder])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _tile_for_dtype(dtype: str) -> int:
    """Per-dtype tile width (module docstring): 32768 for fp16/bf16 (192 KiB/block at
    2 stages), 16384 for fp32 (160 KiB/block -- 32768 would be 320 KiB, over
    BitwiseNotFwdOp's confirmed 256 KiB overflow point for a 4-byte-wide input)."""
    return _TILE_FP32 if dtype == "float32" else _TILE_FP16_BF16


def _make_isfinite_kernel(n_total: int, dtype: str):
    """Pick the tiling for ``n_total`` elements of ``dtype``: persistent grid when it
    divides ``num_blocks * tile`` exactly, else the generic single-block fallback."""
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")
    if _packed_shape(n_total, dtype):
        blocks, tile_packed, stages, threads = _PACK_CONFIG[dtype]
        return _make_isfinite_word32(n_total, dtype, blocks, tile_packed, stages, threads)
    num_blocks = _calibrated_num_blocks()
    tile = _tile_for_dtype(dtype)
    block_span = num_blocks * tile
    if n_total % block_span == 0:
        return _make_isfinite_persistent(n_total, dtype, num_blocks, tile, _NUM_STAGES)
    return _make_isfinite_generic_single_block(n_total, dtype, tile)


class AscendIsfiniteKernel(Kernel, UnaryPredicateFwdInterface):
    """IsFinite on Ascend NPUs: ``y = isfinite(x)``, a new contiguous ``torch.bool`` tensor."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: ElementwiseCall) -> "str | None":
        return None

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return (call, lambda: cls(call.n_total, call.dtype))

    def __init__(self, n_total: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self._use_pack = dtype in FLOAT_DTYPES and _packed_shape(n_total, _dtype_str(dtype))
        self._compiled_fn = (
            _make_isfinite_kernel(n_total, _dtype_str(dtype)) if dtype in FLOAT_DTYPES else None
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return ``isfinite(input)`` elementwise; nothing is written in place."""
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got {input.device}. Another target's backend serves other devices."
            )
        if self.dtype not in FLOAT_DTYPES:
            return torch.ones(input.shape, dtype=torch.bool, device=input.device)
        flat_input = input.reshape(-1)
        if flat_input.numel() != self.n_total:
            raise ValueError(
                f"{type(self).__name__} was built for n_total={self.n_total}, got {flat_input.numel()} elements"
            )
        flat_output = self._compiled_fn(
            flat_input.view(torch.int32) if self._use_pack else flat_input
        )
        if self._use_pack:
            return flat_output.view(torch.uint8).view(torch.bool).reshape(input.shape)
        return flat_output.reshape(input.shape)


register_kernel_type("IsfiniteFwdOp", "ascend", AscendIsfiniteKernel)
