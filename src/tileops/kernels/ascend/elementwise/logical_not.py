"""Ascend 950 implementations for LogicalNotFwdOp."""

from functools import lru_cache
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryPredicateFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["AscendLogicalNotKernel"]
_NUM_BLOCKS = 64
_NUM_STAGES = 2
_GENERIC_TILE = 2048
_PACK_TILE = 8192
_ZERO_WORD_LOW7 = 2139062143
_ZERO_WORD_HIGH = 16843009
_FLOAT_PACK_TILE = {"float16": 8192, "bfloat16": 8192, "float32": 4096}


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _choose_grid(n: int, tile: int, num_blocks_default: int, threads_cap: int = 2048):
    """Pick ``(num_blocks, tile, threads, iterations, single_block)`` for ``n`` elements."""
    if n <= 0:
        raise ValueError(f"_choose_grid: n must be positive, got {n}")
    if n < tile:
        return (1, n, min(n, threads_cap), 1, True)
    if n % tile == 0:
        num_full = n // tile
        num_blocks = min(num_blocks_default, num_full)
        iterations = num_full // num_blocks
        return (num_blocks, tile, min(tile, threads_cap), iterations, False)
    raise ValueError(
        f"_choose_grid: n={n} is >= tile={tile} but does not divide it; no remainder-tile handling"
    )


@T.macro
def _cmp_zero_body(a_ub, c_ub, n: int, dtype_str: str, threads: int) -> None:
    """``c_ub[:n] = (a_ub[:n] == 0)`` as bool. ``@T.macro``, not a plain function --"""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n):
            c_ub[i] = a_ub[i] == T.cast(0, dtype_str)


@T.macro
def _pack_zero_word_body(a_ub, c_ub, n: int, threads: int) -> None:
    """``c_ub[:n] = canonical_byte01(input word == 0 per byte)``."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n):
            u = T.reinterpret(a_ub[i], "uint32")
            low = T.bitwise_and(u, T.uint32(_ZERO_WORD_LOW7))
            carried = low + T.uint32(_ZERO_WORD_LOW7)
            merged = T.bitwise_or(T.bitwise_or(carried, u), T.uint32(_ZERO_WORD_LOW7))
            inverted = T.bitwise_xor(merged, T.uint32(4294967295))
            canonical = T.bitwise_and(
                T.shift_right(inverted, T.uint32(7)), T.uint32(_ZERO_WORD_HIGH)
            )
            c_ub[i] = T.reinterpret(canonical, "int32")


@lru_cache(maxsize=None)
def _build_generic_single_block(n_total: int, dtype_str: str, threads: int):
    """One block, one untiled GM->UB->compare->UB->GM pass. Correctness-only --"""

    @T.prim_func
    def main(A: T.Tensor((n_total,), dtype_str), C: T.Tensor((n_total,), "bool")):
        with T.Kernel(1):
            a = T.alloc_shared((n_total,), dtype_str)
            c = T.alloc_shared((n_total,), "bool")
            T.copy(A, a)
            _cmp_zero_body(a, c, n_total, dtype_str, threads)
            T.copy(c, C)

    return tilelang.compile(main, target="ascend", out_idx=-1)


@lru_cache(maxsize=None)
def _build_generic_persistent(
    n_total: int, dtype_str: str, num_blocks: int, tile: int, threads: int, iterations: int
):
    """Persistent multi-block grid with ``T.Pipelined`` staging."""

    @T.prim_func
    def main(A: T.Tensor((n_total,), dtype_str), C: T.Tensor((n_total,), "bool")):
        with T.Kernel(num_blocks) as bx:
            a = T.alloc_shared((tile,), dtype_str)
            c = T.alloc_shared((tile,), "bool")
            T.annotate_buffer_versions({a: _NUM_STAGES, c: _NUM_STAGES})
            for step in T.Pipelined(iterations, num_stages=_NUM_STAGES):
                begin = (step * num_blocks + bx) * tile
                T.copy(A[begin : begin + tile], a)
                _cmp_zero_body(a, c, tile, dtype_str, threads)
                T.copy(c, C[begin : begin + tile])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _build_generic_kernel(n_total: int, dtype_str: str):
    """Pick the strategy-1 tiling for ``n_total`` elements of ``dtype_str``."""
    num_blocks, tile, threads, iterations, single_block = _choose_grid(
        n_total, _GENERIC_TILE, _NUM_BLOCKS
    )
    if single_block:
        return _build_generic_single_block(n_total, dtype_str, threads)
    return _build_generic_persistent(n_total, dtype_str, num_blocks, tile, threads, iterations)


@lru_cache(maxsize=None)
def _build_pack_single_block(n_packed: int, threads: int):
    """One block, one untiled pass over ``n_packed`` int32 words. Correctness-only --"""

    @T.prim_func
    def main(A: T.Tensor((n_packed,), "int32"), C: T.Tensor((n_packed,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((n_packed,), "int32")
            c = T.alloc_shared((n_packed,), "int32")
            T.copy(A, a)
            _pack_zero_word_body(a, c, n_packed, threads)
            T.copy(c, C)

    return tilelang.compile(main, target="ascend", out_idx=-1)


@lru_cache(maxsize=None)
def _build_pack_persistent(
    n_packed: int, num_blocks: int, tile: int, iterations: int, num_stages: int = _NUM_STAGES
):
    @T.prim_func
    def main(A: T.Tensor((n_packed,), "int32"), C: T.Tensor((n_packed,), "int32")):
        with T.Kernel(num_blocks) as bx:
            a = T.alloc_shared((tile,), "int32")
            c = T.alloc_shared((tile,), "int32")
            T.annotate_buffer_versions({a: num_stages, c: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin = (step * num_blocks + bx) * tile
                T.copy(A[begin : begin + tile], a)
                _pack_zero_word_body(a, c, tile, threads=min(tile, 2048))
                T.copy(c, C[begin : begin + tile])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _build_bool_pack_kernel(n_packed: int, num_stages: int = _NUM_STAGES):
    """Pick the strategy-2 (zero-word) tiling for ``n_packed`` int32 words."""
    num_blocks, tile, threads, iterations, single_block = _choose_grid(
        n_packed, _PACK_TILE, _NUM_BLOCKS
    )
    if single_block:
        return _build_pack_single_block(n_packed, threads)
    return _build_pack_persistent(n_packed, num_blocks, tile, iterations, num_stages=num_stages)


@T.macro
def _pack_cmp_zero_body(a_ub, c_ub, n_packed: int, dtype_str: str, threads: int) -> None:
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n_packed):
            b0 = T.cast(a_ub[4 * i + 0] == T.cast(0, dtype_str), "int32")
            b1 = T.cast(a_ub[4 * i + 1] == T.cast(0, dtype_str), "int32")
            b2 = T.cast(a_ub[4 * i + 2] == T.cast(0, dtype_str), "int32")
            b3 = T.cast(a_ub[4 * i + 3] == T.cast(0, dtype_str), "int32")
            lo = T.bitwise_or(b0, T.shift_left(b1, T.cast(8, "int32")))
            hi = T.bitwise_or(
                T.shift_left(b2, T.cast(16, "int32")), T.shift_left(b3, T.cast(24, "int32"))
            )
            c_ub[i] = T.bitwise_or(lo, hi)


@lru_cache(maxsize=None)
def _build_float_pack_single_block(n_total: int, dtype_str: str, n_packed: int, threads: int):
    """One block, one untiled pass. Correctness-only -- serves the strategy-3 path's"""

    @T.prim_func
    def main(A: T.Tensor((n_total,), dtype_str), C: T.Tensor((n_packed,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((n_total,), dtype_str)
            c = T.alloc_shared((n_packed,), "int32")
            T.copy(A, a)
            _pack_cmp_zero_body(a, c, n_packed, dtype_str, threads)
            T.copy(c, C)

    return tilelang.compile(main, target="ascend", out_idx=-1)


@lru_cache(maxsize=None)
def _build_float_pack_persistent(
    n_total: int, dtype_str: str, num_blocks: int, tile_in: int, tile_packed: int, iterations: int
):
    @T.prim_func
    def main(A: T.Tensor((n_total,), dtype_str), C: T.Tensor((n_total // 4,), "int32")):
        with T.Kernel(num_blocks) as bx:
            a = T.alloc_shared((tile_in,), dtype_str)
            c = T.alloc_shared((tile_packed,), "int32")
            T.annotate_buffer_versions({a: _NUM_STAGES, c: _NUM_STAGES})
            for step in T.Pipelined(iterations, num_stages=_NUM_STAGES):
                begin_in = (step * num_blocks + bx) * tile_in
                begin_out = (step * num_blocks + bx) * tile_packed
                T.copy(A[begin_in : begin_in + tile_in], a)
                _pack_cmp_zero_body(a, c, tile_packed, dtype_str, threads=min(tile_packed, 2048))
                T.copy(c, C[begin_out : begin_out + tile_packed])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _build_float_pack_kernel(n_total: int, dtype_str: str):
    """Pick the strategy-3 (float 4-lane pack-compare) tiling for ``n_total`` elements
    of ``dtype_str`` (one of ``_FLOAT_PACK_TILE``'s 3 keys)."""
    n_packed = n_total // 4
    num_blocks, tile_packed, threads, iterations, single_block = _choose_grid(
        n_packed, _FLOAT_PACK_TILE[dtype_str], _NUM_BLOCKS
    )
    if single_block:
        return _build_float_pack_single_block(n_total, dtype_str, n_packed, threads)
    return _build_float_pack_persistent(
        n_total, dtype_str, num_blocks, tile_packed * 4, tile_packed, iterations
    )


class AscendLogicalNotKernel(Kernel, UnaryPredicateFwdInterface):
    """Ascend implementation of ``LogicalNotFwdOp``: ``y = not x``, bandwidth-bound, no Cube."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return (call, lambda: cls(call.n_total, call.dtype))

    def __init__(self, n_total: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        dtype_str = self.dtype_to_str(dtype)
        divisible_by_4 = n_total % 4 == 0
        self._use_bool_pack = dtype == torch.bool and divisible_by_4
        self._use_float_pack = dtype_str in _FLOAT_PACK_TILE and divisible_by_4
        if self._use_bool_pack:
            self.kernel = _build_bool_pack_kernel(n_total // 4)
        elif self._use_float_pack:
            self.kernel = _build_float_pack_kernel(n_total, dtype_str)
        else:
            self.kernel = _build_generic_kernel(n_total, dtype_str)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return ``torch.logical_not(input)``; nothing is written in place."""
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got a tensor on {input.device}. Another target's backend serves other devices."
            )
        shape = input.shape
        flat_in = input.reshape(-1)
        if flat_in.numel() != self.n_total:
            raise ValueError(
                f"{type(self).__name__} was built for n_total={self.n_total}, got {flat_in.numel()} elements"
            )
        if self._use_bool_pack:
            packed_in = flat_in.view(torch.uint8).view(torch.int32)
            packed_out = self.kernel(packed_in)
            result = packed_out.view(torch.uint8).view(torch.bool)
        elif self._use_float_pack:
            packed_out = self.kernel(flat_in)
            result = packed_out.view(torch.uint8).view(torch.bool)
        else:
            result = self.kernel(flat_in)
        return result.reshape(shape)


register_kernel_type("LogicalNotFwdOp", "ascend", AscendLogicalNotKernel)
