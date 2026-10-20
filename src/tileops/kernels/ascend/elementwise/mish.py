"""Ascend 950 implementations for MishFwdOp."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_vector_cores

import tileops.backend
from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel


def _floor_pow2(n: int) -> int:
    """Largest power of two ``<= n`` (``n >= 1``)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


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


_TILE = 8192
_NUM_STAGES = 3
_THREADS = 2048
_SINGLE_BLOCK_THRESHOLD = _TILE


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


_SATURATION = 20.0


@T.macro
def _mish_body(x_ub, y_ub, n: int, threads: int) -> None:
    """``y_ub[:n] = mish(x_ub[:n])``, compiler-vectorized over ``threads`` lanes."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n):
            wide = T.Cast("float32", x_ub[i])
            e = T.exp(T.min(wide, T.cast(_SATURATION, "float32")))
            saturated = e * e + T.cast(2.0, "float32") * e
            ratio = saturated / (saturated + T.cast(2.0, "float32"))
            y_ub[i] = T.Cast(x_ub.dtype, wide * ratio)


@functools.lru_cache(maxsize=64)
def _make_mish_single_block(n_total: int, dtype: str):
    """One block, one tile, no pipelining: correctness-first fallback for a small shape
    that does not divide ``_calibrated_num_blocks()`` evenly."""

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((n_total,), dtype)
            y_ub = T.alloc_shared((n_total,), dtype)
            T.copy(X, x_ub)
            _mish_body(x_ub, y_ub, n_total, threads=min(128, n_total))
            T.copy(y_ub, Y)

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _make_mish_decode_multicore(n_total: int, dtype: str, num_blocks: int):
    assert n_total % num_blocks == 0, (
        f"_make_mish_decode_multicore requires n_total ({n_total}) to divide num_blocks ({num_blocks}) exactly; the caller routes other shapes elsewhere"
    )
    chunk = n_total // num_blocks

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
        with T.Kernel(num_blocks) as bx:
            x_ub = T.alloc_shared((chunk,), dtype)
            y_ub = T.alloc_shared((chunk,), dtype)
            begin = bx * chunk
            T.copy(X[begin : begin + chunk], x_ub)
            _mish_body(x_ub, y_ub, chunk, threads=min(2048, chunk))
            T.copy(y_ub, Y[begin : begin + chunk])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _make_mish_persistent(n_total: int, dtype: str, num_blocks: int, tile: int, num_stages: int):
    block_span = num_blocks * tile
    assert n_total % block_span == 0, (
        f"_make_mish_persistent requires n_total ({n_total}) to divide num_blocks*tile ({block_span}) exactly; the caller routes other shapes elsewhere"
    )
    iterations = n_total // block_span

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
        with T.Kernel(num_blocks) as bx:
            x_ub = T.alloc_shared((tile,), dtype)
            y_ub = T.alloc_shared((tile,), dtype)
            T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin = (step * num_blocks + bx) * tile
                T.copy(X[begin : begin + tile], x_ub)
                _mish_body(x_ub, y_ub, tile, threads=min(_THREADS, tile))
                T.copy(y_ub, Y[begin : begin + tile])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _make_mish_generic_single_block(n_total: int, dtype: str, tile: int):
    full_tiles, remainder = divmod(n_total, tile)

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
        with T.Kernel(1):
            if full_tiles:
                x_ub = T.alloc_shared((tile,), dtype)
                y_ub = T.alloc_shared((tile,), dtype)
                T.annotate_buffer_versions({x_ub: 2, y_ub: 2})
                for step in T.Pipelined(full_tiles, num_stages=2):
                    begin = step * tile
                    T.copy(X[begin : begin + tile], x_ub)
                    _mish_body(x_ub, y_ub, tile, threads=min(_THREADS, tile))
                    T.copy(y_ub, Y[begin : begin + tile])
            if remainder:
                xt_ub = T.alloc_shared((remainder,), dtype)
                yt_ub = T.alloc_shared((remainder,), dtype)
                begin = full_tiles * tile
                T.copy(X[begin : begin + remainder], xt_ub)
                _mish_body(xt_ub, yt_ub, remainder, threads=min(128, remainder))
                T.copy(yt_ub, Y[begin : begin + remainder])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _make_mish_kernel(n_total: int, dtype: str, tile: int = _TILE):
    """Pick the tiling for ``n_total`` elements: persistent, decode-multicore, single-block,
    or generic fallback."""
    num_blocks = _calibrated_num_blocks()
    block_span = num_blocks * tile
    if n_total % block_span == 0:
        return _make_mish_persistent(n_total, dtype, num_blocks, tile, _NUM_STAGES)
    if n_total <= _SINGLE_BLOCK_THRESHOLD and n_total % num_blocks == 0:
        return _make_mish_decode_multicore(n_total, dtype, num_blocks)
    if n_total <= _SINGLE_BLOCK_THRESHOLD:
        return _make_mish_single_block(n_total, dtype)
    return _make_mish_generic_single_block(n_total, dtype, tile)


class AscendMishKernel(Kernel, UnaryElementwiseFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return (call, lambda: cls(call.n_total, call.dtype))

    def __init__(self, n_total: int, dtype: torch.dtype):
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self._compiled_fn = _make_mish_kernel(n_total, _dtype_str(dtype))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        flat = input.reshape(-1)
        result = self._compiled_fn(flat)
        return result.reshape(input.shape)


tileops.backend.register_kernel_type("MishFwdOp", "ascend", AscendMishKernel)
