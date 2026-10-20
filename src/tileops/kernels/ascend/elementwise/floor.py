"""Ascend 950 implementations for FloorFwdOp."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_vector_cores

import tileops.backend
from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryElementwiseFwdInterface
from tileops.kernels.elementwise.math_unary import INT_DTYPES
from tileops.kernels.kernel_base import Entry, Kernel

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _floor_pow2(n: int) -> int:
    """Largest power of two ``<= n`` (``n >= 1``)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


@functools.lru_cache(maxsize=1)
def _calibrated_num_blocks(fallback: int = 64) -> int:
    """Persistent-grid block count for the large, bandwidth-bound tiled path."""
    try:
        queried = get_num_vector_cores(device_id=0)
    except Exception:
        queried = None
    if not queried:
        return fallback
    return _floor_pow2(int(queried))


_UB_BUDGET_BYTES = 216 * 1024
_NUM_STAGES = 3
_MAX_THREADS = 2048


def _tile_elems_for_dtype(dtype_str: str) -> int:
    """The largest power-of-two element count keeping the tiled path's UB use in budget."""
    itemsize = getattr(torch, dtype_str).itemsize
    max_elems = _UB_BUDGET_BYTES // (2 * _NUM_STAGES * itemsize)
    return _floor_pow2(max_elems)


_SINGLE_BLOCK_THRESHOLD = 8192


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


@T.macro
def _floor_body(x_ub, y_ub, n: int, threads: int) -> None:
    """``y_ub[:n] = floor(x_ub[:n])``, computed in fp32, compiler-vectorized over
    ``threads`` lanes."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n):
            y_ub[i] = T.cast(T.floor(T.cast(x_ub[i], "float32")), y_ub.dtype)


@functools.lru_cache(maxsize=64)
def _make_floor_single_block(n_total: int, dtype: str):
    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((n_total,), dtype)
            y_ub = T.alloc_shared((n_total,), dtype)
            T.copy(X, x_ub)
            _floor_body(x_ub, y_ub, n_total, threads=min(_MAX_THREADS, n_total))
            T.copy(y_ub, Y)

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _make_floor_persistent(n_total: int, dtype: str, num_blocks: int, tile: int, num_stages: int):
    block_span = num_blocks * tile
    assert n_total % block_span == 0, (
        f"_make_floor_persistent requires n_total ({n_total}) to divide num_blocks*tile ({block_span}) exactly; the caller routes other shapes elsewhere"
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
                _floor_body(x_ub, y_ub, tile, threads=min(_MAX_THREADS, tile))
                T.copy(y_ub, Y[begin : begin + tile])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _make_floor_generic_single_block(n_total: int, dtype: str, tile: int):
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
                    _floor_body(x_ub, y_ub, tile, threads=min(_MAX_THREADS, tile))
                    T.copy(y_ub, Y[begin : begin + tile])
            if remainder:
                xt_ub = T.alloc_shared((remainder,), dtype)
                yt_ub = T.alloc_shared((remainder,), dtype)
                begin = full_tiles * tile
                T.copy(X[begin : begin + remainder], xt_ub)
                _floor_body(xt_ub, yt_ub, remainder, threads=min(128, remainder))
                T.copy(yt_ub, Y[begin : begin + remainder])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _make_floor_kernel(n_total: int, dtype_str: str):
    """Pick the tiling for ``n_total`` elements: single-block, persistent, or fallback."""
    if n_total <= _SINGLE_BLOCK_THRESHOLD:
        return _make_floor_single_block(n_total, dtype_str)
    tile = _tile_elems_for_dtype(dtype_str)
    num_blocks = _calibrated_num_blocks()
    block_span = num_blocks * tile
    if n_total % block_span == 0:
        return _make_floor_persistent(n_total, dtype_str, num_blocks, tile, _NUM_STAGES)
    return _make_floor_generic_single_block(n_total, dtype_str, tile)


class AscendFloorKernel(Kernel, UnaryElementwiseFwdInterface):
    """Floor on Ascend NPUs: ``y = floor(x)``, computed in fp32 for all float dtypes."""

    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: ElementwiseCall) -> "str | None":
        if call.dtype not in _FLOAT_DTYPES and call.dtype not in INT_DTYPES:
            return f"serves {_FLOAT_DTYPES + INT_DTYPES}, not {call.dtype}"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return (call, lambda: cls(call.n_total, call.dtype))

    def __init__(self, n_total: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self._is_int = dtype in INT_DTYPES
        if self._is_int:
            self._compiled_fn = None
        else:
            self._compiled_fn = _make_floor_kernel(n_total, _dtype_str(dtype))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._is_int:
            return input.clone().contiguous()
        flat = input.reshape(-1)
        result = self._compiled_fn(flat)
        return result.reshape(input.shape)


tileops.backend.register_kernel_type("FloorFwdOp", "ascend", AscendFloorKernel)
