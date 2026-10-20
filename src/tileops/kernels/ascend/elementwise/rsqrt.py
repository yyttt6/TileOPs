"""Ascend 950 implementations for RsqrtFwdOp."""

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
    """Persistent-grid block count for the pipelined, large-n path."""
    try:
        queried = get_num_vector_cores(device_id=0)
    except Exception:
        queried = None
    if not queried:
        return fallback
    return _floor_pow2(int(queried))


_TILE_BY_DTYPE = {"float16": 16384, "bfloat16": 16384, "float32": 8192}
_NUM_STAGES = 2
_THREADS = 2048


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


@T.macro
def _rsqrt_body(x_ub, y_ub, n: int, dtype_str: str, threads: int) -> None:
    """``y_ub[:n] = rsqrt(x_ub[:n])``, compiler-vectorized over ``threads`` lanes."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n):
            y_ub[i] = T.cast(T.rsqrt(T.cast(x_ub[i], "float32")), dtype_str)


@functools.lru_cache(maxsize=64)
def _make_rsqrt_single_block(n_total: int, dtype_str: str):
    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((n_total,), dtype_str)
            y_ub = T.alloc_shared((n_total,), dtype_str)
            T.copy(X, x_ub)
            _rsqrt_body(x_ub, y_ub, n_total, dtype_str, threads=min(_THREADS, n_total))
            T.copy(y_ub, Y)

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _make_rsqrt_persistent(
    n_total: int, dtype_str: str, num_blocks: int, tile: int, num_stages: int
):
    block_span = num_blocks * tile
    assert n_total % block_span == 0, (
        f"_make_rsqrt_persistent requires n_total ({n_total}) to divide num_blocks*tile ({block_span}) exactly; the caller routes other shapes elsewhere"
    )
    iterations = n_total // block_span
    threads = min(_THREADS, tile)

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
        with T.Kernel(num_blocks) as bx:
            x_ub = T.alloc_shared((tile,), dtype_str)
            y_ub = T.alloc_shared((tile,), dtype_str)
            T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin = (step * num_blocks + bx) * tile
                T.copy(X[begin : begin + tile], x_ub)
                _rsqrt_body(x_ub, y_ub, tile, dtype_str, threads)
                T.copy(y_ub, Y[begin : begin + tile])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _make_rsqrt_kernel(n_total: int, dtype_str: str):
    """Pick the tiling for ``n_total`` elements: persistent pipelined grid, or the
    small-shape single-block fallback."""
    num_blocks = _calibrated_num_blocks()
    tile = _TILE_BY_DTYPE[dtype_str]
    block_span = num_blocks * tile
    if n_total % block_span == 0:
        return _make_rsqrt_persistent(n_total, dtype_str, num_blocks, tile, _NUM_STAGES)
    if n_total <= tile:
        return _make_rsqrt_single_block(n_total, dtype_str)
    raise NotImplementedError(
        f"AscendRsqrtKernel: n_total={n_total} divides neither into {num_blocks}x{tile} pipelined blocks nor fits the single-block fallback (<= {tile} elements)."
    )


class AscendRsqrtKernel(Kernel, UnaryElementwiseFwdInterface):
    """Rsqrt on Ascend NPUs: ``y = 1/sqrt(x)``."""

    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return (call, lambda: cls(call.n_total, call.dtype))

    def __init__(self, n_total: int, dtype: torch.dtype):
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self._compiled_fn = _make_rsqrt_kernel(n_total, _dtype_str(dtype))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        flat = input.reshape(-1)
        result = self._compiled_fn(flat)
        return result.reshape(input.shape)


tileops.backend.register_kernel_type("RsqrtFwdOp", "ascend", AscendRsqrtKernel)
