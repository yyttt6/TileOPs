"""Ascend 950 implementations for DropoutFwdOp."""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.dropout import DropoutCall, DropoutFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["AscendDropoutKernel"]
_NUM_BLOCKS = 64
_NUM_STAGES = 2
_TILE_BY_DTYPE = {"float16": 8192, "bfloat16": 8192, "float32": 4096}
_THREADS_BY_DTYPE = {"float16": 2048, "bfloat16": 2048, "float32": 1024}


@T.macro
def _dropout_iter_body(
    x_ub, y_ub, n: int, threads: int, seed: int, p: float, scale: float, step, num_blocks: int, bx
) -> None:
    """One tile of inverted dropout with a step-folded RNG stream."""
    with T.SimtVF(threads=threads):
        tx = T.get_thread_binding()
        seq = (step * num_blocks + bx) * threads + tx
        T.rng_init(seed, seq=seq, off=0)
        for i in T.Parallel(n):
            r = T.rng_rand_float(dist="uniform")
            keep = r >= T.cast(p, "float32")
            y_ub[i] = T.if_then_else(
                keep,
                T.cast(x_ub[i] * T.cast(scale, x_ub.dtype), x_ub.dtype),
                T.cast(0.0, x_ub.dtype),
            )


@functools.lru_cache(maxsize=64)
def _build_persistent(
    n_total: int, dtype_str: str, p: float, seed: int, num_blocks: int, tile: int, num_stages: int
):
    scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0
    block_span = num_blocks * tile
    assert n_total % block_span == 0, (
        f"_build_persistent requires n_total ({n_total}) to divide num_blocks*tile ({block_span}) exactly; the caller routes other shapes elsewhere"
    )
    iterations = n_total // block_span
    threads = _THREADS_BY_DTYPE[dtype_str]

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
        with T.Kernel(num_blocks) as bx:
            x_ub = T.alloc_shared((tile,), dtype_str)
            y_ub = T.alloc_shared((tile,), dtype_str)
            T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin = (step * num_blocks + bx) * tile
                T.copy(X[begin : begin + tile], x_ub)
                _dropout_iter_body(x_ub, y_ub, tile, threads, seed, p, scale, step, num_blocks, bx)
                T.copy(y_ub, Y[begin : begin + tile])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _build_generic_single_block(n_total: int, dtype_str: str, p: float, seed: int, tile: int):
    """Single-core fallback for an ``n_total`` the persistent tiling does not evenly divide."""
    scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0
    full_tiles, remainder = divmod(n_total, tile)
    threads_main = min(tile, 2048)

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
        with T.Kernel(1):
            if full_tiles:
                x_ub = T.alloc_shared((tile,), dtype_str)
                y_ub = T.alloc_shared((tile,), dtype_str)
                T.annotate_buffer_versions({x_ub: 2, y_ub: 2})
                for step in T.Pipelined(full_tiles, num_stages=2):
                    begin = step * tile
                    T.copy(X[begin : begin + tile], x_ub)
                    _dropout_iter_body(x_ub, y_ub, tile, threads_main, seed, p, scale, step, 1, 0)
                    T.copy(y_ub, Y[begin : begin + tile])
            if remainder:
                xt_ub = T.alloc_shared((remainder,), dtype_str)
                yt_ub = T.alloc_shared((remainder,), dtype_str)
                begin = full_tiles * tile
                T.copy(X[begin : begin + remainder], xt_ub)
                threads_tail = min(remainder, 2048)
                _dropout_iter_body(
                    xt_ub, yt_ub, remainder, threads_tail, seed, p, scale, full_tiles, 1, 0
                )
                T.copy(yt_ub, Y[begin : begin + remainder])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _make_dropout_kernel(n_total: int, dtype_str: str, p: float, seed: int):
    """Pick the persistent-grid or single-block-fallback tiling for ``n_total`` elements."""
    tile = _TILE_BY_DTYPE[dtype_str]
    block_span = _NUM_BLOCKS * tile
    if n_total % block_span == 0:
        return _build_persistent(n_total, dtype_str, p, seed, _NUM_BLOCKS, tile, _NUM_STAGES)
    return _build_generic_single_block(n_total, dtype_str, p, seed, tile)


class AscendDropoutKernel(Kernel, DropoutFwdInterface):
    """Ascend NPU inverted dropout: ``y = where(rand() >= p, x / (1 - p), 0)``."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: DropoutCall) -> Entry:
        return (call, lambda: cls(call.count, call.dtype, call.p, call.seed))

    def __init__(
        self, n_total: int, dtype: torch.dtype, p: float, seed: int, config: Optional[dict] = None
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.p = p
        self.seed = seed
        self._compiled_fn = _make_dropout_kernel(n_total, self.dtype_to_str(dtype), p, seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run inverted dropout on the flat ``(call.count,)`` input ``x``; see
        ``DropoutFwdInterface.forward`` for the exact contract."""
        if x.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got a tensor on {x.device}. Another target's backend serves other devices."
            )
        return self._compiled_fn(x.reshape(-1))


register_kernel_type("DropoutFwdOp", "ascend", AscendDropoutKernel)
