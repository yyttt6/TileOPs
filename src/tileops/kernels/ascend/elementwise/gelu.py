"""Ascend 950 implementations for GeluFwdOp."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise._erf import gelu
from tileops.kernels.elementwise.call_spec import GeluCall, UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Kernel

__all__ = ["AscendGeluKernel"]
_NUM_BLOCKS = 64
_TILE = 8192
_NUM_STAGES = 2
_THREADS = 2048
_SINGLE_SHOT_MAX_ELEMENTS = 65536


def _mode_for(n_total: int) -> str:
    if n_total % (_NUM_BLOCKS * _TILE) == 0:
        return "pipelined"
    if n_total <= _SINGLE_SHOT_MAX_ELEMENTS and n_total % _NUM_BLOCKS == 0:
        return "decode_multicore"
    if n_total <= _SINGLE_SHOT_MAX_ELEMENTS:
        return "single_shot"
    raise NotImplementedError(
        f"AscendGeluKernel: n_total={n_total} must divide into {_NUM_BLOCKS}x{_TILE} "
        f"pipelined blocks or fit the {_SINGLE_SHOT_MAX_ELEMENTS}-element decode path."
    )


@functools.lru_cache(maxsize=None)
def _build_pipelined(n_total: int, dtype_str: str, num_blocks: int, tile: int, num_stages: int):
    """Multi-block, ``T.Pipelined`` GELU kernel for the two prefill cases."""
    assert n_total % (num_blocks * tile) == 0, (
        f"_build_pipelined requires n_total={n_total} to divide evenly into num_blocks={num_blocks} x tile={tile}"
    )
    iterations = n_total // (num_blocks * tile)
    threads = min(tile, _THREADS)

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
        with T.Kernel(num_blocks) as bx:
            x_ub = T.alloc_shared((tile,), dtype_str)
            y_ub = T.alloc_shared((tile,), dtype_str)
            T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin = (step * num_blocks + bx) * tile
                T.copy(X[begin : begin + tile], x_ub)
                with T.SimtVF(threads=threads):
                    for i in T.Parallel(tile):
                        y_ub[i] = T.cast(gelu(x_ub[i], dtype_str), dtype_str)
                T.copy(y_ub, Y[begin : begin + tile])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=None)
def _build_single_shot(n_total: int, dtype_str: str):
    """Single-block, no-pipelining GELU kernel; fallback for a small shape that does not
    divide evenly across ``_NUM_BLOCKS`` AIV cores."""
    threads = min(n_total, _THREADS)

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((n_total,), dtype_str)
            y_ub = T.alloc_shared((n_total,), dtype_str)
            T.copy(X, x_ub)
            with T.SimtVF(threads=threads):
                for i in T.Parallel(n_total):
                    y_ub[i] = T.cast(gelu(x_ub[i], dtype_str), dtype_str)
            T.copy(y_ub, Y)

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=None)
def _build_decode_multicore(n_total: int, dtype_str: str, num_cores: int):
    assert n_total % num_cores == 0, (
        f"_build_decode_multicore requires n_total={n_total} to divide evenly by num_cores={num_cores}; _mode_for should not route here otherwise"
    )
    block = n_total // num_cores
    threads = min(block, _THREADS)

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
        with T.Kernel(num_cores) as core_id:
            x_ub = T.alloc_shared((block,), dtype_str)
            y_ub = T.alloc_shared((block,), dtype_str)
            begin = core_id * block
            T.copy(X[begin : begin + block], x_ub)
            with T.SimtVF(threads=threads):
                for i in T.Parallel(block):
                    y_ub[i] = T.cast(gelu(x_ub[i], dtype_str), dtype_str)
            T.copy(y_ub, Y[begin : begin + block])

    return tilelang.compile(main, target="ascend", out_idx=-1)


class AscendGeluKernel(Kernel, UnaryElementwiseFwdInterface):
    """Ascend NPU element-wise GELU, exact-erf form only."""

    devices = frozenset({"npu"})

    def __init__(self, call: GeluCall, *, device_index: "int | None" = None) -> None:
        super().__init__(device_index=device_index)
        self.n_total = call.n_total
        self.dtype = call.dtype
        self.approximate = call.approximate
        dtype_str = self.dtype_to_str(call.dtype)
        mode = _mode_for(call.n_total)
        if mode == "pipelined":
            self.kernel = _build_pipelined(call.n_total, dtype_str, _NUM_BLOCKS, _TILE, _NUM_STAGES)
        elif mode == "decode_multicore":
            self.kernel = _build_decode_multicore(call.n_total, dtype_str, _NUM_BLOCKS)
        else:
            self.kernel = _build_single_shot(call.n_total, dtype_str)

    @classmethod
    def refusal(cls, call: GeluCall) -> "str | None":
        if call.approximate != "none":
            return f"evaluates the erf form, not approximate={call.approximate!r}"
        return super().refusal(call)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return a new tensor, GELU applied element-wise; ``input`` is left untouched."""
        flat = input.reshape(-1)
        result = self.kernel(flat)
        return result.reshape(input.shape)


register_kernel_type("GeluFwdOp", "ascend", AscendGeluKernel)
