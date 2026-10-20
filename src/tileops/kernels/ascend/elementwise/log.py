"""Ascend 950 implementations for LogFwdOp."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_vector_cores

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["AscendLogKernel"]
_UB_BUDGET_BYTES = 216 * 1024
_NUM_STAGES = 3
_FALLBACK_VECTOR_CORES = 64
_MAX_THREADS = 2048
_FP32_MIN_NORMAL = 1.1754943508222875e-38


def _floor_pow2(n: int) -> int:
    """Largest power of two ``<= n`` (``n >= 1``)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


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
def _log_body(x_ub, y_ub, n: int, threads: int) -> None:
    """``y_ub[:n] = log(x_ub[:n])``, computed in fp32, compiler-vectorized over ``threads``
    lanes."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n):
            y_ub[i] = T.cast(T.log(T.cast(x_ub[i], "float32")), y_ub.dtype)


@T.macro
def _log_body_edge(x_ub, y_ub, n: int, threads: int) -> None:
    """``y_ub[:n] = log(x_ub[:n])``, with a subnormal-fp32 flush-to-zero before the call."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n):
            x32 = T.cast(x_ub[i], "float32")
            flushed = T.if_then_else(T.abs(x32) < T.float32(_FP32_MIN_NORMAL), T.float32(0.0), x32)
            y_ub[i] = T.cast(T.log(flushed), y_ub.dtype)


@functools.lru_cache(maxsize=64)
def _build_single_block(n_total: int, dtype_str: str):
    """One block, one untiled GM->UB->compute->UB->GM pass over ``n_total`` elements."""

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((n_total,), dtype_str)
            y_ub = T.alloc_shared((n_total,), dtype_str)
            T.copy(X, x_ub)
            _log_body_edge(x_ub, y_ub, n_total, threads=min(n_total, _MAX_THREADS))
            T.copy(y_ub, Y)

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _build_tiled(n_total: int, dtype_str: str, num_blocks: int, tile_elems: int, num_stages: int):
    """Persistent-block, pipelined main loop over an exact multiple of ``tile_elems``."""
    block_span = num_blocks * tile_elems
    assert n_total % block_span == 0, (
        f"_build_tiled requires n_total ({n_total}) to divide num_blocks*tile_elems ({block_span}) exactly; the caller routes other shapes elsewhere"
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
                _log_body(x_ub, y_ub, tile_elems, threads=min(tile_elems, _MAX_THREADS))
                T.copy(y_ub, Y[begin : begin + tile_elems])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _build_log_kernel(n_total: int, dtype_str: str):
    """Pick the tiling for ``n_total`` elements: persistent tiled path, or single-block
    fallback."""
    tile_elems = _tile_elems_for_dtype(dtype_str)
    if n_total < tile_elems:
        return _build_single_block(n_total, dtype_str)
    num_blocks = _calibrated_num_blocks()
    block_span = num_blocks * tile_elems
    if n_total % block_span == 0:
        return _build_tiled(n_total, dtype_str, num_blocks, tile_elems, _NUM_STAGES)
    if n_total == tile_elems:
        return _build_single_block(n_total, dtype_str)
    raise NotImplementedError(
        f"AscendLogKernel: n_total={n_total} dtype={dtype_str} fits neither the tiled path (needs a multiple of num_blocks*tile_elems={block_span}) nor the single-block fallback (needs n_total<={tile_elems}); no manifest/test shape should reach this"
    )


class AscendLogKernel(Kernel, UnaryElementwiseFwdInterface):
    """Ascend NPU log: ``y = log(x)``, computed in fp32 for every storage dtype."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return (call, lambda: cls(call.n_total, call.dtype))

    def __init__(self, n_total: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.kernel = _build_log_kernel(n_total, self.dtype_str)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply log elementwise; nothing is written in place."""
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


register_kernel_type("LogFwdOp", "ascend", AscendLogKernel)
