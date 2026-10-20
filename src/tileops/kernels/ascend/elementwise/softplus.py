"""Ascend 950 implementations for SoftplusFwdOp."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_vector_cores

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.call_spec import SoftplusCall, SoftplusFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["AscendSoftplusKernel"]
_UB_BUDGET_BYTES = 216 * 1024
_NUM_STAGES = 2
_FALLBACK_VECTOR_CORES = 64
_MAX_THREADS = 2048


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


_TUNED_LAYOUTS = {(8388608, "bfloat16"): (1024, 2)}


def _layout_for_call(
    n_total: int, dtype_str: str, beta: float, threshold: float
) -> tuple[int, int]:
    """``(simt_threads, coalesced_width)`` for one full production call."""
    if beta == 1.0 and threshold == 20.0:
        tuned = _TUNED_LAYOUTS.get((n_total, dtype_str))
        if tuned is not None:
            return tuned
    return (_MAX_THREADS, 0)


@T.macro
def _softplus_body(
    x_ub, y_ub, n: int, threads: int, beta: float, threshold: float, coalesced_width: int = 0
) -> None:
    """``y_ub[:n] = softplus(x_ub[:n]; beta, threshold)``, computed in fp32."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(
            n, coalesced_width=None if coalesced_width == 0 else T.int32(coalesced_width)
        ):
            one = T.Cast("float32", 1.0)
            t = T.Cast("float32", threshold)
            wide = T.Cast("float32", x_ub[i])
            if beta == 1.0:
                scaled = wide
                e = T.exp(scaled)
                sp = T.log(one + e)
            else:
                b = T.Cast("float32", beta)
                scaled = wide * b
                e = T.exp(scaled)
                sp = T.log(one + e) / b
            y_ub[i] = T.if_then_else(scaled > t, x_ub[i], T.Cast(x_ub.dtype, sp))


@functools.lru_cache(maxsize=64)
def _build_tiled(
    n_total: int,
    dtype_str: str,
    num_blocks: int,
    tile_elems: int,
    num_stages: int,
    beta: float,
    threshold: float,
    simt_threads: int = _MAX_THREADS,
    coalesced_width: int = 0,
):
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
                _softplus_body(
                    x_ub,
                    y_ub,
                    tile_elems,
                    threads=simt_threads,
                    beta=beta,
                    threshold=threshold,
                    coalesced_width=coalesced_width,
                )
                T.copy(y_ub, Y[begin : begin + tile_elems])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _build_softplus_kernel(n_total: int, dtype_str: str, beta: float, threshold: float):
    """Pick the tiling for ``n_total`` elements: the persistent tiled path only."""
    simt_threads, coalesced_width = _layout_for_call(n_total, dtype_str, beta, threshold)
    tile_elems = _tile_elems_for_dtype(dtype_str)
    num_blocks = _calibrated_num_blocks()
    block_span = num_blocks * tile_elems
    if n_total % block_span == 0:
        return _build_tiled(
            n_total,
            dtype_str,
            num_blocks,
            tile_elems,
            _NUM_STAGES,
            beta,
            threshold,
            simt_threads,
            coalesced_width,
        )
    raise NotImplementedError(
        f"AscendSoftplusKernel: n_total={n_total} dtype={dtype_str} is not a multiple of num_blocks*tile_elems={block_span}; no manifest/test shape should reach this"
    )


class AscendSoftplusKernel(Kernel, SoftplusFwdInterface):
    """Ascend NPU softplus: ``y = log(1 + exp(beta*x)) / beta`` if ``beta*x <= threshold`` else
    ``x``, computed in fp32 and cast back to the storage dtype at the boundary."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: SoftplusCall) -> Entry:
        return (call, lambda: cls(call.n_total, call.dtype, call.beta, call.threshold))

    def __init__(
        self, n_total: int, dtype: torch.dtype, beta: float = 1.0, threshold: float = 20.0
    ) -> None:
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.beta = beta
        self.threshold = threshold
        self.kernel = _build_softplus_kernel(n_total, self.dtype_str, beta, threshold)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply softplus elementwise; nothing is written in place."""
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


register_kernel_type("SoftplusFwdOp", "ascend", AscendSoftplusKernel)
