"""Ascend 950 implementations for CumsumFwdOp."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Entry, Kernel, vector_aligned
from tileops.kernels.reduction._primitives import (
    restore_same_shape,
    rows_for_axes,
    torch_dtype_nbytes,
)
from tileops.kernels.reduction.call_spec import CumsumFwdInterface, CumulativeCall

__all__ = ["AscendCumsumKernel"]
_UB_BUDGET_BYTES = 216 * 1024
_UB_SCRATCH_RESERVE_BYTES = 64 * 1024
_MAX_THREADS = 2048
_MAX_ROWS_TIMES_P = 512
_P_CANDIDATES = (512, 256, 128)
_FALLBACK_VECTOR_CORES = 72


def _num_vector_cores() -> int:
    """Real AIV (vector) core count of the current NPU device, re-queried each call."""
    try:
        from tilelang.carver.arch.driver import get_num_vector_cores

        cores = get_num_vector_cores(device_id=0)
        if cores:
            return int(cores)
    except Exception:
        pass
    return _FALLBACK_VECTOR_CORES


def _largest_divisor_leq(n: int, cap: int) -> int:
    """The largest divisor of *n* that does not exceed *cap*; always >= 1 for n >= 1."""
    cap = max(1, min(n, cap))
    for candidate in range(cap, 0, -1):
        if n % candidate == 0:
            return candidate
    return 1


def _pick_tiling(M: int, N: int, itemsize: int, num_cores: int) -> tuple[int, int, int]:
    """Return ``(num_blocks, rows_per_iter, P)`` for one ``AscendCumsumKernel`` build."""
    num_blocks = _largest_divisor_leq(M, num_cores)
    rows_per_group = M // num_blocks
    budget = _UB_BUDGET_BYTES - _UB_SCRATCH_RESERVE_BYTES
    rows_cap = max(1, min(rows_per_group, _MAX_ROWS_TIMES_P // min(_P_CANDIDATES)))
    rows_per_iter = 1
    for candidate in range(rows_cap, 0, -1):
        if rows_per_group % candidate:
            continue
        iterations = rows_per_group // candidate
        stages = 2 if iterations > 1 else 1
        if candidate * N * itemsize * stages <= budget:
            rows_per_iter = candidate
            break
    P = 0
    for candidate in _P_CANDIDATES:
        if N % candidate == 0 and rows_per_iter * candidate <= _MAX_ROWS_TIMES_P:
            P = candidate
            break
    if P == 0:
        cap = max(1, min(N, _MAX_ROWS_TIMES_P // rows_per_iter, _MAX_THREADS // rows_per_iter))
        P = _largest_divisor_leq(N, cap)
    return (num_blocks, rows_per_iter, P)


@functools.lru_cache(maxsize=None)
def _build_cumsum_kernel(
    N: int, dtype_str: str, num_blocks: int, rows_per_iter: int, iterations: int, P: int
):
    """Build the three-phase block-scan program for one ``(N, dtype, tiling)`` instance."""
    chunk = N // P
    threads = rows_per_iter * P
    assert threads <= _MAX_THREADS, f"threads={threads} exceeds the {_MAX_THREADS} SIMT cap"
    assert chunk * P == N, f"P={P} must divide N={N} exactly"

    @T.prim_func
    def main(
        X: T.Tensor((num_blocks * iterations * rows_per_iter, N), dtype_str),
        Y: T.Tensor((num_blocks * iterations * rows_per_iter, N), dtype_str),
    ):
        with T.Kernel(num_blocks) as bx:
            x_ub = T.alloc_shared((rows_per_iter, N), dtype_str)
            totals_ub = T.alloc_shared((rows_per_iter, P), "float32")
            offsets_ub = T.alloc_shared((rows_per_iter, P), "float32")
            if iterations > 1:
                T.annotate_buffer_versions({x_ub: 2, totals_ub: 2, offsets_ub: 2})
                for it in T.Pipelined(iterations, num_stages=2):
                    row0 = (it * num_blocks + bx) * rows_per_iter
                    T.copy(X[row0 : row0 + rows_per_iter, :], x_ub)
                    with T.SimtVF(threads=threads):
                        acc = T.alloc_fragment((rows_per_iter, P), "float32")
                        for r, p in T.Parallel(rows_per_iter, P):
                            acc[r, p] = T.float32(0)
                        for c in range(chunk):
                            for r, p in T.Parallel(rows_per_iter, P):
                                v = T.cast(x_ub[r, p * chunk + c], "float32")
                                acc[r, p] = acc[r, p] + v
                                x_ub[r, p * chunk + c] = T.cast(acc[r, p], dtype_str)
                        T.copy(acc, totals_ub)
                    for r in range(rows_per_iter):
                        running = T.alloc_var("float32", init=0.0)
                        for p in range(P):
                            offsets_ub[r, p] = running
                            running = running + totals_ub[r, p]
                    with T.SimtVF(threads=threads):
                        for c in range(chunk):
                            for r, p in T.Parallel(rows_per_iter, P):
                                v = T.cast(x_ub[r, p * chunk + c], "float32") + offsets_ub[r, p]
                                x_ub[r, p * chunk + c] = T.cast(v, dtype_str)
                    T.copy(x_ub, Y[row0 : row0 + rows_per_iter, :])
            else:
                row0 = bx * rows_per_iter
                T.copy(X[row0 : row0 + rows_per_iter, :], x_ub)
                with T.SimtVF(threads=threads):
                    acc = T.alloc_fragment((rows_per_iter, P), "float32")
                    for r, p in T.Parallel(rows_per_iter, P):
                        acc[r, p] = T.float32(0)
                    for c in range(chunk):
                        for r, p in T.Parallel(rows_per_iter, P):
                            v = T.cast(x_ub[r, p * chunk + c], "float32")
                            acc[r, p] = acc[r, p] + v
                            x_ub[r, p * chunk + c] = T.cast(acc[r, p], dtype_str)
                    T.copy(acc, totals_ub)
                for r in range(rows_per_iter):
                    running = T.alloc_var("float32", init=0.0)
                    for p in range(P):
                        offsets_ub[r, p] = running
                        running = running + totals_ub[r, p]
                with T.SimtVF(threads=threads):
                    for c in range(chunk):
                        for r, p in T.Parallel(rows_per_iter, P):
                            v = T.cast(x_ub[r, p * chunk + c], "float32") + offsets_ub[r, p]
                            x_ub[r, p * chunk + c] = T.cast(v, dtype_str)
                T.copy(x_ub, Y[row0 : row0 + rows_per_iter, :])

    return tilelang.compile(main, target="ascend", out_idx=-1)


class AscendCumsumKernel(Kernel, CumsumFwdInterface):
    """Ascend NPU inclusive prefix sum (cumsum) along a contiguous trailing axis."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: CumulativeCall) -> Entry:
        return (call, lambda: cls(call))

    def __init__(self, call: CumulativeCall) -> None:
        if call.op_kind != "sum":
            raise ValueError(f"AscendCumsumKernel only serves op_kind='sum', got {call.op_kind!r}")
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.M = call.m
        self.N = call.n
        self.dtype = call.dtype
        self.scan_axis = call.axis
        itemsize = torch_dtype_nbytes(self.dtype)
        num_cores = _num_vector_cores()
        num_blocks, rows_per_iter, P = _pick_tiling(self.M, self.N, itemsize, num_cores)
        iterations = self.M // num_blocks // rows_per_iter
        self.tiling = {
            "num_blocks": num_blocks,
            "rows_per_iter": rows_per_iter,
            "iterations": iterations,
            "P": P,
        }
        self.kernel = _build_cumsum_kernel(
            self.N, self.dtype_str, num_blocks, rows_per_iter, iterations, P
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scan ``self.scan_axis`` of *x*; nothing is written in place."""
        if x.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got {x.device}. Another target's backend serves other devices."
            )
        x = vector_aligned(x)
        in_shape = tuple(x.shape)
        axes = (self.scan_axis,)
        rows = rows_for_axes(x, axes)
        if tuple(rows.shape) != (self.M, self.N):
            raise ValueError(
                f"{type(self).__name__} was built for an ({self.M}, {self.N}) row layout, got {tuple(rows.shape)}"
            )
        y = self.kernel(rows)
        return restore_same_shape(y, in_shape, axes)


register_kernel_type("CumsumFwdOp", "ascend", AscendCumsumKernel)
