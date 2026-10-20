"""Ascend 950 implementations for CeilFwdOp."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_vector_cores

import tileops.backend
from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
_UB_BUDGET_BYTES = 216 * 1024
_NUM_STAGES = 3
_FALLBACK_VECTOR_CORES = 64
_MAX_THREADS = 2048


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _floor_pow2(n: int) -> int:
    """Largest power of two ``<= n`` (``n >= 1``)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


@functools.lru_cache(maxsize=1)
def _calibrated_num_blocks(fallback: int = _FALLBACK_VECTOR_CORES) -> int:
    """Persistent-grid block count for the tiled, bandwidth-bound path."""
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


@T.macro
def _ceil_body(x_ub, y_ub, n: int, threads: int) -> None:
    """``y_ub[:n] = ceil(x_ub[:n])``, computed in fp32, compiler-vectorized over ``threads``."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n):
            y_ub[i] = T.cast(T.ceil(T.cast(x_ub[i], "float32")), y_ub.dtype)


@functools.lru_cache(maxsize=16)
def _build_tiled(n_total: int, dtype_str: str, num_blocks: int, tile_elems: int, num_stages: int):
    """Persistent-block, ``T.Pipelined`` main loop over an exact multiple of ``tile_elems``."""
    block_span = num_blocks * tile_elems
    assert n_total % block_span == 0, (
        f"_build_tiled requires n_total ({n_total}) to divide num_blocks*tile_elems ({block_span}) exactly"
    )
    iterations = n_total // block_span
    threads = min(tile_elems, _MAX_THREADS)

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
        with T.Kernel(num_blocks) as bx:
            x_ub = T.alloc_shared((tile_elems,), dtype_str)
            y_ub = T.alloc_shared((tile_elems,), dtype_str)
            T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin = (step * num_blocks + bx) * tile_elems
                T.copy(X[begin : begin + tile_elems], x_ub)
                _ceil_body(x_ub, y_ub, tile_elems, threads=threads)
                T.copy(y_ub, Y[begin : begin + tile_elems])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _build_ceil_kernel(n_total: int, dtype_str: str):
    """Build the single persistent-tiled kernel for ``n_total`` elements."""
    tile_elems = _tile_elems_for_dtype(dtype_str)
    num_blocks = _calibrated_num_blocks()
    block_span = num_blocks * tile_elems
    if n_total % block_span == 0:
        return _build_tiled(n_total, dtype_str, num_blocks, tile_elems, _NUM_STAGES)
    raise NotImplementedError(
        f"AscendCeilKernel: n_total={n_total} dtype={dtype_str} does not divide num_blocks*tile_elems={block_span}"
    )


class AscendCeilKernel(Kernel, UnaryElementwiseFwdInterface):
    """Ascend NPU implementation of ``CeilFwdOp``: ``y = ceil(x)``."""

    devices = frozenset({"npu"})

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
        self._is_int = dtype in _INT_DTYPES
        self._compiled_fn = None if self._is_int else _build_ceil_kernel(n_total, _dtype_str(dtype))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._is_int:
            return input.clone()
        flat = input.reshape(-1)
        result = self._compiled_fn(flat)
        return result.reshape(input.shape)


tileops.backend.register_kernel_type("CeilFwdOp", "ascend", AscendCeilKernel)
