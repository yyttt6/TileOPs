"""Ascend 950 implementations for ReciprocalFwdOp."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_vector_cores

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.call_spec import ElementwiseCall, ReciprocalFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["AscendReciprocalKernel"]
INT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_UB_BUDGET_BYTES = 216 * 1024
_NUM_STAGES = 3
_FALLBACK_VECTOR_CORES = 72
_MAX_THREADS = 2048


def _floor_pow2(n: int) -> int:
    """Largest power of two ``<= n`` (``n >= 1``)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


@functools.lru_cache(maxsize=1)
def _calibrated_num_blocks() -> int:
    """Persistent-grid block count for the pipelined, large-``n`` path."""
    try:
        queried = get_num_vector_cores(device_id=0)
    except Exception:
        queried = None
    if not queried:
        return _FALLBACK_VECTOR_CORES
    return _floor_pow2(int(queried))


def _tile_elems_for_dtype(dtype_str: str) -> int:
    """The largest power-of-two element count keeping the persistent path's UB use in budget."""
    itemsize = getattr(torch, dtype_str).itemsize
    max_elems = _UB_BUDGET_BYTES // (2 * _NUM_STAGES * itemsize)
    tile = 1
    while tile * 2 <= max_elems:
        tile *= 2
    return tile


def _choose_num_blocks(num_full_tiles: int, vector_cores: int) -> int:
    """The largest divisor of ``num_full_tiles`` that does not exceed ``vector_cores``."""
    cap = max(1, min(num_full_tiles, vector_cores))
    for candidate in range(cap, 0, -1):
        if num_full_tiles % candidate == 0:
            return candidate
    return 1


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


@T.macro
def _reciprocal_body(x_ub, y_ub, n: int, dtype_str: str, threads: int) -> None:
    """``y_ub[:n] = 1 / x_ub[:n]``, computed in fp32, compiler-vectorized over ``threads``."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n):
            one = T.cast(1.0, "float32")
            y_ub[i] = T.cast(one / T.cast(x_ub[i], "float32"), dtype_str)


@functools.lru_cache(maxsize=64)
def _build_single_block(n_total: int, dtype_str: str):
    """One block, one untiled GM->UB->compute->UB->GM pass over ``n_total`` elements."""

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((n_total,), dtype_str)
            y_ub = T.alloc_shared((n_total,), dtype_str)
            T.copy(X, x_ub)
            _reciprocal_body(x_ub, y_ub, n_total, dtype_str, threads=min(n_total, _MAX_THREADS))
            T.copy(y_ub, Y)

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _build_persistent(n_total: int, dtype_str: str, num_blocks: int, tile: int, num_stages: int):
    """Persistent multi-block grid with ``T.Pipelined`` staging."""
    block_span = num_blocks * tile
    assert n_total % block_span == 0, (
        f"_build_persistent requires n_total ({n_total}) to divide num_blocks*tile ({block_span}) exactly; the caller routes other shapes elsewhere"
    )
    iterations = n_total // block_span
    threads = min(tile, _MAX_THREADS)

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
        with T.Kernel(num_blocks) as bx:
            x_ub = T.alloc_shared((tile,), dtype_str)
            y_ub = T.alloc_shared((tile,), dtype_str)
            T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin = (step * num_blocks + bx) * tile
                T.copy(X[begin : begin + tile], x_ub)
                _reciprocal_body(x_ub, y_ub, tile, dtype_str, threads)
                T.copy(y_ub, Y[begin : begin + tile])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _make_reciprocal_kernel(n_total: int, dtype_str: str):
    """Pick the tiling for ``n_total`` elements of ``dtype_str``."""
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")
    tile = _tile_elems_for_dtype(dtype_str)
    if n_total <= tile:
        return _build_single_block(n_total, dtype_str)
    if n_total % tile != 0:
        raise NotImplementedError(
            f"AscendReciprocalKernel: n_total={n_total} is larger than tile={tile} for dtype={dtype_str} but does not divide it evenly"
        )
    num_full_tiles = n_total // tile
    num_blocks = _choose_num_blocks(num_full_tiles, _calibrated_num_blocks())
    return _build_persistent(n_total, dtype_str, num_blocks, tile, _NUM_STAGES)


class AscendReciprocalKernel(Kernel, ReciprocalFwdInterface):
    """Reciprocal on Ascend NPUs: ``y = 1/x``, plain fp32 divide, cast back to the storage dtype."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: ElementwiseCall) -> "str | None":
        if call.dtype in FLOAT_DTYPES or call.dtype in INT_DTYPES:
            return None
        return f"serves {FLOAT_DTYPES + INT_DTYPES}, not {call.dtype}"

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        dtype = torch.float32 if call.dtype in INT_DTYPES else call.dtype
        return (call, lambda: cls(call.n_total, dtype))

    def __init__(self, n_total: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.kernel = _make_reciprocal_kernel(n_total, _dtype_str(dtype))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return ``1 / input`` elementwise; nothing is written in place."""
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got {input.device}. Another target's backend serves other devices."
            )
        if input.dtype != self.dtype:
            input = input.to(self.dtype)
        flat_input = input.reshape(-1)
        if flat_input.numel() != self.n_total:
            raise ValueError(
                f"{type(self).__name__} was built for n_total={self.n_total}, got {flat_input.numel()} elements"
            )
        flat_output = self.kernel(flat_input)
        return flat_output.reshape(input.shape)


register_kernel_type("ReciprocalFwdOp", "ascend", AscendReciprocalKernel)
