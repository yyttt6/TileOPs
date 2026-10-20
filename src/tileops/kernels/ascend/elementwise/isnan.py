"""Ascend 950 implementations for IsnanFwdOp."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_vector_cores

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryPredicateFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["AscendIsnanKernel"]
FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
INT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
NON_FLOAT_DTYPES = INT_DTYPES + (torch.bool,)
_TILE_FP32 = 8192
_TILE_FP16_BF16 = 16384
_NUM_STAGES_FP32 = 2
_NUM_STAGES_FP16_BF16 = 2
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


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _tile_for_dtype(dtype: str) -> int:
    return _TILE_FP32 if dtype == "float32" else _TILE_FP16_BF16


def _num_stages_for_dtype(dtype: str) -> int:
    return _NUM_STAGES_FP32 if dtype == "float32" else _NUM_STAGES_FP16_BF16


@T.macro
def _isnan_body(x_ub, y_ub, n: int, threads: int) -> None:
    """``y_ub[:n] = isnan(x_ub[:n])`` as bool, compiler-vectorized over ``threads``."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n):
            y_ub[i] = T.isnan(T.cast(x_ub[i], "float32"))


@functools.lru_cache(maxsize=64)
def _make_isnan_persistent(n_total: int, dtype: str, num_blocks: int, tile: int, num_stages: int):
    block_span = num_blocks * tile
    assert n_total % block_span == 0, (
        f"_make_isnan_persistent requires n_total ({n_total}) to divide num_blocks*tile ({block_span}) exactly; the caller routes other shapes elsewhere"
    )
    iterations = n_total // block_span

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), "bool")):
        with T.Kernel(num_blocks) as bx:
            x_ub = T.alloc_shared((tile,), dtype)
            y_ub = T.alloc_shared((tile,), "bool")
            T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin = (step * num_blocks + bx) * tile
                T.copy(X[begin : begin + tile], x_ub)
                _isnan_body(x_ub, y_ub, tile, threads=min(_MAX_THREADS, tile))
                T.copy(y_ub, Y[begin : begin + tile])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _make_isnan_generic_single_block(n_total: int, dtype: str, tile: int):
    full_tiles, remainder = divmod(n_total, tile)

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), "bool")):
        with T.Kernel(1):
            if full_tiles:
                x_ub = T.alloc_shared((tile,), dtype)
                y_ub = T.alloc_shared((tile,), "bool")
                T.annotate_buffer_versions({x_ub: 2, y_ub: 2})
                for step in T.Pipelined(full_tiles, num_stages=2):
                    begin = step * tile
                    T.copy(X[begin : begin + tile], x_ub)
                    _isnan_body(x_ub, y_ub, tile, threads=min(_MAX_THREADS, tile))
                    T.copy(y_ub, Y[begin : begin + tile])
            if remainder:
                xt_ub = T.alloc_shared((remainder,), dtype)
                yt_ub = T.alloc_shared((remainder,), "bool")
                begin = full_tiles * tile
                T.copy(X[begin : begin + remainder], xt_ub)
                _isnan_body(xt_ub, yt_ub, remainder, threads=min(128, remainder))
                T.copy(yt_ub, Y[begin : begin + remainder])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _make_isnan_kernel(n_total: int, dtype: str):
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")
    tile = _tile_for_dtype(dtype)
    num_stages = _num_stages_for_dtype(dtype)
    num_blocks = _calibrated_num_blocks()
    block_span = num_blocks * tile
    if n_total % block_span == 0:
        return _make_isnan_persistent(n_total, dtype, num_blocks, tile, num_stages)
    return _make_isnan_generic_single_block(n_total, dtype, tile)


class AscendIsnanKernel(Kernel, UnaryPredicateFwdInterface):
    """IsNaN on Ascend NPUs: ``y = isnan(x)``, a new contiguous ``torch.bool`` tensor."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

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
        self._compiled_fn = (
            _make_isnan_kernel(n_total, _dtype_str(dtype)) if dtype in FLOAT_DTYPES else None
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return ``isnan(input)`` elementwise; nothing is written in place."""
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got {input.device}. Another target's backend serves other devices."
            )
        if self.dtype not in FLOAT_DTYPES:
            return torch.zeros(input.shape, dtype=torch.bool, device=input.device)
        flat_input = input.reshape(-1)
        if flat_input.numel() != self.n_total:
            raise ValueError(
                f"{type(self).__name__} was built for n_total={self.n_total}, got {flat_input.numel()} elements"
            )
        flat_output = self._compiled_fn(flat_input)
        return flat_output.reshape(input.shape)


register_kernel_type("IsnanFwdOp", "ascend", AscendIsnanKernel)
