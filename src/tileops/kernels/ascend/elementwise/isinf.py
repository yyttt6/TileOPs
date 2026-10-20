"""Ascend 950 implementations for IsinfFwdOp."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_vector_cores

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryPredicateFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["AscendIsinfKernel"]
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_INT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_FALLBACK_DTYPES = _INT_DTYPES + (torch.bool,)
_TILE_FOR_DTYPE = {"float16": 8192, "bfloat16": 8192, "float32": 16384}
_CHUNK = 2048
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
def _calibrated_num_blocks() -> int:
    """Persistent-grid block count for the tiled, bandwidth-bound path."""
    try:
        queried = get_num_vector_cores(device_id=0)
    except Exception:
        queried = None
    if not queried:
        return _FALLBACK_VECTOR_CORES
    return _floor_pow2(int(queried))


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


@T.macro
def _isinf_body(x_ub, y_ub, n: int) -> None:
    """``y_ub[:n] = isinf(x_ub[:n])`` as bool, computed in fp32."""
    full_chunks, remainder = divmod(n, _CHUNK)
    for c in range(full_chunks):
        base = c * _CHUNK
        with T.SimtVF(threads=_CHUNK):
            for i in T.Parallel(_CHUNK):
                xf = T.Cast("float32", x_ub[base + i])
                y_ub[base + i] = T.isinf(xf)
    if remainder:
        base = full_chunks * _CHUNK
        with T.SimtVF(threads=remainder):
            for i in T.Parallel(remainder):
                xf = T.Cast("float32", x_ub[base + i])
                y_ub[base + i] = T.isinf(xf)


@functools.lru_cache(maxsize=64)
def _make_isinf_persistent(n_total: int, dtype: str, num_blocks: int, tile: int, num_stages: int):
    block_span = num_blocks * tile
    assert n_total % block_span == 0, (
        f"_make_isinf_persistent requires n_total ({n_total}) to divide num_blocks*tile ({block_span}) exactly; the caller routes other shapes elsewhere"
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
                _isinf_body(x_ub, y_ub, tile)
                T.copy(y_ub, Y[begin : begin + tile])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _make_isinf_generic_single_block(n_total: int, dtype: str, tile: int):
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
                    _isinf_body(x_ub, y_ub, tile)
                    T.copy(y_ub, Y[begin : begin + tile])
            if remainder:
                xt_ub = T.alloc_shared((remainder,), dtype)
                yt_ub = T.alloc_shared((remainder,), "bool")
                begin = full_tiles * tile
                T.copy(X[begin : begin + remainder], xt_ub)
                _isinf_body(xt_ub, yt_ub, remainder)
                T.copy(yt_ub, Y[begin : begin + remainder])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def _make_isinf_kernel(n_total: int, dtype: str):
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")
    num_blocks = _calibrated_num_blocks()
    tile = _TILE_FOR_DTYPE[dtype]
    block_span = num_blocks * tile
    if n_total % block_span == 0:
        return _make_isinf_persistent(n_total, dtype, num_blocks, tile, _NUM_STAGES)
    return _make_isinf_generic_single_block(n_total, dtype, tile)


class AscendIsinfKernel(Kernel, UnaryPredicateFwdInterface):
    """IsInf on Ascend NPUs: ``y = isinf(x)``, a new contiguous ``torch.bool`` tensor."""

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
            _make_isinf_kernel(n_total, _dtype_str(dtype)) if dtype in _FLOAT_DTYPES else None
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return ``isinf(input)`` elementwise; nothing is written in place."""
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got {input.device}. Another target's backend serves other devices."
            )
        if self.dtype not in _FLOAT_DTYPES:
            return torch.zeros(input.shape, dtype=torch.bool, device=input.device)
        flat_input = input.reshape(-1)
        if flat_input.numel() != self.n_total:
            raise ValueError(
                f"{type(self).__name__} was built for n_total={self.n_total}, got {flat_input.numel()} elements"
            )
        flat_output = self._compiled_fn(flat_input)
        return flat_output.reshape(input.shape)


register_kernel_type("IsinfFwdOp", "ascend", AscendIsinfKernel)
