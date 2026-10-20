"""Ascend 950 implementations for NegFwdOp."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

import tileops.backend
from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
_NATIVE_NPU_NEG_INT_DTYPES = (torch.int8, torch.int32, torch.int64)


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


@T.macro
def _neg_body(x_ub, y_ub, n: int, threads: int) -> None:
    """``y_ub[:n] = -x_ub[:n]``, compiler-vectorized over ``threads`` lanes."""
    with T.SimtVF(threads=threads):
        for i in T.Parallel(n):
            y_ub[i] = -x_ub[i]


@functools.lru_cache(maxsize=16)
def _make_neg_kernel(n_total: int, dtype: str, tile: int):
    num_blocks, num_stages = (64, 2)
    block_span = num_blocks * tile
    assert n_total % block_span == 0, (
        f"_make_neg_kernel requires n_total ({n_total}) to divide num_blocks*tile ({block_span}) exactly"
    )
    iterations = n_total // block_span
    threads = min(tile, 2048)

    @T.prim_func
    def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
        with T.Kernel(num_blocks) as bx:
            x_ub = T.alloc_shared((tile,), dtype)
            y_ub = T.alloc_shared((tile,), dtype)
            T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin = (step * num_blocks + bx) * tile
                T.copy(X[begin : begin + tile], x_ub)
                _neg_body(x_ub, y_ub, tile, threads=threads)
                T.copy(y_ub, Y[begin : begin + tile])

    return tilelang.compile(main, target="ascend", out_idx=-1)


class AscendNegKernel(Kernel, UnaryElementwiseFwdInterface):
    """Ascend NPU implementation of ``NegFwdOp``: ``y = -x``."""

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
        tile = 8192
        self._compiled_fn = (
            None if self._is_int else _make_neg_kernel(n_total, _dtype_str(dtype), tile)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._is_int:
            if input.dtype in _NATIVE_NPU_NEG_INT_DTYPES:
                result = torch.neg(input)
            else:
                result = torch.neg(input.to(torch.int32)).to(input.dtype)
            return result.contiguous()
        flat = input.reshape(-1)
        result = self._compiled_fn(flat)
        return result.reshape(input.shape)


tileops.backend.register_kernel_type("NegFwdOp", "ascend", AscendNegKernel)
