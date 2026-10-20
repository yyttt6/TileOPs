"""Ascend 950 implementations for SiluAndMulFwdOp."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.call_spec import FusedGatedCall, FusedGatedFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["AscendSiluAndMulKernel"]
_TORCH_DTYPE_TO_TL = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}
_NUM_AIV_CORES = 64


def _largest_divisor_leq(n: int, upper_bound: int) -> int:
    """The largest divisor of *n* that is <= *upper_bound* (upper_bound clamped to >= 1)."""
    upper_bound = max(1, min(upper_bound, n))
    for candidate in range(upper_bound, 0, -1):
        if n % candidate == 0:
            return candidate
    return 1


def _pick_block_n(m: int, n: int) -> int:
    """The column tile width."""
    if n <= 0:
        raise ValueError(f"N must be positive, got {n}")
    if m >= _NUM_AIV_CORES:
        return _largest_divisor_leq(n, 8192)
    return _largest_divisor_leq(n, max(1, n // _NUM_AIV_CORES))


def _silu_and_mul_program(m: int, n: int, dtype: str):
    """Build the ``(m, 2*n) -> (m, n)`` SiluAndMul prim_func for one (m, n, dtype)."""
    block_n = _pick_block_n(m, n)
    col_tiles = n // block_n
    total_tiles = m * col_tiles
    num_stages = 2 if total_tiles // _NUM_AIV_CORES >= 2 else 1
    threads = min(1024, block_n)

    @T.prim_func
    def main(X: T.Tensor((m, 2 * n), dtype), Y: T.Tensor((m, n), dtype)):
        with T.Kernel(_NUM_AIV_CORES) as core_id:
            for tile in T.Persistent([total_tiles], _NUM_AIV_CORES, core_id, num_stages=num_stages):
                row = tile // col_tiles
                col = tile % col_tiles * block_n
                gate_ub = T.alloc_shared((block_n,), dtype)
                value_ub = T.alloc_shared((block_n,), dtype)
                out_ub = T.alloc_shared((block_n,), dtype)
                T.annotate_buffer_versions(
                    {gate_ub: num_stages, value_ub: num_stages, out_ub: num_stages}
                )
                T.copy(X[row, col : col + block_n], gate_ub)
                T.copy(X[row, n + col : n + col + block_n], value_ub)
                with T.SimtVF(threads=threads):
                    for i in T.Parallel(block_n):
                        g = T.Cast("float32", gate_ub[i])
                        one = T.cast(1.0, "float32")
                        silu_val = g / (one + T.exp(-g))
                        out_ub[i] = T.Cast(dtype, silu_val) * value_ub[i]
                T.copy(out_ub, Y[row, col : col + block_n])

    return main


@functools.lru_cache(maxsize=None)
def _build_silu_and_mul_kernel(m: int, n: int, dtype_str: str):
    program = _silu_and_mul_program(m, n, dtype_str)
    return tilelang.compile(program, target="ascend", out_idx=-1)


class AscendSiluAndMulKernel(Kernel, FusedGatedFwdInterface):
    """TileLang-Ascend SiluAndMul: ``y = silu(gate) * value``, ``silu`` in fp32."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: FusedGatedCall) -> Entry:
        return (call, lambda: cls(call.m, call.n, call.dtype))

    def __init__(self, m: int, n: int, dtype: torch.dtype):
        super().__init__()
        if dtype not in _TORCH_DTYPE_TO_TL:
            raise ValueError(f"AscendSiluAndMulKernel does not support dtype {dtype!r}")
        self.m = m
        self.n = n
        self.dtype = dtype
        self._dtype_str = _TORCH_DTYPE_TO_TL[dtype]
        self.kernel = _build_silu_and_mul_kernel(m, n, self._dtype_str)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``silu(x[:, :n]) * x[:, n:]``; nothing is written in place."""
        if not x.is_contiguous():
            x = x.contiguous()
        return self.kernel(x)


register_kernel_type("SiluAndMulFwdOp", "ascend", AscendSiluAndMulKernel)
