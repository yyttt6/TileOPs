"""Ascend 950 implementations for GeluTanhAndMulFwdOp."""

from __future__ import annotations

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.kernels.constants import GELU_TANH_COEFF, SQRT_2_OVER_PI
from tileops.kernels.elementwise.call_spec import FusedGatedCall, FusedGatedFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

_ASCEND_AVAILABLE = True
__all__ = ["AscendGeluTanhAndMulKernel"]
_UB_BYTES_BUDGET_PER_BUFFER = 32768
_BLOCK_M_TARGET = 16
_ELEM_BYTES: dict[torch.dtype, int] = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}
_DIVISORS_TO_TRY = (2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)


def _pick_tile(m: int, n: int, elem_bytes: int) -> tuple[int, int]:
    """(block_m, block_n): general for any (m, n), not tuned only for N=14336."""
    block_m = min(m, _BLOCK_M_TARGET)
    budget_elems = max(1, _UB_BYTES_BUDGET_PER_BUFFER // elem_bytes)
    block_n = max(1, budget_elems // block_m)
    block_n = min(block_n, n)
    return (block_m, block_n)


def _choose_threads(total: int) -> int:
    """A SimtVF thread count that evenly divides *total*, close to vecadd's 2048."""
    for candidate in _DIVISORS_TO_TRY:
        if candidate <= total and total % candidate == 0:
            return candidate
    return 1


@functools.lru_cache(maxsize=None)
def _build_kernel(m: int, n: int, dtype: torch.dtype):
    """Compile the Ascend GELU-tanh-and-mul kernel for one (m, n, dtype)."""
    dtype_str = Kernel.dtype_to_str(dtype)
    elem_bytes = _ELEM_BYTES[dtype]
    block_m, block_n = _pick_tile(m, n, elem_bytes)
    num_row_tiles = (m + block_m - 1) // block_m
    num_col_tiles = (n + block_n - 1) // block_n
    num_blocks = num_row_tiles * num_col_tiles
    threads = _choose_threads(block_m * block_n)
    sqrt_2_over_pi = float(SQRT_2_OVER_PI)
    coeff = float(GELU_TANH_COEFF)

    @T.prim_func
    def main(X: T.Tensor((m, 2 * n), dtype_str), Y: T.Tensor((m, n), dtype_str)):
        with T.Kernel(num_blocks) as bx:
            row_tile = bx // num_col_tiles
            col_tile = bx % num_col_tiles
            row0 = row_tile * block_m
            col0 = col_tile * block_n
            valid_m = T.min(block_m, m - row0)
            valid_n = T.min(block_n, n - col0)
            gate = T.alloc_shared((block_m, block_n), dtype_str)
            value = T.alloc_shared((block_m, block_n), dtype_str)
            out = T.alloc_shared((block_m, block_n), dtype_str)
            T.copy(X[row0 : row0 + valid_m, col0 : col0 + valid_n], gate[:valid_m, :valid_n])
            T.copy(
                X[row0 : row0 + valid_m, n + col0 : n + col0 + valid_n], value[:valid_m, :valid_n]
            )
            with T.SimtVF(threads=threads):
                half = T.cast(0.5, dtype_str)
                one = T.cast(1.0, dtype_str)
                sqrt_2_over_pi_f32 = T.float32(sqrt_2_over_pi)
                coeff_f32 = T.float32(coeff)
                for r, c in T.Parallel(block_m, block_n):
                    g = gate[r, c]
                    v = value[r, c]
                    g_f32 = T.Cast("float32", g)
                    inner = sqrt_2_over_pi_f32 * (g_f32 + coeff_f32 * g_f32 * g_f32 * g_f32)
                    tanh_val = T.Cast(dtype_str, T.tanh(inner))
                    out[r, c] = half * g * (one + tanh_val) * v
            T.copy(out[:valid_m, :valid_n], Y[row0 : row0 + valid_m, col0 : col0 + valid_n])

    return tilelang.compile(main, target="ascend", out_idx=-1)


if _ASCEND_AVAILABLE:

    class AscendGeluTanhAndMulKernel(Kernel, FusedGatedFwdInterface):
        """Ascend TileLang implementation of ``GeluTanhAndMulFwdOp``."""

        devices: ClassVar[frozenset[str]] = frozenset({"npu"})

        @classmethod
        def entry_for(cls, call: FusedGatedCall) -> Entry:
            return (call, lambda: cls(call.m, call.n, call.dtype))

        def __init__(self, m: int, n: int, dtype: torch.dtype):
            super().__init__()
            self.m, self.n, self.dtype = (m, n, dtype)
            self.kernel = _build_kernel(m, n, dtype)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """``activation(x[:, :n]) * x[:, n:]``; see ``FusedGatedFwdInterface``."""
            return self.kernel(x)

    from tileops.backend import register_kernel_type

    register_kernel_type("GeluTanhAndMulFwdOp", "ascend", AscendGeluTanhAndMulKernel)
