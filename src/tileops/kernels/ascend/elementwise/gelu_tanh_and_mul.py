"""Ascend TileLang kernel for ``GeluTanhAndMulFwdOp``.

``y = gelu_tanh(gate) * value`` where ``gate = x[:, :n]``, ``value = x[:, n:]`` and

    gelu_tanh(g) = 0.5 * g * (1 + tanh(sqrt(2/pi) * (g + 0.044715 * g^3)))

The activation is evaluated in float32, then rounded to the storage dtype
before multiplication by value, matching the public CPU reference boundary.

New file only; ``ops/elementwise/activations.py`` and ``kernels/constants.py`` are
imported from, never edited, per ``docs/ascend950/dev_process.md`` §1.
"""

from __future__ import annotations

import functools
from typing import ClassVar

import torch

from tileops.kernels.constants import GELU_TANH_COEFF, SQRT_2_OVER_PI
from tileops.kernels.elementwise.call_spec import FusedGatedCall, FusedGatedFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

try:
    import tilelang
    import tilelang.ascend.language as T

    _ASCEND_AVAILABLE = True
except ImportError:
    _ASCEND_AVAILABLE = False

__all__ = ["AscendGeluTanhAndMulKernel"]

# Bytes one (gate | value | out) UB buffer occupies for one tile: reproduces the
# validated anchor from examples/ascend/example_vecadd.py (tile=8192 fp32 elements,
# 3 buffers, num_stages=2 -> 8192*4*3*2 = 196608 bytes total UB), divided by
# (3 buffers) to get the per-buffer budget this (unpipelined, single-stage) kernel
# uses instead: 8192 * 4 = 32768 bytes per buffer. Not an official UB capacity figure
# (round_1_schedule.md 4.2: "950 每个 AIV 核心的实际 UB 容量本轮未确认"), only a
# starting anchor that compiled and ran correctly elsewhere in this repo.
_UB_BYTES_BUDGET_PER_BUFFER = 32768

# Row count per tile. Small and fixed: on this op's shapes, the row dimension alone
# gives ample grid parallelism (round_1_schedule.md 4.1), and a small block_m keeps
# block_n (the dimension that actually varies across M3/M4 shapes) close to the UB
# budget above for every dtype.
_BLOCK_M_TARGET = 16

_ELEM_BYTES: dict[torch.dtype, int] = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
}

_DIVISORS_TO_TRY = (2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)


def _pick_tile(m: int, n: int, elem_bytes: int) -> tuple[int, int]:
    """(block_m, block_n): general for any (m, n), not tuned only for N=14336.

    block_m is clamped to m; block_n is the UB-budget-derived tile width, clamped to
    n. Neither is required to divide (m, n) evenly -- the kernel below clips the
    actual DMA region with ``T.min`` for a boundary tile, the pattern already
    exercised on this backend in
    testing/ascend/language/test_tilelang_ascend_dynamic_ub.py.
    """
    block_m = min(m, _BLOCK_M_TARGET)
    budget_elems = max(1, _UB_BYTES_BUDGET_PER_BUFFER // elem_bytes)
    block_n = max(1, budget_elems // block_m)
    block_n = min(block_n, n)
    return block_m, block_n


def _choose_threads(total: int) -> int:
    """A SimtVF thread count that evenly divides *total*, close to vecadd's 2048."""
    for candidate in _DIVISORS_TO_TRY:
        if candidate <= total and total % candidate == 0:
            return candidate
    return 1


@functools.lru_cache(maxsize=None)
def _build_fallback_kernel(m: int, n: int, dtype: torch.dtype):
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
                X[row0 : row0 + valid_m, n + col0 : n + col0 + valid_n],
                value[:valid_m, :valid_n],
            )

            with T.SimtVF(threads=threads):
                half = T.cast(0.5, "float32")
                one = T.cast(1.0, "float32")
                sqrt_2_over_pi_f32 = T.float32(sqrt_2_over_pi)
                coeff_f32 = T.float32(coeff)
                for r, c in T.Parallel(block_m, block_n):
                    g = gate[r, c]
                    v = value[r, c]
                    g_f32 = T.Cast("float32", g)
                    inner = sqrt_2_over_pi_f32 * (g_f32 + coeff_f32 * g_f32 * g_f32 * g_f32)
                    tanh_val = T.tanh(inner)
                    activated = half * g_f32 * (one + tanh_val)
                    out[r, c] = T.Cast(dtype_str, activated) * v

            T.copy(out[:valid_m, :valid_n], Y[row0 : row0 + valid_m, col0 : col0 + valid_n])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=None)
def _build_aligned_kernel(m: int, n: int, dtype: torch.dtype):
    """Pipeline contiguous column tiles while distributing work across rows."""
    dtype_str = Kernel.dtype_to_str(dtype)
    target = min(8192, max(2048, m * n // 64))
    vectors = n // 64
    divisor = min(vectors, target // 64)
    while vectors % divisor:
        divisor -= 1
    block_n = divisor * 64
    col_tiles = n // block_n
    total_tiles = m * col_tiles
    grid = min(64, total_tiles)
    stages = 2 if total_tiles >= 2 * grid else 1
    threads = _choose_threads(block_n)
    sqrt_2_over_pi = float(SQRT_2_OVER_PI)
    coeff = float(GELU_TANH_COEFF)

    @T.prim_func
    def main(X: T.Tensor((m, 2 * n), dtype_str), Y: T.Tensor((m, n), dtype_str)):
        with T.Kernel(grid) as bx:
            for tile in T.Persistent([m * col_tiles], grid, bx, num_stages=stages):
                row = tile // col_tiles
                col = (tile % col_tiles) * block_n
                gate = T.alloc_shared((block_n,), dtype_str)
                value = T.alloc_shared((block_n,), dtype_str)
                out = T.alloc_shared((block_n,), dtype_str)
                T.annotate_buffer_versions({gate: stages, value: stages, out: stages})
                T.copy(X[row, col : col + block_n], gate)
                T.copy(X[row, n + col : n + col + block_n], value)
                with T.SimtVF(threads=threads):
                    half = T.cast(0.5, "float32")
                    one = T.cast(1.0, "float32")
                    sqrt_2_over_pi_f32 = T.float32(sqrt_2_over_pi)
                    coeff_f32 = T.float32(coeff)
                    for i in T.Parallel(block_n):
                        g_f32 = T.Cast("float32", gate[i])
                        inner = sqrt_2_over_pi_f32 * (g_f32 + coeff_f32 * g_f32 * g_f32 * g_f32)
                        tanh_val = T.tanh(inner)
                        activated = half * g_f32 * (one + tanh_val)
                        out[i] = T.Cast(dtype_str, activated) * value[i]
                T.copy(out, Y[row, col : col + block_n], l2_cache_ctrl=0)

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=None)
def _build_kernel(m: int, n: int, dtype: torch.dtype):
    if n % 128 == 0:
        return _build_aligned_kernel(m, n, dtype)
    return _build_fallback_kernel(m, n, dtype)


if _ASCEND_AVAILABLE:

    class AscendGeluTanhAndMulKernel(Kernel, FusedGatedFwdInterface):
        """Ascend TileLang implementation of ``GeluTanhAndMulFwdOp``.

        Tiling is a general function of ``(m, n, dtype)`` (``T.ceildiv``-equivalent
        row/col tile counts with a ``T.min``-clipped boundary tile), not special-cased
        to the manifest's ``N=14336`` performance cases: ``tests/ops/test_fused_gated.py``
        exercises this kernel at ``N in {1024, 2048}`` and ``dtype=float32`` as well.
        """

        devices: ClassVar[frozenset[str]] = frozenset({"npu"})
        # supported_archs left at the Kernel default (None): no CUDA arch check applies.

        @classmethod
        def entry_for(cls, call: FusedGatedCall) -> Entry:
            return call, lambda: cls(call.m, call.n, call.dtype)

        def __init__(self, m: int, n: int, dtype: torch.dtype):
            super().__init__()
            if dtype not in _ELEM_BYTES:
                raise ValueError(f"Unsupported GeluTanhAndMul dtype: {dtype}")
            self.m, self.n, self.dtype = m, n, dtype
            self.kernel = _build_kernel(m, n, dtype) if m and n else None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """``activation(x[:, :n]) * x[:, n:]``; see ``FusedGatedFwdInterface``."""
            if (
                x.device.type != "npu"
                or x.dtype != self.dtype
                or tuple(x.shape) != (self.m, 2 * self.n)
            ):
                raise ValueError("GeluTanhAndMul input differs from the compiled NPU signature")
            if not self.m or not self.n:
                return torch.empty((self.m, self.n), dtype=self.dtype, device=x.device)
            return self.kernel(x.contiguous())

    from tileops.backend import register_kernel_type

    register_kernel_type("GeluTanhAndMulFwdOp", "ascend", AscendGeluTanhAndMulKernel)
