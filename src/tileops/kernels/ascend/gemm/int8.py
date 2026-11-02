"""Ascend 950 INT8 GEMM: INT8 operands, INT32 accumulation, row and column scales.

Two passes. The cube multiplies the two INT8 operands into an INT32 accumulator, whose
partial sums are exact integers; a vector pass then applies the two scales — one per
tensor or one per row of its operand — and the optional bias, and writes the call's
output dtype. Scaling inside the cube launch would need a mixed Cube/AIV kernel, so the
INT32 round trip through GM buys a build that is known to run.
"""

import functools
from typing import Optional

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_cube_cores

import tileops.backend
from tileops.kernels.gemm.call_spec import GemmINT8W8A8Call, GemmINT8W8A8FwdInterface
from tileops.kernels.kernel_base import Kernel

__all__ = ["AscendGemmINT8Kernel"]

_OPERAND = "int8"
_ACC = "int32"
_NUM_AIV_CORES = 64
_CAST_STAGES = 2


def _tiles(m: int, n: int, k: int) -> tuple[int, int, int, int, int]:
    """The tile ``(tile_m, tile_n, tile_k, stages, blocks)`` one call runs with.

    A 256-deep K stage with two operand versions is what this GEMM wants at every row
    count, so the row tile is what gives: it grows with the call up to the 256 rows the
    L0C budget allows, and shrinks by 64 rows when the L1 operand budget cannot hold the
    deeper stage at 256. Row tiles below that stay narrow so a short call does not
    compute rows it has no tokens for. The column tile prefers 224, which splits 7168
    into exactly 32 tiles — one per Cube core.
    """
    tile_m = 16 if m <= 16 else 64 if m <= 64 else 128 if m <= 256 else 256
    tile_n = 224 if n % 224 == 0 else 256 if n % 256 == 0 else 128
    tile_k = 256 if k % 256 == 0 else 128
    # Two operand versions cost 2 * (rows + columns) * tile_k * stages bytes in L1.
    while tile_m > 16 and 2 * (tile_m + tile_n) * tile_k * 2 >= 512 * 1024:
        tile_m -= 64
    return tile_m, tile_n, tile_k, 2, 32


def _gemm_program(
    m: int, n: int, k: int, tile_m: int, tile_n: int, tile_k: int, stages: int, blocks: int
):
    """Build the INT8 product: one persistent block per Cube core, FP32-sized INT32 output.

    The accumulator is INT32 and every partial sum is an exact integer, so the only
    rounding in the whole op happens once, in the scale pass.
    """
    mad_k = 64  # one MAD step: two L0 versions of a 128-row INT8 operand take 16 KiB
    k_tiles = -(-k // tile_k)
    if k % tile_k or tile_k % mad_k or n % tile_n:
        raise ValueError(f"this schedule requires whole K and N tiles, got n={n}, k={k}")
    if 2 * (tile_m + tile_n) * tile_k * stages > 512 * 1024:
        raise ValueError("L1 operand versions exceed the 512 KiB capacity")
    if 4 * tile_m * tile_n > 256 * 1024:
        raise ValueError("INT32 accumulator exceeds the 256 KiB L0C capacity")

    @T.prim_func
    def main(
        A: T.Tensor((m, k), _OPERAND),
        B: T.Tensor((n, k), _OPERAND),
        D: T.Tensor((m, n), _ACC),
    ):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(blocks) as bx:
            a_l1 = T.alloc_l1((tile_m, tile_k), _OPERAND)
            b_l1 = T.alloc_l1((tile_n, tile_k), _OPERAND)
            a_l0 = T.alloc_l0a((tile_m, mad_k), _OPERAND)
            b_l0 = T.alloc_l0b((tile_n, mad_k), _OPERAND)
            acc = T.alloc_l0c((tile_m, tile_n), _ACC)
            T.annotate_buffer_versions({a_l1: stages, b_l1: stages, a_l0: 2, b_l0: 2})
            sched = T.AscendTileScheduler(
                block_m=tile_m,
                block_n=tile_n,
                num_cores=blocks,
                shape_m=m,
                shape_n=n,
            )
            sched.init(bx)
            while sched.valid():
                m_start, n_start, rows, cols = sched.tile()
                for ki in T.Pipelined(k_tiles, num_stages=stages):
                    start = ki * tile_k
                    T.copy(
                        A[m_start : m_start + rows, start : start + tile_k], a_l1[0:rows, 0:tile_k]
                    )
                    T.copy(
                        B[n_start : n_start + cols, start : start + tile_k], b_l1[0:cols, 0:tile_k]
                    )
                    for m_slice in range(tile_k // mad_k):
                        k_slice = m_slice * mad_k
                        T.copy(a_l1[0:rows, k_slice : k_slice + mad_k], a_l0[0:rows, 0:mad_k])
                        T.copy(b_l1[0:cols, k_slice : k_slice + mad_k], b_l0[0:cols, 0:mad_k])
                        T.gemm(
                            a_l0[0:rows, 0:mad_k],
                            b_l0[0:cols, 0:mad_k],
                            acc[0:rows, 0:cols],
                            transpose_B=True,
                            clear_accum=ki == 0 and m_slice == 0,
                        )
                T.copy(acc[0:rows, 0:cols], D[m_start : m_start + rows, n_start : n_start + cols])
                sched.next_block()

    return main


def _scale_program(
    m: int, n: int, out_dtype: str, has_bias: bool, per_token: bool, per_channel: bool
):
    """Build the scale pass: ``out = cast(x * scale_a * scale_b + bias)``.

    The two multiplies and the bias add happen in FP32 in the reference's order, so the
    one rounding to ``out_dtype`` is the only rounding after the exact INT32 product.
    The INT32-to-FP32 widening runs on the vector unit: a scalar ``Cast`` between those
    two types is a bit reinterpretation in this backend, not a conversion.
    One vector core takes a row at a time in column tiles, so the row buffers stay small
    however wide the output is.
    """
    scale_a_shape = (m, 1) if per_token else (1, 1)
    scale_b_shape = (1, n) if per_channel else (1, 1)
    # Row buffers cost 12 bytes per column (double-buffered INT32 in, out-dtype out) on
    # top of the resident scales, so a wide output is walked in column tiles: the widest
    # divisor of N that keeps the pass inside the vector unit's 192 KiB.
    ub_budget = 176 * 1024
    resident = (4 * n if per_channel else 0) + (4 * n if has_bias else 0)
    tile_n = n if resident + 16 * n <= ub_budget else None
    if tile_n is None:
        cap = (ub_budget - resident) // 12
        divisors = [d for d in range(128, min(n, cap) + 1, 128) if n % d == 0]
        if not divisors:
            raise ValueError(f"this scale pass cannot hold N={n}: its resident rows are too wide")
        tile_n = divisors[-1]
    threads = min(1024, n)

    @T.prim_func
    def main(
        X: T.Tensor((m, n), _ACC),
        ScaleA: T.Tensor(scale_a_shape, "float32"),
        ScaleB: T.Tensor(scale_b_shape, "float32"),
        Bias: T.Tensor((n,), out_dtype) if has_bias else None,
        Y: T.Tensor((m, n), out_dtype),
    ):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(_NUM_AIV_CORES) as core:
            if per_channel:
                channel_ub = T.alloc_shared((n,), "float32")
                T.copy(ScaleB[0, 0:n], channel_ub)
            else:
                sb_ub = T.alloc_shared((1, 1), "float32")
                T.copy(ScaleB[0:1, 0:1], sb_ub)
            if has_bias:
                # The bias is added in FP32, so it is widened once per block.
                bias_ub = T.alloc_shared((n,), "float32")
                with T.SimtVF(threads=threads):
                    for j in T.Parallel(n):
                        bias_ub[j] = T.Cast("float32", Bias[j])
            for row in T.Persistent([m], _NUM_AIV_CORES, core, num_stages=_CAST_STAGES):
                x_ub = T.alloc_shared((tile_n,), _ACC)
                out_ub = T.alloc_shared((tile_n,), out_dtype)
                sa_ub = T.alloc_shared((1, 1), "float32")
                T.annotate_buffer_versions({x_ub: _CAST_STAGES, out_ub: _CAST_STAGES})
                sa_row = row if per_token else 0
                T.copy(ScaleA[sa_row : sa_row + 1, 0:1], sa_ub)
                for col in T.serial(n // tile_n):
                    T.copy(X[row, col * tile_n : (col + 1) * tile_n], x_ub)
                    with T.SimdVF():
                        sa = T.simd.vld(sa_ub[0, 0], "BRC_B32")
                        if not per_channel:
                            sb = T.simd.vld(sb_ub[0, 0], "BRC_B32")
                        for j in T.serial(tile_n // 64):
                            base = j * 64
                            value = T.simd.vcvt(T.simd.vld(x_ub[base]), "float32")
                            value = T.simd.vmul(value, sa)
                            if per_channel:
                                value = T.simd.vmul(
                                    value, T.simd.vld(channel_ub[col * tile_n + base])
                                )
                            else:
                                value = T.simd.vmul(value, sb)
                            if has_bias:
                                value = T.simd.vadd(value, T.simd.vld(bias_ub[col * tile_n + base]))
                            T.simd.vsts(out_ub[base], T.simd.vcvt(value, out_dtype), dist="PK_B32")
                    T.copy(out_ub, Y[row, col * tile_n : (col + 1) * tile_n])

    return main


@functools.lru_cache(maxsize=256)
def _make_gemm(m: int, n: int, k: int):
    """Compile the INT8 product serving ``(m, n, k)``."""
    tile_m, tile_n, tile_k, stages, blocks = _tiles(m, n, k)
    return tilelang.compile(
        _gemm_program(m, n, k, tile_m, tile_n, tile_k, stages, blocks),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )


@functools.lru_cache(maxsize=256)
def _make_scale(m: int, n: int, out_dtype: str, has_bias: bool, per_token: bool, per_channel: bool):
    """Compile the scale pass for one output side."""
    return tilelang.compile(
        _scale_program(m, n, out_dtype, has_bias, per_token, per_channel),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )


class AscendGemmINT8Kernel(Kernel, GemmINT8W8A8FwdInterface):
    """INT8 NT matmul on the Ascend 950 matrix unit, dequantized by row and column scales.

    The product is exact in INT32 and the epilogue runs in FP32 in the reference's order,
    so the result differs from the reference only by the order of the two scale
    multiplies and the bias add, which can move one result across a single rounding
    boundary of the output dtype.

    ``a`` is ``(M, K)`` and ``b`` is ``(N, K)`` in ``int8``; ``scale_a`` is ``(M, 1)`` or
    ``(1, 1)`` and ``scale_b`` is ``(1, N)`` or ``(1, 1)``.
    """

    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: GemmINT8W8A8Call) -> Optional[str]:
        if call.out_dtype not in (torch.bfloat16, torch.float16):
            return "requires a float16 or bfloat16 output"
        if call.m <= 0 or call.n <= 0 or call.k <= 0:
            return f"requires positive extents, got m={call.m}, n={call.n}, k={call.k}"
        return super().refusal(call)

    def __init__(self, call: GemmINT8W8A8Call):
        device_index = call.device.index
        super().__init__(device_index=device_index)
        name = torch.npu.get_device_name(device_index)
        if "Ascend950" not in name:
            raise RuntimeError(f"These INT8 schedules require Ascend 950, got {name}")
        # The schedules launch a fixed grid of 32 blocks. Larger devices can run
        # that grid unchanged; do not enlarge it without retuning the schedule.
        if get_num_cube_cores(device_id=device_index or 0) < 32:
            raise RuntimeError("These INT8 schedules require at least 32 Cube cores")
        out_dtype = "bfloat16" if call.out_dtype == torch.bfloat16 else "float16"
        self._gemm = _make_gemm(call.m, call.n, call.k)
        self._scale = _make_scale(
            call.m, call.n, out_dtype, call.has_bias, call.per_token, call.per_channel
        )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        acc = self._gemm(a, b)
        if bias is None:
            return self._scale(acc, scale_a, scale_b)
        return self._scale(acc, scale_a, scale_b, bias)


tileops.backend.register_kernel_type("GemmINT8W8A8FwdOp", "ascend", AscendGemmINT8Kernel)
