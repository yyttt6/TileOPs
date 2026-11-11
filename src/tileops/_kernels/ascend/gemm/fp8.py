"""Ascend 950 FP8 GEMM with per-tensor or arbitrary FP32 block scales.

Per-tensor scaling uses a native FP8 Cube product followed by a vector epilogue.
Block scaling folds the operands into FP16 before the Cube contraction; the hardware's
E8M0 scale slots cannot represent arbitrary FP32 factors. One-row contractions instead
stream the original operands and scales on the vector unit when their UB footprint fits,
avoiding both folded operands and the extra launches.
"""

import functools
import math
from typing import Optional

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_cube_cores

import tileops.backend._registry
from tileops._kernels.gemm import GEMMFP8Call, GEMMFP8FwdInterface
from tileops._kernels.kernel_base import Kernel

__all__ = ["AscendGEMMFP8Kernel"]

_OPERAND = "float8_e4m3fn"
_ACC = "float32"
_OUT_DTYPE_NAMES = {torch.bfloat16: "bfloat16", torch.float16: "float16"}
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
    tile_n = next(d for d in (224, 256, 128, *range(240, 15, -16)) if n % d == 0)
    tile_k = 256 if k % 256 == 0 else 128
    while tile_m > 16 and 2 * (tile_m + tile_n) * tile_k * 2 >= 512 * 1024:
        tile_m -= 64
    return tile_m, tile_n, tile_k, 2, 32


def _fp8_program(
    m: int, n: int, k: int, tile_m: int, tile_n: int, tile_k: int, stages: int, blocks: int
):
    """Build one ``D = A @ B.T`` in FP32 over ``float8_e4m3fn`` operands.

    One persistent block per Cube core walks the output tiles; each tile stages its
    operands through L1 and hands 64-deep MAD slices to the cube.
    """
    mad_k = 64  # one MAD step: two L0 versions of a 128-row FP8 operand take 16 KiB
    k_tiles = -(-k // tile_k)
    if k % tile_k or tile_k % mad_k or n % tile_n:
        raise ValueError(f"this schedule requires whole K and N tiles, got n={n}, k={k}")
    if 2 * (tile_m + tile_n) * tile_k * stages > 512 * 1024:
        raise ValueError("L1 operand versions exceed the 512 KiB capacity")
    if 4 * tile_m * tile_n > 256 * 1024:
        raise ValueError("FP32 accumulator exceeds the 256 KiB L0C capacity")
    if 4 * max(tile_m, tile_n) * mad_k > 64 * 1024:
        raise ValueError("two L0 versions exceed the 64 KiB per-operand capacity")

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


def _scale_cast_program(m: int, n: int, out_dtype: str, has_bias: bool, has_scale: bool = True):
    """Build the epilogue pass: ``out = cast(x * scale_a * scale_b (+ bias))``.

    The multiplies and the bias add happen in FP32, so the one cast to ``out_dtype``
    is the only rounding after the accumulator. One vector core takes a row at a time
    in column tiles, so the row buffers stay small however wide the output is. The
    scaled path passes both per-tensor scales; the block-scale path has already folded
    its grid into the operands and only adds the bias.
    """
    # Row buffers cost 10 bytes per column (double-buffered FP32 in, out-dtype out).
    ub_budget = 176 * 1024
    tile_n = n if 10 * n <= ub_budget else None
    if tile_n is None:
        cap = ub_budget // 10
        divisors = [d for d in range(128, min(n, cap) + 1, 128) if n % d == 0]
        if not divisors:
            raise ValueError(f"this epilogue cannot hold N={n}: its row is too wide")
        tile_n = divisors[-1]
    threads = min(1024, tile_n)

    @T.prim_func
    def main(
        X: T.Tensor((m, n), _ACC),
        ScaleA: T.Tensor((1, 1), "float32") if has_scale else None,
        ScaleB: T.Tensor((1, 1), "float32") if has_scale else None,
        Bias: T.Tensor((n,), out_dtype) if has_bias else None,
        Y: T.Tensor((m, n), out_dtype),
    ):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(_NUM_AIV_CORES) as core:
            for row in T.Persistent([m], _NUM_AIV_CORES, core, num_stages=_CAST_STAGES):
                x_ub = T.alloc_shared((tile_n,), _ACC)
                out_ub = T.alloc_shared((tile_n,), out_dtype)
                T.annotate_buffer_versions({x_ub: _CAST_STAGES, out_ub: _CAST_STAGES})
                for col in T.serial(n // tile_n):
                    T.copy(X[row, col * tile_n : (col + 1) * tile_n], x_ub)
                    with T.SimtVF(threads=threads):
                        for j in T.Parallel(tile_n):
                            value = x_ub[j]
                            if has_scale:
                                value = value * ScaleA[0, 0] * ScaleB[0, 0]
                            if has_bias:
                                value = value + T.Cast("float32", Bias[col * tile_n + j])
                            out_ub[j] = T.Cast(out_dtype, value)
                    T.copy(out_ub, Y[row, col * tile_n : (col + 1) * tile_n])

    return main


def _fold_program(rows: int, k: int, scale_rows: int, block_rows: int):
    """Build the operand fold: ``y = fp16(x * scale[row // block_rows, k // 128])``.

    The reference multiplies each FP8 operand by its scale grid in FP32 and contracts
    the products. This pass does the same multiply in FP16 — the operand has four
    significand bits, so the folded value loses nothing the tolerance can see — and
    feeds the 16-bit product below, because an arbitrary float32 block scale has no
    place in the Cube's E8M0 scale slots.
    """
    chunk = max(c for c in range(128, min(k, 2048) + 1, 128) if k % c == 0)
    chunks = k // chunk
    groups = chunk // 128
    tile_rows = min(4, rows)
    # Short rows otherwise pay a DMA/VF handoff for very little payload. Batch
    # them only when both operand versions fit and each AIV still gets a pipeline.
    batched_rows = 32
    scale_rows_per_tile = batched_rows if block_rows == 1 else 1
    staged_bytes = 3 * batched_rows * chunk + 4 * scale_rows_per_tile * groups
    if (
        _CAST_STAGES * staged_bytes <= 176 * 1024
        and -(-rows // batched_rows) * chunks >= _NUM_AIV_CORES * _CAST_STAGES
    ):
        tile_rows = batched_rows
    if block_rows > 1:
        tile_rows = math.gcd(tile_rows, block_rows)
    scale_tile_rows = tile_rows if block_rows == 1 else 1

    @T.prim_func
    def main(
        X: T.Tensor((rows, k), _OPERAND),
        S: T.Tensor((scale_rows, k // 128), "float32"),
        Y: T.Tensor((rows, k), "float16"),
    ):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(_NUM_AIV_CORES) as core:
            for item in T.Persistent(
                [T.ceildiv(rows, tile_rows) * chunks],
                _NUM_AIV_CORES,
                core,
                num_stages=_CAST_STAGES,
            ):
                row = (item // chunks) * tile_rows
                col = item % chunks
                valid = T.min(tile_rows, rows - row)
                scale_row = row // block_rows
                valid_scales = valid if block_rows == 1 else 1
                x_ub = T.alloc_shared((tile_rows, chunk), _OPERAND)
                s_ub = T.alloc_shared((scale_tile_rows, groups), "float32")
                out_ub = T.alloc_shared((tile_rows, chunk), "float16")
                T.annotate_buffer_versions(
                    {x_ub: _CAST_STAGES, s_ub: _CAST_STAGES, out_ub: _CAST_STAGES}
                )
                T.copy(X[row : row + valid, col * chunk : (col + 1) * chunk], x_ub[:valid, :])
                T.copy(
                    S[scale_row : scale_row + valid_scales, col * groups : (col + 1) * groups],
                    s_ub[:valid_scales, :],
                )
                # Broadcast each scale once per group. Keep the existing FP16 scale
                # rounding; changing its precision is separate from this schedule.
                with T.SimdVF():
                    full = T.simd.pset(32)
                    for r in T.serial(tile_rows):
                        for g in T.serial(groups):
                            scale = T.simd.vld(s_ub[r // block_rows, g], "BRC_B32")
                            rounded = T.simd.vcvt(scale, "float16", full, sat=False)
                            factor = T.simd.vcvt(rounded, "float32", full, part=0)
                            for part in T.serial(2):
                                offset = g * 128 + part * 64
                                raw = T.simd.vld(x_ub[r, offset], "UNPK4_B8")
                                value = T.simd.vcvt(raw, "float32", full, part=0)
                                product = T.simd.vmul(value, factor, full)
                                narrowed = T.simd.vcvt(product, "float16", full, sat=False)
                                T.simd.vsts(out_ub[r, offset], narrowed, full, "PK_B32")
                T.copy(out_ub[:valid, :], Y[row : row + valid, col * chunk : (col + 1) * chunk])

    return main


def _gemm16_program(
    m: int,
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    stages: int,
    blocks: int,
    out_dtype: str = _ACC,
):
    """Build one ``D = A @ B.T`` over ``float16`` operands.

    ``out_dtype`` is what the fixpipe writes. ``float32`` keeps the accumulator's own
    precision for an epilogue to read back; anything else makes the fixpipe convert on the
    way out, which drops the whole FP32 round trip through GM and the second launch with it.
    The Cube's fixpipe already has that conversion (``F322BF16``/``F322F16``), so this costs
    no extra hardware.
    """
    mad_k = 64  # one MAD step: two L0 versions of a 128-row FP16 operand take 32 KiB
    k_tiles = -(-k // tile_k)
    if k % tile_k or tile_k % mad_k or n % tile_n:
        raise ValueError(f"this schedule requires whole K and N tiles, got n={n}, k={k}")
    if 2 * (tile_m + tile_n) * tile_k * 2 * stages > 512 * 1024:
        raise ValueError("L1 operand versions exceed the 512 KiB capacity")
    if 4 * tile_m * tile_n > 256 * 1024:
        raise ValueError("FP32 accumulator exceeds the 256 KiB L0C capacity")
    if 4 * max(tile_m, tile_n) * mad_k > 64 * 1024:
        raise ValueError("two L0 versions exceed the 64 KiB per-operand capacity")

    @T.prim_func
    def main(
        A: T.Tensor((m, k), "float16"),
        B: T.Tensor((n, k), "float16"),
        D: T.Tensor((m, n), out_dtype),
    ):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(blocks) as bx:
            a_l1 = T.alloc_l1((tile_m, tile_k), "float16")
            b_l1 = T.alloc_l1((tile_n, tile_k), "float16")
            a_l0 = T.alloc_l0a((tile_m, mad_k), "float16")
            b_l0 = T.alloc_l0b((tile_n, mad_k), "float16")
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


def _gemv_tile(n: int, k: int, per_tensor: bool, per_row: bool) -> Optional[int]:
    """Largest row tile fitting both operand versions and the resident activation."""
    columns = 1 if per_tensor else k // 128
    for tile_n in (16, 8, 4, 2, 1):
        scale_rows = tile_n if per_row else 1
        staged = tile_n * k + 4 * scale_rows * columns + 8 * tile_n
        if n % tile_n == 0 and _CAST_STAGES * staged + k + 4 * columns <= 176 * 1024:
            return tile_n
    return None


def _gemv_program(n, k, scale_a_shape, scale_b_shape, out_dtype, has_bias, tile_n):
    """One-row contraction with arbitrary FP32 scales, without folded GM operands."""
    groups = k // 128
    per_tensor = scale_a_shape == (1, 1) and scale_b_shape == (1, 1)
    per_row = scale_b_shape[0] == n
    maximum = _gemv_tile(n, k, per_tensor, per_row)
    if (
        k <= 0
        or k % 128
        or tile_n <= 0
        or n % tile_n
        or 128 % tile_n
        or maximum is None
        or tile_n > maximum
    ):
        raise ValueError("single-row tile violates scale boundaries or UB capacity")
    scale_rows = tile_n if per_row else 1
    scale_columns = 1 if per_tensor else groups

    @T.prim_func
    def main(
        A: T.Tensor((1, k), _OPERAND),
        B: T.Tensor((n, k), _OPERAND),
        SA: T.Tensor(scale_a_shape, "float32"),
        SB: T.Tensor(scale_b_shape, "float32"),
        Bias: T.Tensor((n,), out_dtype) if has_bias else None,
        D: T.Tensor((1, n), out_dtype),
    ):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(_NUM_AIV_CORES) as core:
            a = T.alloc_shared((k,), _OPERAND)
            sa = T.alloc_shared((scale_columns,), "float32")
            T.copy(A[0, :], a)
            T.copy(SA[0, :], sa)
            for task in T.Persistent([n // tile_n], _NUM_AIV_CORES, core, num_stages=_CAST_STAGES):
                first = task * tile_n
                b = T.alloc_shared((tile_n, k), _OPERAND)
                sb = T.alloc_shared((scale_rows, scale_columns), "float32")
                sums = T.alloc_shared((tile_n,), "float32")
                out = T.alloc_shared((tile_n,), out_dtype)
                bias = T.alloc_shared((tile_n,), out_dtype)
                T.annotate_buffer_versions(
                    {
                        b: _CAST_STAGES,
                        sb: _CAST_STAGES,
                        sums: _CAST_STAGES,
                        out: _CAST_STAGES,
                        bias: _CAST_STAGES,
                    }
                )
                T.copy(B[first : first + tile_n, :], b)
                scale_first = 0 if per_tensor else first if per_row else first // 128
                T.copy(SB[scale_first : scale_first + scale_rows, :], sb)
                if has_bias:
                    T.copy(Bias[first : first + tile_n], bias)
                with T.SimdVF():
                    full = T.simd.pset(32)
                    one = T.simd.pset(32, "PAT_VL1")
                    for row in T.serial(tile_n):
                        acc = T.alloc_var("float32x64", init=T.simd.vdup(T.float32(0), "float32"))
                        sr = row if per_row else 0
                        for group in T.serial(groups):
                            scale_col = 0 if per_tensor else group
                            factor_a = T.simd.vld(sa[scale_col], "BRC_B32")
                            factor_b = T.simd.vld(sb[sr, scale_col], "BRC_B32")
                            for part in T.unroll(2):
                                offset = group * 128 + part * 64
                                wide_a = T.simd.vcvt(
                                    T.simd.vld(a[offset], "UNPK4_B8"), "float32", full, part=0
                                )
                                wide_b = T.simd.vcvt(
                                    T.simd.vld(b[row, offset], "UNPK4_B8"), "float32", full, part=0
                                )
                                product = T.simd.vmul(
                                    T.simd.vmul(wide_a, factor_a, full),
                                    T.simd.vmul(wide_b, factor_b, full),
                                    full,
                                )
                                acc = T.simd.vadd(acc, product, full)
                        # Packed narrow stores cannot address arbitrary row offsets.
                        T.simd.vsts(sums[row], T.simd.vcadd(acc, full), one, "ONEPT_B32")
                with T.SimtVF(threads=32):
                    for row in T.Parallel(tile_n):
                        value = sums[row]
                        if has_bias:
                            value = value + T.Cast("float32", bias[row])
                        out[row] = T.Cast(out_dtype, value)
                T.copy(out, D[0, first : first + tile_n])

    return main


@functools.lru_cache(maxsize=256)
def _make_gemv(n, k, scale_a_shape, scale_b_shape, out_dtype, has_bias, tile_n):
    return tilelang.compile(
        _gemv_program(n, k, scale_a_shape, scale_b_shape, out_dtype, has_bias, tile_n),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )


@functools.lru_cache(maxsize=256)
def _make_fold(rows: int, k: int, scale_rows: int, block_rows: int):
    """Compile the operand fold for one operand shape and scale grid."""
    return tilelang.compile(
        _fold_program(rows, k, scale_rows, block_rows),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )


@functools.lru_cache(maxsize=256)
def _make_gemm16(m: int, n: int, k: int, out_dtype: str = _ACC):
    """Compile the FP16 product serving ``(m, n, k)`` for the block-scale path."""
    tile_m, tile_n, tile_k, stages, blocks = _tiles(m, n, k)
    # Sixteen-bit operands double the L1 footprint, so the K stage halves to 128.
    tile_k = 128 if k % 128 == 0 else tile_k
    while tile_m > 16 and 2 * (tile_m + tile_n) * tile_k * 2 * 2 > 512 * 1024:
        tile_m -= 64
    return tilelang.compile(
        _gemm16_program(m, n, k, tile_m, tile_n, tile_k, stages, blocks, out_dtype),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )


@functools.lru_cache(maxsize=256)
def _make_gemm(m: int, n: int, k: int):
    """Compile the FP8 GEMM serving ``(m, n, k)``."""
    tile_m, tile_n, tile_k, stages, blocks = _tiles(m, n, k)
    return tilelang.compile(
        _fp8_program(m, n, k, tile_m, tile_n, tile_k, stages, blocks),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )


@functools.lru_cache(maxsize=256)
def _make_epilogue(m: int, n: int, out_dtype: str, has_bias: bool, has_scale: bool = True):
    """Compile the scale, bias and cast pass for ``(m, n)`` and its output side."""
    return tilelang.compile(
        _scale_cast_program(m, n, out_dtype, has_bias, has_scale),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )


class AscendGEMMFP8Kernel(Kernel, GEMMFP8FwdInterface):
    """FP8 NT matmul on the Ascend 950 matrix unit.

    With per-tensor scales on both operands the Cube runs at its FP8 rate and a vector
    pass applies the two scales and the bias. With a per-128-block scale grid — the
    reference's ``torch._scaled_mm`` contract — the scales are folded into FP16 operands
    by one vector pass each and the product runs at the 16-bit rate: the Cube's
    block-scale slots hold E8M0 factors, so an arbitrary float32 block scale cannot ride
    the FP8 product itself. The folded value keeps every bit the tolerance can see (the
    operand has four significand bits), and the epilogue then adds only the bias.
    A one-row contraction streams the original FP8 operands on the vector unit when
    the resident activation and staged weight rows fit UB, avoiding intermediate operands.
    """

    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: GEMMFP8Call) -> Optional[str]:
        if call.dtype != torch.float8_e4m3fn:
            return "requires float8_e4m3fn operands"
        if call.out_dtype not in _OUT_DTYPE_NAMES:
            return "requires a float16 or bfloat16 output"
        if call.k <= 0 or call.k % 128 or call.n % 16:
            return f"requires positive K divisible by 128 and N divisible by 16, got n={call.n}, k={call.k}"
        return super().refusal(call)

    def __init__(self, call: GEMMFP8Call):
        device_index = call.device.index
        super().__init__(device_index=device_index)
        name = torch.npu.get_device_name(device_index)
        if "Ascend950" not in name:
            raise RuntimeError(f"These FP8 schedules require Ascend 950, got {name}")
        # The schedules launch a fixed grid of 32 blocks. Larger devices can run
        # that grid unchanged; do not enlarge it without retuning the schedule.
        if get_num_cube_cores(device_id=device_index or 0) < 32:
            raise RuntimeError("These FP8 schedules require at least 32 Cube cores")
        out_dtype = _OUT_DTYPE_NAMES[call.out_dtype]
        per_tensor = tuple(call.scale_a_shape) == (1, 1) and tuple(call.scale_b_shape) == (1, 1)
        self._per_tensor = per_tensor
        tile_n = (
            _gemv_tile(call.n, call.k, per_tensor, call.scale_b_shape[0] == call.n)
            if call.m == 1
            else None
        )
        self._single_row = tile_n is not None
        if self._single_row:
            self._gemv = _make_gemv(
                call.n,
                call.k,
                call.scale_a_shape,
                call.scale_b_shape,
                out_dtype,
                call.has_bias,
                tile_n,
            )
            return
        if per_tensor:
            self._gemm = _make_gemm(call.m, call.n, call.k)
            self._epilogue = _make_epilogue(call.m, call.n, out_dtype, call.has_bias)
            return
        self._fold_a = _make_fold(
            call.m,
            call.k,
            call.scale_a_shape[0],
            1 if call.scale_a_shape[0] == call.m else call.m,
        )
        self._fold_b = _make_fold(
            call.n,
            call.k,
            call.scale_b_shape[0],
            1 if call.scale_b_shape[0] == call.n else 128,
        )
        # Without a bias there is nothing left for an epilogue to do once the fixpipe casts
        # the accumulator, so the product writes the call's dtype directly and the FP32
        # accumulator never reaches GM. With a bias the epilogue still runs.
        self._fused_cast = not call.has_bias
        self._gemm = _make_gemm16(call.m, call.n, call.k, out_dtype if self._fused_cast else _ACC)
        self._epilogue = (
            None
            if self._fused_cast
            else _make_epilogue(call.m, call.n, out_dtype, True, has_scale=False)
        )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._single_row:
            if bias is None:
                return self._gemv(a, b, scale_a, scale_b)
            return self._gemv(a, b, scale_a, scale_b, bias)
        if not self._per_tensor:
            a = self._fold_a(a, scale_a)
            b = self._fold_b(b, scale_b)
        out = self._gemm(a, b)
        if self._per_tensor:
            if bias is None:
                return self._epilogue(out, scale_a, scale_b)
            return self._epilogue(out, scale_a, scale_b, bias)
        if self._fused_cast:
            return out
        return self._epilogue(out, bias)


tileops.backend._registry.register_kernel_type("GEMMFP8FwdOp", "ascend", AscendGEMMFP8Kernel)
