"""Ascend 950 INT8 GEMM: INT8 operands, INT32 accumulation, row and column scales.

Two shapes of launch, chosen by what the scales allow.

When both scales are per tensor, the whole call is one cube launch: the fixpipe that
drains the INT32 accumulator to GM dequantizes it on the way out
(``QS322BF16_PRE``/``DEQF16`` with the scale in the dequantization register), so the
INT32 never reaches GM and there is no second pass.

A scale that varies per row or per channel cannot ride that register, so those calls
still run twice: the cube writes the exact INT32 product, and a vector pass applies
the scales to it.
"""

import functools
import os
from pathlib import Path
from typing import Optional

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.carver.arch.driver import get_num_cube_cores

import tileops.backend
from tileops._kernels.gemm.call_spec import GEMMINT8W8A8Call, GEMMINT8W8A8FwdInterface
from tileops._kernels.kernel_base import Compilation, Kernel

__all__ = ["AscendGemmINT8Kernel"]

_OPERAND = "int8"
_ACC = "int32"
_NUM_AIV_CORES = 64
_CAST_STAGES = 2

# TileLang 0.1.15 has no way to ask for the fixpipe dequantization an INT32 accumulator
# needs: `GetCCQuantPre` in src/ascend/op/copy.cc maps a float32 source only, so an
# INT32-to-bfloat16 copy lowers to asc_quant_mode 0 and reinterprets the accumulator's
# bits. This prelude is the workaround, and it took four things to get right.
#
# 1. The mode value. `QuantMode_t` is defined twice, with different values: the compiler's
#    built-in cce_aicore_intrinsics.h has QS322BF16_PRE = 36 and VQS322BF16_PRE = 35,
#    while the pto:: costmodel enum in aarch64-linux/include has them as 35 and 39. 36 is
#    the per-tensor mode and is the one to use; 35 is the per-channel mode, which reads a
#    scale tensor that a scalar call does not provide.
# 2. The scale. It goes through asc_set_l0c_copy_prequant, not through the copy call.
# 3. The address spaces. The destination must be declared `__gm__` and the accumulator
#    `__cc__`; without those the C API overloads do not match.
# 4. The trailing arguments. The 14-argument overload wants
#    (channel_split, nz2nd, nz2dn, clip_relu_pre) = (0, 1, 0, 0), and
#    asc_set_l0c_copy_nz_para(1, 0, 0) must run before the copy.
#
# Verified on one idle 950DT_9582 against a float32 reference: 16384/16384 elements equal,
# max_abs_err 0. The natural form of this (an int32 branch in GetCCQuantPre, and a way to
# pass deqScalar) is proposed upstream.
def _fixpipe_dequant_available() -> bool:
    """Whether this CANN build can dequantize an INT32 L0C accumulator in the fixpipe.

    The recipe needs ``asc_set_l0c_copy_prequant`` to reach the hardware. CANN 9.1 declares
    it for dav-3510 and links, but the ``set_quant_pre`` it reaches is a stub: the kernel
    compiles and the device then faults, so a compile-time check proves nothing. CANN 9.2
    honours it.

    The signal is therefore what the toolkit contains, not what it calls itself: only 9.2
    ships the register-based instruction tree, so its presence next to the headers this
    build resolves means the register is real. A toolkit that was moved, renamed or
    partially assembled is classified by its contents.
    """
    root = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME") or ""
    if not root:
        return False
    # CANN 9.2 carries the register-based instruction tree; 9.1 has only the instruction
    # tree, and its dav-3510 copy of set_quant_pre is a stub. The directory is the signal
    # because it sits beside the headers the build actually uses, so a toolkit that was
    # moved or renamed is detected by what it contains, not by how it is labelled.
    return (Path(root) / "asc" / "impl" / "c_api" / "reg_base_impl").is_dir()


_DEQUANT_PRELUDE = r"""
#include "cce_aicore_intrinsics.h"
__aicore__ inline void tl_l0c2bf16_dequant(__gm__ bfloat16_t* dst, __cc__ int32_t* src,
                                           uint16_t n, uint16_t m, uint16_t dst_row_gap,
                                           __gm__ float* scale_a, __gm__ float* scale_b) {
  // Both scales are read from GM here rather than staged through UB: the cube task this
  // runs in may not touch vector-core memory.
  // The MMA writes L0C on PIPE_M and the fixpipe drains it on PIPE_FIX. TileLang sees a
  // call_extern as scalar work and brackets it with PIPE_M -> PIPE_S sync, which does not
  // order the fixpipe against the MMA; without this the drain reads the previous tile's
  // accumulator. Order the two pipes explicitly.
  asc_set_l0c_copy_nz_para(1, 0, 0);
  // The scale goes in as a packed config, not as the (float, offset, is_signed) overload:
  // that overload exists only from CANN 9.2, and on 9.1 the symbol resolves to nothing for
  // dav-3510, so the kernel links and then faults. This is the same bit layout the 9.2
  // overload assembles internally, so it works on both.
  //   bit 46      is_signed
  //   bits 37..45 offset (9 bits)
  //   bits 13..31 scale's float32 bits, low 13 masked off
  float scale = scale_a[0] * scale_b[0];
  uint32_t scale_bits;
  __builtin_memcpy(&scale_bits, &scale, sizeof(scale_bits));
  uint64_t prequant = (static_cast<uint64_t>(0) << 46) | (static_cast<uint64_t>(0) << 37) |
                      static_cast<uint64_t>(scale_bits & 0xFFFFE000u);
  asc_set_l0c_copy_prequant(prequant);
  asc_copy_l0c2gm(dst, src, n, m, dst_row_gap, m,
                  static_cast<asc_store_l2_cache_mode>(0),
                  static_cast<asc_unit_flag_mode>(0),
                  static_cast<asc_quant_mode>(QS322BF16_PRE),
                  static_cast<asc_relu_pre_mode>(0),
                  0, 1, 0, 0);
}
"""


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


@functools.lru_cache(maxsize=256)
def _fused_program(
    m: int,
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    stages: int,
    blocks: int,
    out_dtype: str,
):
    """Build the INT8 product with both scales applied by the fixpipe that drains it.

    One launch: the cube accumulates the exact INT32 product and the fixpipe dequantizes
    it to ``out_dtype`` on the way to GM, so the INT32 never reaches GM. Only the operator's
    single scalar form goes through here; see the module docstring for why.

    Raises:
        ValueError: A tile does not divide the shape, or exceeds an on-chip budget, or
            ``out_dtype`` is not the bfloat16 the prelude implements.
    """
    mad_k = 64
    k_tiles = -(-k // tile_k)
    if out_dtype != "bfloat16":
        raise ValueError(f"the fused path writes bfloat16, got {out_dtype}")
    if k % tile_k or tile_k % mad_k or n % tile_n or m % tile_m:
        raise ValueError(f"the fused path needs whole M, N and K tiles, got m={m}, n={n}, k={k}")
    if 2 * (tile_m + tile_n) * tile_k * stages > 512 * 1024:
        raise ValueError("L1 operand versions exceed the 512 KiB capacity")
    if 4 * tile_m * tile_n > 256 * 1024:
        raise ValueError("INT32 accumulator exceeds the 256 KiB L0C capacity")

    @T.prim_func
    def main(
        A: T.Tensor((m, k), _OPERAND),
        B: T.Tensor((n, k), _OPERAND),
        ScaleA: T.Tensor((1, 1), "float32"),
        ScaleB: T.Tensor((1, 1), "float32"),
        D: T.Tensor((m, n), out_dtype),
    ):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(blocks, prelude=_DEQUANT_PRELUDE) as bx:
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
                # The fixpipe drains L0C on a different pipe than the MMA fills it on.
                # TileLang brackets a call_extern with PIPE_M -> PIPE_S sync, which does not
                # order the fixpipe, so a late drain would read the previous tile's
                # accumulator. Barrier both sides.
                T.ascend_pipe_barrier("PIPE_ALL")
                T.call_extern(
                    "handle",
                    "tl_l0c2bf16_dequant",
                    T.address_of(D[m_start, n_start]),
                    T.address_of(acc[0, 0]),
                    tile_n,
                    tile_m,
                    n,
                    T.address_of(ScaleA[0, 0]),
                    T.address_of(ScaleB[0, 0]),
                )
                T.ascend_pipe_barrier("PIPE_ALL")
                sched.next_block()

    return main


@functools.lru_cache(maxsize=256)
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


@functools.lru_cache(maxsize=256)
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


def _fusable(m: int, n: int, k: int) -> bool:
    """Whether ``(m, n, k)`` is whole tiles under the schedule ``_tiles`` picks.

    The fused path drains the accumulator with a hand-written C call that takes whole
    tiles, so it cannot serve a call whose M, N or K leaves a remainder.
    """
    tile_m, tile_n, tile_k, _, _ = _tiles(m, n, k)
    return m % tile_m == 0 and n % tile_n == 0 and k % tile_k == 0


@functools.lru_cache(maxsize=256)
def _make_fused(m: int, n: int, k: int, out_dtype: str):
    """Compile the fixpipe-dequantizing INT8 product serving ``(m, n, k)``."""
    tile_m, tile_n, tile_k, stages, blocks = _tiles(m, n, k)
    return tilelang.compile(
        _fused_program(m, n, k, tile_m, tile_n, tile_k, stages, blocks, out_dtype),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )


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


class AscendGemmINT8Kernel(Kernel, GEMMINT8W8A8FwdInterface):
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
    def refusal(cls, call: GEMMINT8W8A8Call) -> Optional[str]:
        if call.out_dtype not in (torch.bfloat16, torch.float16):
            return "requires a float16 or bfloat16 output"
        if call.m <= 0 or call.n <= 0 or call.k <= 0:
            return f"requires positive extents, got m={call.m}, n={call.n}, k={call.k}"
        return super().refusal(call)

    @classmethod
    def compilations(cls, tensors, params):
        """Describe the existing runtime paths without constructing tensors or kernels."""
        (m, k), _ = tensors["a"]
        (n, _), _ = tensors["b"]
        per_token = tuple(tensors["a_scale"][0]) == (m, 1)
        per_channel = tuple(tensors["b_scale"][0]) == (1, n)
        has_bias = tensors.get("bias") is not None
        out_dtype = params.get("out_dtype", "bfloat16")
        if out_dtype not in ("bfloat16", "float16") or min(m, n, k) <= 0:
            return ()
        tile_m, tile_n, tile_k, stages, blocks = _tiles(m, n, k)
        fused = (
            not per_token and not per_channel and not has_bias
            and out_dtype == "bfloat16" and _fusable(m, n, k)
            and _fixpipe_dequant_available()
        )
        if fused:
            programs = (_fused_program(m, n, k, tile_m, tile_n, tile_k, stages, blocks, out_dtype),)
        else:
            programs = (
                _gemm_program(m, n, k, tile_m, tile_n, tile_k, stages, blocks),
                _scale_program(m, n, out_dtype, has_bias, per_token, per_channel),
            )
        return tuple(Compilation(p, target="ascend", out_idx=-1,
                                 pass_configs={"tl.enable_fast_math": False}) for p in programs)

    def __init__(self, call: GEMMINT8W8A8Call):
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
        # One launch when a single scalar per operand carries both scales and no bias
        # rides along: the fixpipe can fold that scalar pair, and nothing else.
        self._fused = (
            not call.per_token and not call.per_channel and not call.has_bias
            and out_dtype == "bfloat16"
            and _fusable(call.m, call.n, call.k)
            and _fixpipe_dequant_available()
        )
        if self._fused:
            self._gemm = _make_fused(call.m, call.n, call.k, out_dtype)
        else:
            self._gemm = _make_gemm(call.m, call.n, call.k)
        self._scale = (
            None
            if self._fused
            else _make_scale(
                call.m, call.n, out_dtype, call.has_bias, call.per_token, call.per_channel
            )
        )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._fused:
            return self._gemm(a, b, scale_a, scale_b)
        acc = self._gemm(a, b)
        if bias is None:
            return self._scale(acc, scale_a, scale_b)
        return self._scale(acc, scale_a, scale_b, bias)


tileops.backend._registry.register_kernel_type("GEMMINT8W8A8FwdOp", "ascend", AscendGemmINT8Kernel)
