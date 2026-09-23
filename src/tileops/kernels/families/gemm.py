"""GemmFwdOp registration for the Ascend Cube path, plus the soft-FP8 family.

`from __future__ import annotations` is DELIBERATELY ABSENT.  PEP 563 turns every
annotation in the module into a string, and TVMScript hands a ``@T.prim_func``'s
parameter annotation straight to ``script.ir_builder.tir.Arg``, which rejects a
``str``::

    InternalError: Check failed: type_code_ == kTVMObjectHandle (11 vs. 8):
    expected Object but got str

Measured both ways in R351 (docs/reports/R351-data/p1-bisect6.py parses at every
stage; the identical file with the one future import added fails with exactly that
message).  Every shipped module in this tree that holds a prim_func --
``kernels/gemm.py``, ``kernels/gemm_epilogue.py``, ``kernels/elementwise_unary.py`` --
is likewise without it.  The only annotation this module ever had was ``kind: str``,
which needs nothing.
"""

import tilelang
import tilelang.language as T

from .._registry import register
from ..common import launch_block_count
from ..gemm import (
    L0AB_BYTES,
    L0C_BYTES,
    L1_BUDGET_BYTES,
    L1_STAGES,
    L0_STAGES,
    build_bmm_kernel,
    build_gemm_kernel,
    build_gemm_w4a16_kernel,
    build_grouped_gemm_kernel,
)
from ..gemm_epilogue import build_gemm_epilogue_kernel
from ..gemm_splitk import build_gemm_splitk_kernel


@register("GemmFwdOp")
def build_gemm(a, b, *, trans_a=False, trans_b=True):
    if a is None or b is None:
        raise ValueError("GemmFwdOp requires both a and b tensors")
    if a.device != b.device:
        raise ValueError(f"GemmFwdOp inputs must share a device, got {a.device} and {b.device}")
    if a.dtype != b.dtype:
        raise TypeError(f"GemmFwdOp inputs must share dtype, got {a.dtype} and {b.dtype}")
    return build_gemm_kernel(tuple(a.shape), tuple(b.shape), a.dtype, trans_a, trans_b)


@register("BmmFwdOp")
def build_bmm(a, b):
    if a is None or b is None:
        raise ValueError("BmmFwdOp requires both a and b tensors")
    if a.device != b.device or a.dtype != b.dtype:
        raise ValueError("BmmFwdOp inputs must share device and dtype")
    return build_bmm_kernel(tuple(a.shape), tuple(b.shape), a.dtype)


@register("GroupedGemmFwdOp")
def build_grouped_gemm(
    a,
    b,
    batch_sizes,
    batch_offsets,
    batch_padded_offsets,
    *,
    transpose_a=False,
    transpose_b=True,
):
    tensors = (a, b, batch_sizes, batch_offsets, batch_padded_offsets)
    if any(tensor is None for tensor in tensors):
        raise ValueError("GroupedGemmFwdOp requires all five input tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("GroupedGemmFwdOp inputs must share a device")
    if a.dtype != b.dtype:
        raise TypeError("GroupedGemmFwdOp a and b must share dtype")
    if batch_sizes.dtype != batch_offsets.dtype or batch_sizes.dtype != batch_padded_offsets.dtype:
        raise TypeError("GroupedGemmFwdOp metadata tensors must share int32 dtype")
    import torch

    if batch_sizes.dtype != torch.int32:
        raise TypeError("GroupedGemmFwdOp metadata tensors must use int32")
    if batch_sizes.ndim != 1 or batch_offsets.shape != batch_sizes.shape or batch_padded_offsets.shape != batch_sizes.shape:
        raise ValueError("GroupedGemmFwdOp metadata tensors must be matching 1D tensors")
    return build_grouped_gemm_kernel(
        tuple(a.shape),
        tuple(b.shape),
        batch_sizes.shape[0],
        a.dtype,
        transpose_a,
        transpose_b,
    )


@register("GemmW4A16FwdOp")
def build_gemm_w4a16(
    activation,
    packed_weight,
    weight_scale,
    weight_zero,
    *,
    group_size=128,
):
    tensors = (activation, packed_weight, weight_scale, weight_zero)
    if any(tensor is None for tensor in tensors):
        raise ValueError("GemmW4A16FwdOp requires all four input tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("GemmW4A16FwdOp inputs must share a device")
    return build_gemm_w4a16_kernel(
        tuple(activation.shape),
        tuple(packed_weight.shape),
        tuple(weight_scale.shape),
        tuple(weight_zero.shape),
        activation.dtype,
        packed_weight.dtype,
        weight_scale.dtype,
        weight_zero.dtype,
        group_size,
    )


def _epilogue_builder(kind: str):
    """One builder per fused-epilogue op (T264 ops 110-112).

    The three ops share ``kernels/gemm_epilogue.py``; only ``kind`` differs, so
    the validation below is written once.
    """
    def build(a, b, bias, *, trans_a=False, trans_b=True):
        tensors = (a, b, bias)
        if any(tensor is None for tensor in tensors):
            raise ValueError(f"gemm_{kind} requires a, b and bias")
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError(f"gemm_{kind} inputs must share a device")
        if a.dtype != b.dtype or a.dtype != bias.dtype:
            raise TypeError(
                f"gemm_{kind} inputs must share dtype, got {a.dtype}, {b.dtype} "
                f"and {bias.dtype}"
            )
        return build_gemm_epilogue_kernel(
            tuple(a.shape), tuple(b.shape), tuple(bias.shape), a.dtype,
            trans_a, trans_b, kind,
        )
    build.__name__ = f"build_gemm_{kind}"
    return build


build_gemm_bias = register("GemmBiasFwdOp")(_epilogue_builder("bias"))
build_gemm_bias_relu = register("GemmBiasReluFwdOp")(_epilogue_builder("bias_relu"))
build_gemm_bias_gelu = register("GemmBiasGeluFwdOp")(_epilogue_builder("bias_gelu"))


def _splitk_builder(a, b, *, trans_a=False, trans_b=True):
    """Builder for op 113 ``gemm_splitk``.

    Same validation as the dense GEMM builder; the split factor itself is the
    kernel planner's decision (see kernels/gemm_splitk.py), so nothing about it
    is exposed here.
    """
    if a is None or b is None:
        raise ValueError("gemm_splitk requires a and b")
    if a.device != b.device:
        raise ValueError("gemm_splitk inputs must share a device")
    if a.dtype != b.dtype:
        raise TypeError(
            f"gemm_splitk inputs must share dtype, got {a.dtype} and {b.dtype}"
        )
    return build_gemm_splitk_kernel(
        tuple(a.shape), tuple(b.shape), a.dtype, trans_a, trans_b
    )


build_gemm_splitk = register("GemmSplitKFwdOp")(_splitk_builder)


# ======================================================================================
# Soft-FP8 family (T351): GemmFp8FwdOp, BmmFp8KNFwdOp, BmmFp8NKFwdOp.
#
# 910B1 has NO FP8 Cube.  An e4m3fn byte is a container: it is decoded to fp16 on the
# vector unit and the contraction runs on the fp16 Cube.  The decode is BIT-EXACT for
# all 254 finite e4m3fn codes and all 14 subnormals, measured on this device against
# torch's own CPU ATen conversion (docs/reports/R351-data/p0-npu-A4_CAST_RINT.tsv), so
# the only thing separating this from a hypothetical native FP8 Cube is the accumulator
# -- and ours is fp32, which is not worse.
#
# The two NaN codes 0x7F / 0xFF decode to +-480.0 rather than NaN.  They cannot occur in
# a tensor produced by ``x.to(torch.float8_e4m3fn)`` (it saturates at +-448 = 0x7E/0xFE);
# the harness CHECKS that rather than assuming it.
#
# WHY FOUR LAUNCHES AND NOT ONE.  Measured, docs/reports/R351-data/p1-diag-03.log, same
# fp16 inputs, 128x128x128 NT:
#     GM->L1, Cube, L0C->GM                    exact       16384/16384
#     GM->L1, Cube, L0C->UB->GM                HALF right   8192/16384
#     GM->UB->L1, Cube, L0C->UB->GM            garbage          1/16384
# A kernel with vector work launches KERNEL_TYPE_MIX_AIC_1_2 (1 AIC + 2 AIV) and both
# AIV subcores run the whole body, so an un-split L0C->UB readout has each subcore
# publish its own half.  ``kernels/gemm_epilogue.py`` hit the same wall in T264 and
# shipped the same answer: keep the Cube launch pure and give the vector work its own
# launch, sliced by ``vid``.  ``launches_per_call`` is reported, not hidden -- the same
# disclosure ``GroupedGemmFwdOp`` makes about its route-3 launches.
# ======================================================================================

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}
_FLAGS = ["-O3", "-DENABLE_BF16"]

#: Two AIV contexts per launch block, as every vector template in this tree uses.
VEC_PER_BLOCK = 2
LAUNCH_CAP = 48


# ------------------------------------------------------------------------------ T353
#: CONTROL-ARM SWITCH.  ``TILEOPS_FP8_GEMM_ARM=r351`` selects R351's Cube
#: (``T.gemm_v0``, 20 launch blocks) and R351's epilogue (two 4-byte GM scalar reads
#: inside the per-tile loop, one row per tile) unchanged, so T353's final round can
#: measure both arms on the same cards in the same interleaving -- T353 section 4 asks
#: for a switch here precisely because ``git stash`` is out of bounds.  Unset, or any
#: other value, selects the R353 path.  Read at BUILD time, so one process is one arm.
def _fp8_arm():
    import os

    return os.environ.get("TILEOPS_FP8_GEMM_ARM", "r353").strip().lower()


# ------------------------------------------------------------------------------ T354
#: CONTROL-ARM SWITCH for the batched FP8 operators' *b layout*.
#: ``TILEOPS_FP8_BMM_ARM=r353`` restores the pre-T354 contract: the Cube always reads
#: ``b`` as [B,N,K], so ``BmmFp8KNFwdOp`` has to materialise one with a full DtoD
#: transpose first.  Unset, or any other value, selects T354's native-KN Cube, which
#: reads [B,K,N] with ``transpose_B=False`` and needs no copy at all.
#:
#: Read at BUILD time, so one process is one arm -- the same shape T353 used for
#: ``TILEOPS_FP8_GEMM_ARM`` (``git stash`` is out of bounds, and the two arms have to be
#: measured on the same card in the same interleaving).
#:
#: 🚨 ``tileops/ops/gemm/bmm.py`` reads the SAME variable through its own one-line twin
#: ``_bmm_fp8_arm()``: the operator side decides whether to transpose and the kernel side
#: decides which layout it compiles for, and those two decisions must agree.  If you
#: change the name here, change it there.
def _fp8_bmm_arm():
    import os

    return os.environ.get("TILEOPS_FP8_BMM_ARM", "t354").strip().lower()


#: Candidate Cube tile edges, largest first.  An edge must DIVIDE its extent: the
#: pipelined loop indexes ``kt * block_k`` and ``n0 + block_n`` with no masking, so a
#: ragged tail would read past the operand and the fixpipe readout would write past the
#: output.  Whatever does not divide keeps ``T.gemm_v0``, which is what R351 shipped and
#: is always correct -- of the 15 GemmFp8FwdOp workloads only m in {1, 8} declines.
_FP8_BLOCK_M = (128, 64, 32)
_FP8_BLOCK_N = (256, 192, 128, 64)
_FP8_BLOCK_K = (256, 128)
_FP8_KL0 = 64


def _fp8_cube_plan(m, n, k, itemsize=2):
    """Tile plan for the pipelined ``T.mma`` Cube, or ``None`` for the gemm_v0 path.

    Every condition is a divisibility or a capacity fact (the four budgets are
    ``kernels/gemm.py``'s own measured constants), not a heuristic.
    """
    block_k = next((bk for bk in _FP8_BLOCK_K if k % bk == 0 and bk % _FP8_KL0 == 0), None)
    if block_k is None:
        return None
    for block_m in _FP8_BLOCK_M:
        if m % block_m:
            continue
        for block_n in _FP8_BLOCK_N:
            if n % block_n:
                continue
            if L1_STAGES * (block_m + block_n) * block_k * itemsize > L1_BUDGET_BYTES:
                continue
            if L0_STAGES * block_m * _FP8_KL0 * itemsize > L0AB_BYTES:
                continue
            if L0_STAGES * _FP8_KL0 * block_n * itemsize > L0AB_BYTES:
                continue
            if block_m * block_n * 4 > L0C_BYTES:
                continue
            return block_m, block_n, block_k, _FP8_KL0
    return None


@T.macro
def _decode(u8s, hs, i16s, s16s, m80, f16s, count):
    """u8 byte -> fp16 value * 2^-8, in place in ``i16s`` (aliased by ``f16s``).

    MUST be a ``T.macro``: a plain Python helper is *evaluated* by the TVMScript
    parser but its expression statements are never emitted into the enclosing
    frame.  Measured (R351-data/p1-decode-gen.cpp): with a plain helper the
    generated kernel contained Fill / copy_gm_to_ub / Muls / copy_ub_to_gm and
    NOT ONE of the eight decode instructions, and the decode silently returned
    garbage (2/4096 bit-exact) instead of failing to compile.

    ``(u & 0x80) << 8 | (u & 0x7F) << 7`` reinterpreted as fp16 is the e4m3fn
    value scaled by 2^-8: fp16's exponent bias is 15 and e4m3's is 7.  The
    caller supplies the 2^8.  Three framework constraints are baked in here:
    there is no uint8->int16 Cast on dav-2201 (go through half), a float->int
    Cast issued with CAST_NONE silently yields zeros (use CAST_RINT), and
    ``T.tile.bitwise_and`` has no scalar form (``tl.ascend_bitwise_ands`` is
    not a registered intrinsic), hence the filled ``m80`` mask.
    """
    T.tile.cast(hs, u8s, "CAST_NONE", count)
    T.tile.cast(i16s, hs, "CAST_RINT", count)
    T.tile.bitwise_and(s16s, i16s, m80)
    T.tile.sub(i16s, i16s, s16s)              # i16s &= 0x7F  (s16s is 0 or 128)
    T.tile.bitwise_lshift(s16s, s16s, 8)
    T.tile.bitwise_lshift(i16s, i16s, 7)
    T.tile.bitwise_or(i16s, i16s, s16s)
    T.reinterpretcast(f16s, i16s, "half")


def build_decode_flat(numel, tile=8192):
    """fp8 bytes -> fp16, no scaling.  Flat elementwise pass."""
    tile = min(tile, numel)
    while numel % VEC_PER_BLOCK and tile > 1:
        tile //= 2
    block_total = tile * VEC_PER_BLOCK
    block_count = max(1, (numel + block_total - 1) // block_total)
    launch_blocks = min(block_count, LAUNCH_CAP)
    grid_repeats = (block_count + launch_blocks - 1) // launch_blocks

    @tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS, compile_flags=_FLAGS)
    def _factory():
        @T.prim_func
        def main(src: T.Tensor((numel,), "uint8"), dst: T.Tensor((numel,), "float16")):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                u8s = T.alloc_ub((tile,), "uint8")
                hs = T.alloc_ub((tile,), "float16")
                i16s = T.alloc_ub((tile,), "int16")
                s16s = T.alloc_ub((tile,), "int16")
                m80 = T.alloc_ub((tile,), "int16")
                f16s = T.alloc_ub((tile,), "float16")
                with T.Scope("V"):
                    T.tile.fill(m80, 128)
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < block_count:
                            start = T.min(logical_cid * block_total + vid * tile,
                                          numel - tile)
                            T.copy(src[start], u8s)
                            _decode(u8s, hs, i16s, s16s, m80, f16s, tile)
                            T.tile.mul(f16s, f16s, T.cast(256.0, "float16"))
                            T.copy(f16s, dst[start])
        return main

    return _factory()


def build_decode_block128(rows, cols, rt=32):
    """fp8 bytes -> fp16, each 128-wide K block scaled by ``scale[row, kb]``.

    ``D = sum_kb sa[m,kb] sb[n,kb] (A_kb B_kb^T)`` and the two diagonal factors
    act only on rows of A and rows of B, so
    ``diag(sa) A_kb (diag(sb) B_kb)^T`` is the SAME sum -- the scale is folded
    per 128-block, never hoisted out of the K loop.  See R351.md "分歧 1".
    """
    assert cols % 128 == 0
    rt = min(rt, rows)
    nkb = cols // 128
    row_tiles = (rows + rt - 1) // rt
    total = row_tiles * nkb
    block_count = max(1, (total + VEC_PER_BLOCK - 1) // VEC_PER_BLOCK)
    launch_blocks = min(block_count, LAUNCH_CAP)
    grid_repeats = (block_count + launch_blocks - 1) // launch_blocks
    count = rt * 128

    @tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS, compile_flags=_FLAGS)
    def _factory():
        @T.prim_func
        def main(src: T.Tensor((rows, cols), "uint8"),
                 scale: T.Tensor((rows * nkb,), "float32"),
                 dst: T.Tensor((rows, cols), "float16")):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                u8s = T.alloc_ub((rt, 128), "uint8")
                hs = T.alloc_ub((rt, 128), "float16")
                i16s = T.alloc_ub((rt, 128), "int16")
                s16s = T.alloc_ub((rt, 128), "int16")
                m80 = T.alloc_ub((rt, 128), "int16")
                f16s = T.alloc_ub((rt, 128), "float16")
                f32s = T.alloc_ub((rt, 128), "float32")
                sc1 = T.alloc_ub((rt * nkb,), "float32")
                out = T.alloc_ub((rt, 128), "float16")
                with T.Scope("V"):
                    T.tile.fill(m80, 128)
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < block_count:
                            t = T.min(logical_cid * VEC_PER_BLOCK + vid, total - 1)
                            r0 = T.min((t // nkb) * rt, rows - rt)
                            kb = t % nkb
                            T.copy(src[r0:r0 + rt, kb * 128:kb * 128 + 128], u8s)
                            _decode(u8s, hs, i16s, s16s, m80, f16s, count)
                            T.tile.cast(f32s, f16s, "CAST_NONE", count)
                            T.tile.mul(f32s, f32s, 256.0)
                            # The scale arrives FLAT and is read as ONE CONTIGUOUS
                            # SLICE.  Three measured constraints force this shape
                            # (R351-data/p1-diag-04.log, p1-b128-only.log):
                            #  * T.tile.broadcast(dst[R,C], src[R,1], axis=1) is a
                            #    SCALAR broadcast -- it replicated src[0,0] into all
                            #    R rows (128/4096 correct), it is not per-row;
                            #  * T.tile.mul(dst, src, buf[i, j]) forwards only
                            #    indices[0] to tl.ascend_muls (ascend_tile.py:995),
                            #    so a 2-D scalar operand silently reads the wrong
                            #    element -- the scalar buffer has to be 1-D;
                            #  * a STRIDED one-column GM->UB read
                            #    (`scale[r0:r0+rt, kb:kb+1]`) delivers only the first
                            #    row's value when r0/kb are runtime expressions:
                            #    measured exactly 1 correct row per tile at rt=8,
                            #    32 and (trivially) 1.  A contiguous flat slice does
                            #    not have that failure mode.
                            T.copy(scale[r0 * nkb], sc1)
                            for i in T.serial(rt):
                                T.tile.mul(f32s[i, 0:128], f32s[i, 0:128], sc1[i * nkb + kb])
                            T.tile.cast(out, f32s, "CAST_RINT", count)
                            T.copy(out, dst[r0:r0 + rt, kb * 128:kb * 128 + 128])
        return main

    return _factory()


def build_cube_nt(m, n, k):
    """fp16 [M,K] x fp16 [N,K]^T -> fp32 [M,N].  Cube-only launch.

    Dispatch only.  The pipelined ``T.mma`` body is the default; the ``T.gemm_v0``
    body below it is R351's and stays the fallback for shapes whose tiles do not
    divide (m in {1, 8} in this manifest) and for the ``r351`` control arm.
    """
    if _fp8_arm() != "r351" and _fp8_cube_plan(m, n, k) is not None:
        return _build_cube_nt_pipelined(m, n, k)
    return _build_cube_nt_v0(m, n, k)


def _build_cube_nt_pipelined(m, n, k):
    """fp16 [M,K] x fp16 [N,K]^T -> fp32 [M,N] with an async GM->L1 stage in front of
    ``T.mma``.  Cube-only launch.

    🚨 WHY THIS EXISTS, measured on THIS kernel (T353 P0, R353-data/p0-pipes.json,
    ds-v3-prefill-down m4096/n7168/k2048, Level1 AiCMetrics.PipeUtilization):

        this file's gemm_v0 Cube   933.3 us   mac .465  mte1 .296  mte2 .407
                                              fixpipe .039  scalar .061  SUM 1.268
        CANN's MatMulV3, same shape 369.8 us   mac .933  mte1 .803  mte2 .900
                                              fixpipe .299  scalar .396  SUM 3.33

    A ratio sum of 1.27 means about ONE pipe is busy at a time -- the kernel was
    running close to the fully serial sum of its own pipes.  That is exactly the
    signature ``kernels/gemm.py``'s module docstring records for R264, and the cause
    is the same: ``tl::ascend::gemm_v0`` closes with ``SetFlag<MTE1_MTE2>(0);
    WaitFlag<MTE1_MTE2>(0);`` (tilelang-ascend common.h:1312-1315), which blocks MTE2
    until MTE1 drains, so no GM->L1 copy can overlap a gemm_v0 call.  R264 measured
    L1 double buffering AROUND gemm_v0 at +1.0-1.1% and the same double buffering with
    gemm_v0 REPLACED by T.mma at +92% (931 -> 486 us on ds-v3-prefill-gate-up).

    The schedule below is ``kernels/gemm.py::_build_gemm_pipelined`` ported to this
    family's NT-only, fp16-in/fp32-out contract; that file is read-only for T353 (it
    serves GemmFwdOp/BmmFwdOp), so the technique is carried over rather than shared.
    Its two measured codegen constraints are carried over with it:

    * a ``T.set_flag`` written as the FIRST statement of the ``T.Scope("C")`` body is
      dropped by the compiler (R264-data/08-hs-deadlock.txt) -- a dropped init showed
      up as a card stuck at AICore=100%, so the flag init is emitted after the
      prologue copy;
    * ``L1_STAGES = 3`` does not fit L1 and segfaults (R264-data/09-recipe-step1.txt);
      the depth constants are imported from ``kernels/gemm.py`` rather than retyped.

    Two things differ from the dense kernel, both forced by this family's contract:
    ``c`` is fp32 (an UNSCALED fp8 product leaves fp16's range: |a|,|b| <= 448 with K
    up to 16384), and ``out_idx`` is absolute rather than ``[-1]`` (R351 section 3.6:
    ``[-1]`` picks an auto-GM workspace when the lowering appends one).
    """
    plan = _fp8_cube_plan(m, n, k)
    if plan is None:  # pragma: no cover - build_cube_nt checks first
        raise ValueError(f"no pipelined Cube plan for m={m} n={n} k={k}")
    block_m, block_n, block_k, kl0 = plan
    s1, s2 = L1_STAGES, L0_STAGES
    m_tiles = m // block_m
    n_tiles = n // block_n
    k_tiles = k // block_k
    kk_steps = block_k // kl0
    logical_blocks = m_tiles * n_tiles
    # R351 wrote ``min(logical_blocks, 20)`` here and left 4 of the card's 24 Cube
    # cores idle: P0 measured Block Num = 20 for this kernel against 24 for CANN's
    # MatMulV3 on the same shape (R353-data/p0-pipes.json).  ``launch_block_count`` is
    # what the dense GEMM uses and lets the runtime spread every tile.
    launch_blocks = launch_block_count(logical_blocks)
    grid_repeats = (logical_blocks + launch_blocks - 1) // launch_blocks

    @tilelang.jit(
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
            # AUTO_SYNC off: the pass is dependency-driven and not multi-buffer aware
            # (tilelang-ascend/src/transform/ascend_sync_insert.cc), so it would put its
            # own drain around every copy on top of the schedule below -- which is
            # exactly the serialisation this kernel exists to remove.
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
        },
        compile_flags=_FLAGS,
    )
    def _factory():
        @T.prim_func
        def main(a: T.Tensor((m, k), "float16"), b: T.Tensor((n, k), "float16"),
                 c: T.Tensor((m, n), "float32")):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, _vid):
                for grid_repeat in T.serial(grid_repeats):
                    logical_cid = cid + grid_repeat * launch_blocks
                    if logical_cid < logical_blocks:
                        m0 = (logical_cid // n_tiles) * block_m
                        n0 = (logical_cid % n_tiles) * block_n

                        a_l1 = T.alloc_L1((s1, block_m, block_k), "float16")
                        b_l1 = T.alloc_L1((s1, block_n, block_k), "float16")
                        a_l0 = T.alloc_L0A((s2, block_m, kl0), "float16")
                        b_l0 = T.alloc_L0B((s2, kl0, block_n), "float16")
                        c_l0 = T.alloc_L0C((block_m, block_n), "float")

                        with T.Scope("C"):
                            T.copy(a[m0:m0 + block_m, 0:block_k], a_l1[0, :, :])
                            T.copy(b[n0:n0 + block_n, 0:block_k], b_l1[0, :, :])
                            T.set_flag("mte2", "mte1", 0)
                            T.set_flag("mte1", "mte2", 1)
                            T.set_flag("m", "mte1", 0)
                            T.set_flag("m", "mte1", 1)

                            for kt in T.serial(k_tiles):
                                nxt = kt + 1
                                if nxt < k_tiles:
                                    T.wait_flag("mte1", "mte2", nxt % s1)
                                    k1 = nxt * block_k
                                    T.copy(a[m0:m0 + block_m, k1:k1 + block_k],
                                           a_l1[nxt % s1, :, :])
                                    T.copy(b[n0:n0 + block_n, k1:k1 + block_k],
                                           b_l1[nxt % s1, :, :])
                                    T.set_flag("mte2", "mte1", nxt % s1)
                                T.wait_flag("mte2", "mte1", kt % s1)

                                for kk in T.serial(kk_steps):
                                    T.wait_flag("m", "mte1", kk % s2)
                                    T.copy(
                                        a_l1[kt % s1, :, kk * kl0:kk * kl0 + kl0],
                                        a_l0[kk % s2, :, :],
                                    )
                                    T.copy(
                                        b_l1[kt % s1, :, kk * kl0:kk * kl0 + kl0],
                                        b_l0[kk % s2, :, :],
                                        transpose=True,
                                    )
                                    T.set_flag("mte1", "m", kk % s2)
                                    T.wait_flag("mte1", "m", kk % s2)
                                    T.mma(
                                        a_l0[kk % s2, :, :],
                                        b_l0[kk % s2, :, :],
                                        c_l0,
                                        init=T.And(kt == 0, kk == 0),
                                    )
                                    T.set_flag("m", "mte1", kk % s2)

                                T.set_flag("mte1", "mte2", kt % s1)

                            # Drain: counted per event id from the schedule above,
                            # because an armed flag survives into the next launch.
                            T.wait_flag("mte1", "mte2", 0)
                            T.wait_flag("mte1", "mte2", 1)
                            T.wait_flag("m", "mte1", 0)
                            T.wait_flag("m", "mte1", 1)
                            # M -> FIX before the readout.  A PipeBarrier<ALL> is not
                            # enough: without this pair the fixpipe read L0C while the
                            # MMADs were still writing it, which surfaced as "FIXP
                            # instruction error: ECC verification failed when the l0c is
                            # read" (R264-data/logs/s7-hm-extra.log).
                            T.set_flag("m", "fix", 0)
                            T.wait_flag("m", "fix", 0)
                            T.copy(c_l0, c[m0:m0 + block_m, n0:n0 + block_n])
                            T.set_flag("fix", "m", 0)
                            T.wait_flag("fix", "m", 0)
        return main

    return _factory()


def _build_cube_nt_v0(m, n, k):
    """fp16 [M,K] x fp16 [N,K]^T -> fp32 [M,N].  Cube-only launch, R351's body.

    Structure copied from tileops.kernels.gemm._build_gemm (the shipped
    gemm_v0 fallback); only the output dtype differs, because an unscaled fp8
    product can exceed fp16's range (|a|,|b| <= 448 and K up to 16384).
    """
    block_m = 32 if m < 128 else 128
    block_n = 64 if n < 256 else 256
    block_k = 64 if k < 256 else 256
    k_l0_size = 64
    m_tiles = (m + block_m - 1) // block_m
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    logical_blocks = m_tiles * n_tiles
    launch_blocks = min(logical_blocks, 20)
    grid_repeats = (logical_blocks + launch_blocks - 1) // launch_blocks

    @tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS, compile_flags=_FLAGS)
    def _factory():
        @T.prim_func
        def main(a: T.Tensor((m, k), "float16"), b: T.Tensor((n, k), "float16"),
                 c: T.Tensor((m, n), "float32")):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                for grid_repeat in T.serial(grid_repeats):
                    logical_cid = cid + grid_repeat * launch_blocks
                    if logical_cid < logical_blocks:
                        m0 = (logical_cid // n_tiles) * block_m
                        n0 = (logical_cid % n_tiles) * block_n
                        a_l1 = T.alloc_shared((block_m, block_k), "float16")
                        b_l1 = T.alloc_shared((block_n, block_k), "float16")
                        c_l0 = T.alloc_fragment((block_m, block_n), "float")
                        for kt in T.serial(k_tiles):
                            k0 = kt * block_k
                            T.copy(a[m0:m0 + block_m, k0:k0 + block_k], a_l1)
                            T.copy(b[n0:n0 + block_n, k0:k0 + block_k], b_l1)
                            T.gemm_v0(a_l1, b_l1, c_l0, transpose_B=True,
                                      init=(kt == 0), kL0Size=k_l0_size)
                        T.copy(c_l0, c[m0:m0 + block_m, n0:n0 + block_n])
        return main

    return _factory()


def _col_chunk(n, cap=2048):
    best = 1
    for cand in range(16, min(cap, n) + 1, 16):
        if n % cand == 0:
            best = cand
    return best if best > 1 else min(n, cap)


# 🚨 T.tile.cast's ROUND MODE IS NOT OPTIONAL, AND THE CORRECT ONE IS DIRECTION-
# DEPENDENT.  The wrong one does not fail to compile; it writes ZEROS.
# Measured on 2048 values each (docs/reports/R351-data/p2-bf16cast.log, p2-widen.log,
# p0-npu-decode4.json):
#
#     direction                       CAST_NONE          CAST_RINT
#     float32 -> bfloat16  (narrow)   0/2048  ZEROS      2048/2048 exact
#     float32 -> float16   (narrow)   2048/2048 exact    2048/2048 exact
#     float16 -> int16     (narrow)   1/256   ZEROS      256/256   exact
#     bfloat16 -> float32  (widen)    2048/2048 exact    0/2048  ZEROS
#     float16  -> float32  (widen)    2048/2048 exact    0/2048  ZEROS
#     uint8    -> float16  (widen)    256/256 exact      (not needed)
#
# So: NARROWING takes CAST_RINT, WIDENING takes CAST_NONE, and fp16 narrowing happens
# to survive either -- which is exactly why this is written down as a rule instead of
# decided per site.  The default out_dtype of all three FP8 operators is BFLOAT16, the
# one narrowing case fp16 would have hidden.  (`T.copy` between two UB buffers is
# correct in BOTH directions and is the safe spelling if in doubt.)
def build_epilogue(m, n, out_dtype, scaled, has_bias):
    """fp32 product -> out_dtype, optionally * (sa*sb) and + bias[N].  Dispatch only."""
    if _fp8_arm() == "r351":
        return _build_epilogue_r351(m, n, out_dtype, scaled, has_bias)
    return _build_epilogue_r353(m, n, out_dtype, scaled, has_bias)


#: Rows of the fp32 product per UB tile, largest first.  Bigger tiles cost nothing
#: numerically (measured bit-identical, R353-data/p1-epilogue.log) and amortise the
#: per-tile fixed cost; the budget below is well under ``kernels/common.py``'s 180224
#: UB waterline because two buffers of this size are live at once.
_EPI_ROWS = (8, 4, 2, 1)
_EPI_UB_BUDGET = 131072


def _epi_rows(m, cn, out_bytes=2):
    for r in _EPI_ROWS:
        if m % r == 0 and r * cn * (4 + out_bytes) <= _EPI_UB_BUDGET:
            return r
    return 1


# 🚨 WHAT R353 CHANGED HERE AND WHY, measured (T353 P0/P1a; raw stdout in
# R353-data/p0-pipes.log and R353-data/p1-epilogue.log, same 4096x7168 fp32 buffer,
# Level1 AiCMetrics.PipeUtilization, reps=20):
#
#     R351's body (two 4-byte GM->UB scalar reads INSIDE the per-tile loop)  624.7 us
#     the same two reads HOISTED out of the loop                            150.5 us
#     hoisted + 8 rows per tile instead of 1                                  52.2 us
#     no scale at all (the floor for this GM traffic)                       118.3 us
#
# The per-tile scalar reads were 16384 tiles x 2 = 32768 four-byte MTE2 transactions
# and cost 474 us -- more than the 176 MB the pass actually moves.  Hoisting them is
# pure code motion: all three variants are bit-identical to each other AND to the CPU
# float64 product correctly rounded to bf16 (0 / 29360128 mismatches each).  The
# epilogue was 703 of the target case's 1693 us, which is 3.6x what R351 section 6.4
# estimated for this pass (196 us); the estimate priced the GM traffic and not the
# scalar reads.
def _build_epilogue_r353(m, n, out_dtype, scaled, has_bias):
    cn = _col_chunk(n)
    # The bias variant keeps one row per tile: broadcasting bias[N] across several rows
    # would need either a real row broadcast (T.tile.broadcast is a SCALAR broadcast on
    # this backend -- R351 section 3.5) or one copy per row, and the single bias case in
    # the manifest is 128 x 2112, already 4.4x.
    rows = 1 if has_bias else _epi_rows(m, cn)
    per_row = n // cn
    total = (m // rows) * per_row
    block_count = max(1, (total + VEC_PER_BLOCK - 1) // VEC_PER_BLOCK)
    launch_blocks = min(block_count, LAUNCH_CAP)
    grid_repeats = (block_count + launch_blocks - 1) // launch_blocks
    out_idx = 1 + (2 if scaled else 0) + (1 if has_bias else 0)

    @tilelang.jit(out_idx=[out_idx], pass_configs=PASS_CONFIGS, compile_flags=_FLAGS)
    def _scaled_bias():
        @T.prim_func
        def main(p: T.Tensor((m, n), "float32"),
                 sa: T.Tensor((1, 1), "float32"), sb: T.Tensor((1, 1), "float32"),
                 bias: T.Tensor((n,), out_dtype), d: T.Tensor((m, n), out_dtype)):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                p32 = T.alloc_ub((1, cn), "float32")
                bu = T.alloc_ub((1, cn), out_dtype)
                b32 = T.alloc_ub((1, cn), "float32")
                ou = T.alloc_ub((1, cn), out_dtype)
                sau = T.alloc_ub((8,), "float32")
                sbu = T.alloc_ub((8,), "float32")
                with T.Scope("V"):
                    # HOISTED.  Two separate 1-D buffers rather than one reused slot:
                    # T.tile.mul forwards only indices[0] of a scalar operand
                    # (ascend_tile.py:995), so each scale gets its own buffer read at 0.
                    T.copy(sa[0:1, 0:1], sau[0:1])
                    T.copy(sb[0:1, 0:1], sbu[0:1])
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < block_count:
                            t = T.min(logical_cid * VEC_PER_BLOCK + vid, total - 1)
                            r = t // per_row
                            c0 = (t % per_row) * cn
                            T.copy(p[r:r + 1, c0:c0 + cn], p32)
                            T.tile.mul(p32, p32, sau[0])
                            T.tile.mul(p32, p32, sbu[0])
                            T.copy(bias[c0:c0 + cn], bu[0, 0:cn])
                            T.tile.cast(b32, bu, "CAST_NONE", cn)   # WIDENING: CAST_NONE
                            T.tile.add(p32, p32, b32)
                            T.tile.cast(ou, p32, "CAST_RINT", cn)
                            T.copy(ou, d[r:r + 1, c0:c0 + cn])
        return main

    @tilelang.jit(out_idx=[out_idx], pass_configs=PASS_CONFIGS, compile_flags=_FLAGS)
    def _scaled_nobias():
        @T.prim_func
        def main(p: T.Tensor((m, n), "float32"),
                 sa: T.Tensor((1, 1), "float32"), sb: T.Tensor((1, 1), "float32"),
                 d: T.Tensor((m, n), out_dtype)):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                p32 = T.alloc_ub((rows, cn), "float32")
                ou = T.alloc_ub((rows, cn), out_dtype)
                sau = T.alloc_ub((8,), "float32")
                sbu = T.alloc_ub((8,), "float32")
                with T.Scope("V"):
                    T.copy(sa[0:1, 0:1], sau[0:1])
                    T.copy(sb[0:1, 0:1], sbu[0:1])
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < block_count:
                            t = T.min(logical_cid * VEC_PER_BLOCK + vid, total - 1)
                            r = (t // per_row) * rows
                            c0 = (t % per_row) * cn
                            T.copy(p[r:r + rows, c0:c0 + cn], p32)
                            T.tile.mul(p32, p32, sau[0])
                            T.tile.mul(p32, p32, sbu[0])
                            T.tile.cast(ou, p32, "CAST_RINT", rows * cn)
                            T.copy(ou, d[r:r + rows, c0:c0 + cn])
        return main

    @tilelang.jit(out_idx=[out_idx], pass_configs=PASS_CONFIGS, compile_flags=_FLAGS)
    def _plain_bias():
        @T.prim_func
        def main(p: T.Tensor((m, n), "float32"), bias: T.Tensor((n,), out_dtype),
                 d: T.Tensor((m, n), out_dtype)):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                p32 = T.alloc_ub((1, cn), "float32")
                bu = T.alloc_ub((1, cn), out_dtype)
                b32 = T.alloc_ub((1, cn), "float32")
                ou = T.alloc_ub((1, cn), out_dtype)
                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < block_count:
                            t = T.min(logical_cid * VEC_PER_BLOCK + vid, total - 1)
                            r = t // per_row
                            c0 = (t % per_row) * cn
                            T.copy(p[r:r + 1, c0:c0 + cn], p32)
                            T.copy(bias[c0:c0 + cn], bu[0, 0:cn])
                            T.tile.cast(b32, bu, "CAST_NONE", cn)   # WIDENING: CAST_NONE
                            T.tile.add(p32, p32, b32)
                            T.tile.cast(ou, p32, "CAST_RINT", cn)
                            T.copy(ou, d[r:r + 1, c0:c0 + cn])
        return main

    @tilelang.jit(out_idx=[out_idx], pass_configs=PASS_CONFIGS, compile_flags=_FLAGS)
    def _plain_nobias():
        @T.prim_func
        def main(p: T.Tensor((m, n), "float32"), d: T.Tensor((m, n), out_dtype)):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                p32 = T.alloc_ub((rows, cn), "float32")
                ou = T.alloc_ub((rows, cn), out_dtype)
                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < block_count:
                            t = T.min(logical_cid * VEC_PER_BLOCK + vid, total - 1)
                            r = (t // per_row) * rows
                            c0 = (t % per_row) * cn
                            T.copy(p[r:r + rows, c0:c0 + cn], p32)
                            T.tile.cast(ou, p32, "CAST_RINT", rows * cn)
                            T.copy(ou, d[r:r + rows, c0:c0 + cn])
        return main

    if scaled and has_bias:
        return _scaled_bias()
    if scaled:
        return _scaled_nobias()
    if has_bias:
        return _plain_bias()
    return _plain_nobias()


def _build_epilogue_r351(m, n, out_dtype, scaled, has_bias):
    """R351's epilogue, kept verbatim as T353's control arm (see ``_fp8_arm``)."""
    cn = _col_chunk(n)
    per_row = n // cn
    total = m * per_row
    block_count = max(1, (total + VEC_PER_BLOCK - 1) // VEC_PER_BLOCK)
    launch_blocks = min(block_count, LAUNCH_CAP)
    grid_repeats = (block_count + launch_blocks - 1) // launch_blocks
    out_idx = 1 + (2 if scaled else 0) + (1 if has_bias else 0)

    @tilelang.jit(out_idx=[out_idx], pass_configs=PASS_CONFIGS, compile_flags=_FLAGS)
    def _scaled_bias():
        @T.prim_func
        def main(p: T.Tensor((m, n), "float32"),
                 sa: T.Tensor((1, 1), "float32"), sb: T.Tensor((1, 1), "float32"),
                 bias: T.Tensor((n,), out_dtype), d: T.Tensor((m, n), out_dtype)):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                p32 = T.alloc_ub((1, cn), "float32")
                bu = T.alloc_ub((1, cn), out_dtype)
                b32 = T.alloc_ub((1, cn), "float32")
                ou = T.alloc_ub((1, cn), out_dtype)
                sab = T.alloc_ub((8,), "float32")
                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < block_count:
                            t = T.min(logical_cid * VEC_PER_BLOCK + vid, total - 1)
                            r = t // per_row
                            c0 = (t % per_row) * cn
                            T.copy(p[r:r + 1, c0:c0 + cn], p32)
                            T.copy(sa[0:1, 0:1], sab[0:1])
                            T.tile.mul(p32, p32, sab[0])
                            T.copy(sb[0:1, 0:1], sab[0:1])
                            T.tile.mul(p32, p32, sab[0])
                            T.copy(bias[c0:c0 + cn], bu[0, 0:cn])
                            T.tile.cast(b32, bu, "CAST_NONE", cn)   # WIDENING: CAST_NONE
                            T.tile.add(p32, p32, b32)
                            T.tile.cast(ou, p32, "CAST_RINT", cn)
                            T.copy(ou, d[r:r + 1, c0:c0 + cn])
        return main

    @tilelang.jit(out_idx=[out_idx], pass_configs=PASS_CONFIGS, compile_flags=_FLAGS)
    def _scaled_nobias():
        @T.prim_func
        def main(p: T.Tensor((m, n), "float32"),
                 sa: T.Tensor((1, 1), "float32"), sb: T.Tensor((1, 1), "float32"),
                 d: T.Tensor((m, n), out_dtype)):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                p32 = T.alloc_ub((1, cn), "float32")
                ou = T.alloc_ub((1, cn), out_dtype)
                sab = T.alloc_ub((8,), "float32")
                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < block_count:
                            t = T.min(logical_cid * VEC_PER_BLOCK + vid, total - 1)
                            r = t // per_row
                            c0 = (t % per_row) * cn
                            T.copy(p[r:r + 1, c0:c0 + cn], p32)
                            T.copy(sa[0:1, 0:1], sab[0:1])
                            T.tile.mul(p32, p32, sab[0])
                            T.copy(sb[0:1, 0:1], sab[0:1])
                            T.tile.mul(p32, p32, sab[0])
                            T.tile.cast(ou, p32, "CAST_RINT", cn)
                            T.copy(ou, d[r:r + 1, c0:c0 + cn])
        return main

    @tilelang.jit(out_idx=[out_idx], pass_configs=PASS_CONFIGS, compile_flags=_FLAGS)
    def _plain_bias():
        @T.prim_func
        def main(p: T.Tensor((m, n), "float32"), bias: T.Tensor((n,), out_dtype),
                 d: T.Tensor((m, n), out_dtype)):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                p32 = T.alloc_ub((1, cn), "float32")
                bu = T.alloc_ub((1, cn), out_dtype)
                b32 = T.alloc_ub((1, cn), "float32")
                ou = T.alloc_ub((1, cn), out_dtype)
                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < block_count:
                            t = T.min(logical_cid * VEC_PER_BLOCK + vid, total - 1)
                            r = t // per_row
                            c0 = (t % per_row) * cn
                            T.copy(p[r:r + 1, c0:c0 + cn], p32)
                            T.copy(bias[c0:c0 + cn], bu[0, 0:cn])
                            T.tile.cast(b32, bu, "CAST_NONE", cn)   # WIDENING: CAST_NONE
                            T.tile.add(p32, p32, b32)
                            T.tile.cast(ou, p32, "CAST_RINT", cn)
                            T.copy(ou, d[r:r + 1, c0:c0 + cn])
        return main

    @tilelang.jit(out_idx=[out_idx], pass_configs=PASS_CONFIGS, compile_flags=_FLAGS)
    def _plain_nobias():
        @T.prim_func
        def main(p: T.Tensor((m, n), "float32"), d: T.Tensor((m, n), out_dtype)):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                p32 = T.alloc_ub((1, cn), "float32")
                ou = T.alloc_ub((1, cn), out_dtype)
                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < block_count:
                            t = T.min(logical_cid * VEC_PER_BLOCK + vid, total - 1)
                            r = t // per_row
                            c0 = (t % per_row) * cn
                            T.copy(p[r:r + 1, c0:c0 + cn], p32)
                            T.tile.cast(ou, p32, "CAST_RINT", cn)
                            T.copy(ou, d[r:r + 1, c0:c0 + cn])
        return main

    if scaled and has_bias:
        return _scaled_bias()
    if scaled:
        return _scaled_nobias()
    if has_bias:
        return _plain_bias()
    return _plain_nobias()


def build_cube_nt_batched(batch, m, n, k, b_is_kn=False):
    """fp16 [B,M,K] x fp16 b -> fp32 [B,M,N].  Cube-only launch.

    ``b_is_kn`` picks which memory order ``b`` arrives in, and nothing else changes:

    * ``False`` -- ``b`` is [B,N,K] and the contraction runs ``transpose_B=True``.
      This is the only spelling that existed before T354.
    * ``True``  -- ``b`` is [B,K,N] (``torch.bmm``'s own order) and the contraction runs
      ``transpose_B=False``, which is ``T.gemm_v0``'s DEFAULT and needs the L1 tile
      shaped [block_k, block_n] instead of [block_n, block_k].

    🚨 WHY THE SECOND SPELLING EXISTS (T354; the numbers are R353's own, re-read from
    ``R353-data/n5/r353/Bmm*/run*/*.json``, 5-run medians, graph regime):

        BmmFp8KN      BmmFp8NK      difference = the transpose
        575.9 us       137.0 us       438.9 us   b=4  1024x1024x1024   (b is   4 MiB)
       3960.4 us      1435.5 us      2524.9 us   b=8  2048x2048x2048   (b is  32 MiB)
        754.5 us       115.2 us       639.2 us   b=32  128x 128x2048   (b is   8 MiB)
       1512.8 us       224.2 us      1288.5 us   b=64  128x2048x 128   (b is  16 MiB)
      12181.2 us      2388.4 us      9792.9 us   b=128 512x 512x2048   (b is 128 MiB)

    The two operators declare the SAME five workloads and the same arithmetic; the only
    difference is the memory order ``b`` arrives in.  ``BmmFp8KNFwdOp`` used to pay for
    that with ``b.transpose(-2, -1).contiguous()``, a full DtoD copy, and the five rows
    above price it at 19-27 GB/s of read+write -- against the 1068 GB/s effective HBM
    roof this tree's own profile calibrates (``perf/profiles/ascend910b1.yaml``).  So it
    is not "one extra pass over b": it is one extra pass at a FORTIETH of the bandwidth,
    and on moe-prefill-b128 it was 80% of the whole operator.

    The docstring the op used to carry said the transpose was there because "the fp8-TN
    WGMMA kernel wants K innermost".  WGMMA is CUDA's instruction; this Cube is
    ``T.gemm_v0``, whose ``transpose_B`` argument defaults to False and whose own API doc
    (tilelang-ascend/docs/api_docs/T.gemm_v0.md, example 3) shows [K,N] as the ordinary
    case and [N,K] as the transposed one.
    """
    block_m = 32 if m < 128 else 128
    block_n = 64 if n < 256 else 256
    block_k = 64 if k < 256 else 256
    k_l0_size = 64
    m_tiles = (m + block_m - 1) // block_m
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    tiles_per_batch = m_tiles * n_tiles
    logical_blocks = batch * tiles_per_batch
    launch_blocks = min(logical_blocks, 20)
    grid_repeats = (logical_blocks + launch_blocks - 1) // launch_blocks

    @tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS, compile_flags=_FLAGS)
    def _factory_nk():
        @T.prim_func
        def main(a: T.Tensor((batch, m, k), "float16"),
                 b: T.Tensor((batch, n, k), "float16"),
                 c: T.Tensor((batch, m, n), "float32")):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                for grid_repeat in T.serial(grid_repeats):
                    logical_cid = cid + grid_repeat * launch_blocks
                    if logical_cid < logical_blocks:
                        bi = logical_cid // tiles_per_batch
                        tile = logical_cid % tiles_per_batch
                        m0 = (tile // n_tiles) * block_m
                        n0 = (tile % n_tiles) * block_n
                        a_l1 = T.alloc_shared((block_m, block_k), "float16")
                        b_l1 = T.alloc_shared((block_n, block_k), "float16")
                        c_l0 = T.alloc_fragment((block_m, block_n), "float")
                        for kt in T.serial(k_tiles):
                            k0 = kt * block_k
                            T.copy(a[bi, m0:m0 + block_m, k0:k0 + block_k], a_l1)
                            T.copy(b[bi, n0:n0 + block_n, k0:k0 + block_k], b_l1)
                            T.gemm_v0(a_l1, b_l1, c_l0, transpose_B=True,
                                      init=(kt == 0), kL0Size=k_l0_size)
                        T.copy(c_l0, c[bi, m0:m0 + block_m, n0:n0 + block_n])
        return main

    @tilelang.jit(out_idx=[2], pass_configs=PASS_CONFIGS, compile_flags=_FLAGS)
    def _factory_kn():
        """Same schedule, ``b`` read as [B,K,N] -- the B tile is [block_k, block_n]."""
        @T.prim_func
        def main(a: T.Tensor((batch, m, k), "float16"),
                 b: T.Tensor((batch, k, n), "float16"),
                 c: T.Tensor((batch, m, n), "float32")):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                for grid_repeat in T.serial(grid_repeats):
                    logical_cid = cid + grid_repeat * launch_blocks
                    if logical_cid < logical_blocks:
                        bi = logical_cid // tiles_per_batch
                        tile = logical_cid % tiles_per_batch
                        m0 = (tile // n_tiles) * block_m
                        n0 = (tile % n_tiles) * block_n
                        a_l1 = T.alloc_shared((block_m, block_k), "float16")
                        b_l1 = T.alloc_shared((block_k, block_n), "float16")
                        c_l0 = T.alloc_fragment((block_m, block_n), "float")
                        for kt in T.serial(k_tiles):
                            k0 = kt * block_k
                            T.copy(a[bi, m0:m0 + block_m, k0:k0 + block_k], a_l1)
                            T.copy(b[bi, k0:k0 + block_k, n0:n0 + block_n], b_l1)
                            T.gemm_v0(a_l1, b_l1, c_l0, transpose_B=False,
                                      init=(kt == 0), kL0Size=k_l0_size)
                        T.copy(c_l0, c[bi, m0:m0 + block_m, n0:n0 + block_n])
        return main

    return _factory_kn() if b_is_kn else _factory_nk()


def build_epilogue_flat(numel, out_dtype, scaled, tile=8192):
    """fp32 product -> out_dtype, optionally * (sa*sb).  No bias (BMM has none).

    The scalar reads are hoisted out of the tile loop for the same measured reason as
    in :func:`_build_epilogue_r353` (R353-data/p1-epilogue.log); the arithmetic is
    unchanged and the two spellings are bit-identical.
    """
    tile = min(tile, numel)
    block_total = tile * VEC_PER_BLOCK
    block_count = max(1, (numel + block_total - 1) // block_total)
    launch_blocks = min(block_count, LAUNCH_CAP)
    grid_repeats = (block_count + launch_blocks - 1) // launch_blocks

    @tilelang.jit(out_idx=[3], pass_configs=PASS_CONFIGS, compile_flags=_FLAGS)
    def _scaled():
        @T.prim_func
        def main(p: T.Tensor((numel,), "float32"),
                 sa: T.Tensor((1,), "float32"), sb: T.Tensor((1,), "float32"),
                 d: T.Tensor((numel,), out_dtype)):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                p32 = T.alloc_ub((tile,), "float32")
                ou = T.alloc_ub((tile,), out_dtype)
                sau = T.alloc_ub((8,), "float32")
                sbu = T.alloc_ub((8,), "float32")
                with T.Scope("V"):
                    T.copy(sa[0:1], sau[0:1])
                    T.copy(sb[0:1], sbu[0:1])
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < block_count:
                            start = T.min(logical_cid * block_total + vid * tile, numel - tile)
                            T.copy(p[start], p32)
                            T.tile.mul(p32, p32, sau[0])
                            T.tile.mul(p32, p32, sbu[0])
                            T.tile.cast(ou, p32, "CAST_RINT", tile)
                            T.copy(ou, d[start])
        return main

    @tilelang.jit(out_idx=[3], pass_configs=PASS_CONFIGS, compile_flags=_FLAGS)
    def _scaled_r351():
        """R351's body, verbatim, for the ``r351`` control arm."""
        @T.prim_func
        def main(p: T.Tensor((numel,), "float32"),
                 sa: T.Tensor((1,), "float32"), sb: T.Tensor((1,), "float32"),
                 d: T.Tensor((numel,), out_dtype)):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                p32 = T.alloc_ub((tile,), "float32")
                ou = T.alloc_ub((tile,), out_dtype)
                sab = T.alloc_ub((8,), "float32")
                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < block_count:
                            start = T.min(logical_cid * block_total + vid * tile, numel - tile)
                            T.copy(p[start], p32)
                            T.copy(sa[0:1], sab[0:1])
                            T.tile.mul(p32, p32, sab[0])
                            T.copy(sb[0:1], sab[0:1])
                            T.tile.mul(p32, p32, sab[0])
                            T.tile.cast(ou, p32, "CAST_RINT", tile)
                            T.copy(ou, d[start])
        return main

    @tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS, compile_flags=_FLAGS)
    def _plain():
        @T.prim_func
        def main(p: T.Tensor((numel,), "float32"), d: T.Tensor((numel,), out_dtype)):
            with T.Kernel(launch_blocks, is_npu=True) as (cid, vid):
                p32 = T.alloc_ub((tile,), "float32")
                ou = T.alloc_ub((tile,), out_dtype)
                with T.Scope("V"):
                    for grid_repeat in T.serial(grid_repeats):
                        logical_cid = cid + grid_repeat * launch_blocks
                        if logical_cid < block_count:
                            start = T.min(logical_cid * block_total + vid * tile, numel - tile)
                            T.copy(p[start], p32)
                            T.tile.cast(ou, p32, "CAST_RINT", tile)
                            T.copy(ou, d[start])
        return main

    if not scaled:
        return _plain()
    return _scaled_r351() if _fp8_arm() == "r351" else _scaled()


def _out_dtype_name(out_dtype):
    """Manifest ``out_dtype`` (a ``torch.dtype`` or its name) -> TileLang dtype string."""
    name = str(out_dtype).removeprefix("torch.")
    if name not in ("float16", "bfloat16"):
        raise ValueError(
            f"the soft-FP8 GEMM writes float16 or bfloat16, got out_dtype={out_dtype!r}"
        )
    return name


class SoftFp8Gemm:
    """Dense soft-FP8 GEMM: decode A, decode B, Cube, epilogue.

    ``launches_per_call`` is 4 (3 when the scales are folded into the decode and
    there is no bias to add -- never, currently).  It is reported, not hidden:
    the same disclosure ``GroupedGemmFwdOp`` makes about its route-3 launches.
    """

    launches_per_call = 4

    def __init__(self, m, n, k, out_dtype_name, block128, has_bias):
        self.m, self.n, self.k = m, n, k
        self.block128 = block128
        self.has_bias = has_bias
        self.out_dtype_name = out_dtype_name
        if block128:
            self._dec_a = build_decode_block128(m, k)
            self._dec_b = build_decode_block128(n, k)
        else:
            self._dec_a = build_decode_flat(m * k)
            self._dec_b = build_decode_flat(n * k)
        self._cube = build_cube_nt(m, n, k)
        self._epi = build_epilogue(m, n, out_dtype_name, not block128, has_bias)

    def __call__(self, a, b, scale_a, scale_b, bias=None):
        import torch
        ra = a.view(torch.uint8)
        rb = b.view(torch.uint8)
        if self.block128:
            a16 = self._dec_a(ra, scale_a.reshape(-1))
            b16 = self._dec_b(rb, scale_b.reshape(-1))
        else:
            a16 = self._dec_a(ra.reshape(-1)).reshape(self.m, self.k)
            b16 = self._dec_b(rb.reshape(-1)).reshape(self.n, self.k)
        p32 = self._cube(a16, b16)
        args = [p32]
        if not self.block128:
            args += [scale_a.reshape(1, 1), scale_b.reshape(1, 1)]
        if self.has_bias:
            args.append(bias)
        return self._epi(*args)


class SoftFp8Bmm:
    """Batched soft-FP8 GEMM, per-tensor scales only (the manifest's contract).

    ``b_is_kn`` is the layout this instance COMPILES FOR, and it is part of the
    contract: [B,K,N] when True, [B,N,K] when False.  ``__call__`` checks it and
    raises rather than guessing, because for a square case (K == N) the two orders
    have the same shape AND the same strides, so a mismatch would otherwise be a
    silent wrong answer rather than an error (T354).

    The decode launch is layout-agnostic by construction -- ``build_decode_flat``
    walks ``b``'s bytes flat and never looks at a row stride -- so the KN path costs
    exactly what the NK path costs, one byte in and two bytes out.  Only the Cube's
    B tile and ``transpose_B`` differ; see :func:`build_cube_nt_batched`.
    """

    launches_per_call = 4

    def __init__(self, batch, m, n, k, out_dtype_name, b_is_kn=False):
        self.batch, self.m, self.n, self.k = batch, m, n, k
        self.out_dtype_name = out_dtype_name
        self.b_is_kn = bool(b_is_kn)
        self._b_shape = (batch, k, n) if self.b_is_kn else (batch, n, k)
        self._dec_a = build_decode_flat(batch * m * k)
        self._dec_b = build_decode_flat(batch * n * k)
        self._cube = build_cube_nt_batched(batch, m, n, k, b_is_kn=self.b_is_kn)
        self._epi = build_epilogue_flat(batch * m * n, out_dtype_name, True)

    def __call__(self, a, b, scale_a, scale_b):
        import torch
        if tuple(b.shape) != self._b_shape:
            raise ValueError(
                f"this batched soft-FP8 kernel was compiled for b in "
                f"{'[B,K,N]' if self.b_is_kn else '[B,N,K]'} = {self._b_shape}, got "
                f"{tuple(b.shape)}.  The caller and the builder must agree on b's "
                f"memory order: since T354 BmmFp8KNFwdOp hands [B,K,N] straight to the "
                f"Cube instead of transposing it, so any caller that still transposes "
                f"(e.g. a harness adapter that reproduces the op's old body) has to drop "
                f"that transpose, or set TILEOPS_FP8_BMM_ARM=r353 to get the old contract"
            )
        ra = a.reshape(-1).view(torch.uint8)
        rb = b.reshape(-1).view(torch.uint8)
        a16 = self._dec_a(ra).reshape(self.batch, self.m, self.k)
        b16 = self._dec_b(rb).reshape(self._b_shape)
        p32 = self._cube(a16, b16).reshape(-1)
        out = self._epi(p32, scale_a.reshape(1), scale_b.reshape(1))
        return out.reshape(self.batch, self.m, self.n)


@register("GemmFp8FwdOp")
def build_gemm_fp8(a, b, scale_a, scale_b, bias, *, out_dtype="bfloat16"):
    """Dense soft-FP8 NT GEMM.  ``a``: [M,K], ``b``: [N,K], so the product is ``a @ b.T``.

    ``scale_a``/``scale_b`` select the mode, exactly as the manifest's ``shape_rules``
    say: (1,1)/(1,1) is per-tensor, (M,ceil(K/128))/(N,ceil(K/128)) is block128.
    """
    import torch

    if a is None or b is None or scale_a is None or scale_b is None:
        raise ValueError("GemmFp8FwdOp requires a, b, scale_a and scale_b")
    if a.dtype != torch.float8_e4m3fn or b.dtype != a.dtype:
        raise TypeError(
            f"GemmFp8FwdOp is float8_e4m3fn only, got {a.dtype} and {b.dtype}"
        )
    if len(a.shape) != 2 or len(b.shape) != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(
            f"GemmFp8FwdOp expects a[M,K] and b[N,K] with matching K, got "
            f"{tuple(a.shape)} and {tuple(b.shape)}"
        )
    m, k = int(a.shape[0]), int(a.shape[1])
    n = int(b.shape[0])
    if k % 128:
        raise ValueError(
            f"the soft-FP8 GEMM steps K in 128-wide blocks (block128's scale granularity "
            f"and the Cube's K tile agree there), got K={k}"
        )
    block128 = tuple(scale_a.shape) != (1, 1)
    return SoftFp8Gemm(m, n, k, _out_dtype_name(out_dtype), block128, bias is not None)


def _bmm_fp8_builder(b_is_nk):
    def build(a, b, scale_a, scale_b, *, out_dtype="bfloat16"):
        import torch

        if a is None or b is None:
            raise ValueError("the batched soft-FP8 GEMM requires both a and b")
        if a.dtype != torch.float8_e4m3fn or b.dtype != a.dtype:
            raise TypeError(f"batched FP8 GEMM is float8_e4m3fn only, got {a.dtype}/{b.dtype}")
        if len(a.shape) != 3 or len(b.shape) != 3 or a.shape[0] != b.shape[0]:
            raise ValueError(
                f"batched FP8 GEMM expects matching 3-D a/b, got {tuple(a.shape)} and "
                f"{tuple(b.shape)}"
            )
        batch, m, k = (int(x) for x in a.shape)
        # ``b`` is the operand as the OPERATOR declares it: [B,N,K] for BmmFp8NKFwdOp,
        # [B,K,N] for BmmFp8KNFwdOp.  Since T354 that is also the order the kernel reads,
        # so no transpose happens anywhere -- unless the r353 control arm is selected, in
        # which case the KN operator goes back to materialising a [B,N,K] copy first and
        # the Cube is compiled for that.
        n = int(b.shape[2 if not b_is_nk else 1])
        if k % 128:
            raise ValueError(f"the soft-FP8 GEMM steps K in 128-wide blocks, got K={k}")
        b_is_kn = (not b_is_nk) and _fp8_bmm_arm() != "r353"
        return SoftFp8Bmm(batch, m, n, k, _out_dtype_name(out_dtype), b_is_kn=b_is_kn)

    return build


build_bmm_fp8_kn = register("BmmFp8KNFwdOp")(_bmm_fp8_builder(False))
build_bmm_fp8_nk = register("BmmFp8NKFwdOp")(_bmm_fp8_builder(True))
