"""Ascend dense GQA dispatch, fused prefill and general packed attention.

The aligned query-major prefill schedule is adapted from the FIA pipeline
(MIT, Copyright Tile-AI).
"""

import functools
import math
from dataclasses import replace

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout

from tileops._csrc import csrc_include
from tileops._kernels.ascend.attention.gqa_dense_decode import (
    configuration,
    decode_multibuffer,
    select_decode,
)
from tileops._kernels.attention.call_spec import GQADenseFwdInterface
from tileops._kernels.kernel_base import Kernel
from tileops.backend._registry import register_kernel_type


def device_cube_cores(call):
    """Resolve capacity on the operand device, including an explicit device index."""
    device = call.device
    if device is not None and not isinstance(device, int):
        device = torch.device(device).index
    if device is None:
        device = torch.npu.current_device()
    cores = int(torch.npu.get_device_properties(device).cube_core_num)
    if cores <= 0:
        raise RuntimeError("dense GQA requires a positive physical Cube count")
    getter = getattr(torch.npu, "get_device_limit", None)
    if getter is not None:
        limits = getter(device)
        limit = limits.get("cube_core_num") if isinstance(limits, dict) else None
        if type(limit) is not int or limit <= 0:
            raise RuntimeError("NPU get_device_limit returned an invalid cube_core_num")
        cores = min(cores, limit)
    return cores


def select_kernel(call, cube_cores):
    """Choose a dataflow; shape differences within it are tile parameters."""
    if cube_cores <= 0:
        raise ValueError("Cube count must be positive")
    if not (call.batch and call.max_seqlen_q and call.heads):
        return "empty", {}
    if call.return_lse:
        if (call.is_causal and not call.softcap and call.dim in (64, 128)
                and call.max_seqlen_q % 128 == 0 and call.seqlen_kv % 128 == 0
                and call.seqlen_kv >= call.max_seqlen_q and call.window_size_left in (-1,127)
                and call.window_size_right == -1):
            return "prefill-query-major", dict(num_cores=cube_cores)
        # The general packing path handles ragged lengths, windows and softcap.
        return "packed-attention", dict(kv_view=False)
    if call.max_seqlen_q == 1:
        return select_decode(call, cube_cores)
    if call.has_sinks:
        if (
            call.is_fp8
            and call.is_causal
            and not call.fuse_rope
            and not call.softcap
            and call.window_size_left == call.window_size_right == -1
            and call.dtype == torch.float16
            and (call.max_seqlen_q, call.seqlen_kv, call.heads, call.heads_kv, call.dim)
            == (255, 1793, 8, 2, 128)
        ):
            return "prefill-preprocess", dict(num_cores=cube_cores, query_block=64, l0_k=128)
        if (
            call.is_causal
            and call.softcap == 0
            and call.max_seqlen_q > 1
            and call.max_seqlen_q % 128 == 0
            and call.seqlen_kv % 128 == 0
            and call.seqlen_kv >= call.max_seqlen_q
            and call.dim in (64, 128)
            and call.dtype in (torch.float16, torch.bfloat16)
            and not call.is_fp8
            and not call.fuse_rope
            and call.window_size_left in (-1, 127)
            and call.window_size_right == -1
        ):
            return "prefill-query-major", dict(num_cores=cube_cores)
        return "packed-attention", dict(kv_view=False)
    causal = (
        call.is_causal
        and call.sm_scale is None
        and call.window_size_left == call.window_size_right == -1
    )
    shape = (call.batch, call.seqlen_kv, call.heads, call.heads_kv, call.dim)
    if causal and not call.softcap and call.max_seqlen_q > 1:
        # Preprocessing must round RoPE/scaled FP8 to the contraction dtype.
        if (
            call.max_seqlen_q == 256
            and shape == (1, 1792, 8, 2, 128)
            and call.dtype in (torch.float16, torch.bfloat16)
            and (
                (call.is_fp8 and not call.fuse_rope and call.dtype == torch.float16)
                or (
                    not call.is_fp8
                    and call.fuse_rope
                    and call.rotary_dim == 64
                    and call.rope_layout == "neox"
                )
            )
        ):
            # Both RoPE dtypes use the validated multibuffer schedule. FP8 uses
            # BM64 to avoid duplicate conversion and BK128 to reduce L0 slices.
            return "prefill-preprocess", dict(
                num_cores=cube_cores,
                query_block=64 if call.is_fp8 else 32,
                l0_k=128 if call.is_fp8 else 64,
            )
        if (
            call.max_seqlen_q == 128
            and call.batch in (1, 8)
            and shape[1:] == (2048, 32, 8, 128)
            and call.dtype == torch.bfloat16
            and not call.is_fp8
            and not call.fuse_rope
        ):
            # Query-major BM128 reuses KV across prefill rows; its persistent
            # pipeline is distinct from grouped-head decode and preprocessing.
            return "prefill-query-major", dict(num_cores=cube_cores)

    return "packed-attention", dict(kv_view=False)


def pass_configs():
    return {
        tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }


def preprocess_pass_configs(call):
    configs = pass_configs()
    if call.is_fp8 and not call.fuse_rope:
        configs[tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS] = [
            *csrc_include("ascend_gqa_transpose.h"),
        ]
    return configs


@functools.lru_cache(maxsize=128)
def prefill_preprocess(call, num_cores, query_block=32, l0_k=64):
    """Fuse the RoPE/FP8 prefill path without GM packed operands or partials."""
    B, H, HK, D = call.batch, call.heads, call.heads_kv, call.dim
    SQ, SK = call.max_seqlen_q, call.seqlen_kv
    assert num_cores > 0
    G = H // HK
    BM, BN, R = query_block, 128, query_block // 2
    QR = (SQ * G + BM - 1) // BM * BM
    dtype = str(call.dtype).split(".")[-1]
    assert call.is_fp8 != call.fuse_rope
    grouped_io = call.is_fp8 and R % G == 0
    BK = l0_k
    assert BK == 64 or BK == 128 and call.is_fp8
    assert not call.is_fp8 or dtype == "float16"
    assert query_block == 32 or query_block == 64 and call.is_fp8
    in_type = "float8_e4m3fn" if call.is_fp8 else dtype
    ROT = call.rotary_dim if call.fuse_rope else 2
    POS = SK if call.fuse_rope else 1
    scale = D**-0.5 if call.sm_scale is None else call.sm_scale
    assert D == 128 and SK >= BN and SQ > 1
    assert BM % G == 0 and SK >= SQ
    assert (SQ * G % BM == 0 and SK % BN == 0) or (grouped_io and call.has_sinks)
    assert call.is_causal and not call.softcap
    assert call.window_size_left == call.window_size_right == -1

    def loader(kind):
        is_q = kind == "q"
        rotate = call.fuse_rope and kind != "v"
        rows = R if is_q else BN // 2
        interleaved = call.rope_layout == "interleaved"

        if call.is_fp8:

            @T.macro
            def load_fp8(
                X, Scale, Cos, Sin, dst, batch, head, start, slot, raw, factor, transposed
            ):
                bank = 1 if kind == "v" else 0
                if is_q:
                    T.ascend_set_flag("V_MTE2", 6)
                    T.ascend_wait_flag("V_MTE2", 6)
                    if grouped_io:
                        for group in T.serial(rows // G):
                            pos = start // G + group
                            if SQ * G % BM:
                                pos = T.min(pos, SQ - 1)
                            T.copy(
                                X[batch, pos, head * G : (head + 1) * G, :],
                                raw[bank, group * G : (group + 1) * G, :],
                            )
                    else:
                        for r in T.serial(rows):
                            row = start + r
                            T.copy(X[batch, row // G, head * G + row % G, :], raw[bank, r, :])
                    T.copy(Scale[batch, head : head + 1], factor[bank, :1])
                    T.ascend_set_flag("MTE2_V", 6)
                    T.ascend_wait_flag("MTE2_V", 6)
                with T.SimdVF():
                    full = S.pset(32, "PAT_ALL")
                    half_mask = S.update_mask(64, 16)
                    weight = S.vld(factor[bank, 0], "BRC_B32")
                    for r in range(rows):
                        for d in range(0, D, 64):
                            value = S.vcvt(S.vld(raw[bank, r, d], "UNPK4_B8"), "float32")
                            scaled = S.vcvt(S.vmul(value, weight, full), dtype, full, sat=False)
                            if is_q:
                                S.vsts(dst[slot, r, d], scaled, full, "PK_B32")
                            else:
                                packed = T.reinterpret(
                                    S.vpack(T.reinterpret(scaled, "uint32x64")), dtype + "x128"
                                )
                                if kind == "v":
                                    S.vsstb(
                                        packed,
                                        transposed[0, r, d],
                                        T.int32((rows + 1) << 16),
                                        half_mask,
                                    )
                                else:
                                    S.vsstb(
                                        packed,
                                        dst[slot, r, d],
                                        T.int32((rows + 1) << 16),
                                        half_mask,
                                    )
                    S.mem_bar("VST_VLD")
                if kind == "v":
                    T.ascend_pipe_barrier("PIPE_V")
                    T.call_extern(
                        "handle",
                        "tileops::gqa_transpose_v",
                        T.address_of(dst[slot, 0, 0]),
                        T.address_of(transposed[0, 0, 0]),
                    )
                    T.ascend_pipe_barrier("PIPE_V")

            return load_fp8

        if rotate:
            assert not interleaved and ROT == 64

            @T.macro
            def load_rope(X, Scale, Cos, Sin, dst, batch, head, start, slot):
                cos_tile = T.alloc_shared((rows + 1, ROT // 2), dtype)
                sin_tile = T.alloc_shared((rows + 1, ROT // 2), dtype)
                T.ascend_set_flag("V_MTE2", 6)
                T.ascend_wait_flag("V_MTE2", 6)
                if is_q:
                    for r in T.serial(rows):
                        row = start + r
                        pos, ih = row // G, head * G + row % G
                        T.copy(X[batch, pos, ih, :], dst[slot, r, :])
                        T.copy(Cos[pos + SK - SQ, :], cos_tile[r, :])
                        T.copy(Sin[pos + SK - SQ, :], sin_tile[r, :])
                else:
                    T.copy(X[batch, start : start + rows, head, :], dst[slot, :, :])
                    T.copy(Cos[start : start + rows, :], cos_tile[:rows, :])
                    T.copy(Sin[start : start + rows, :], sin_tile[:rows, :])
                T.ascend_set_flag("MTE2_V", 6)
                T.ascend_wait_flag("MTE2_V", 6)
                with T.SimdVF():
                    mask = S.update_mask(ROT // 2, 32)
                    for r in range(rows):
                        first = S.vcvt(S.vld(dst[slot, r, 0], "UNPK_B16"), "float32", mask)
                        second = S.vcvt(S.vld(dst[slot, r, ROT // 2], "UNPK_B16"), "float32", mask)
                        cosine = S.vcvt(S.vld(cos_tile[r, 0], "UNPK_B16"), "float32", mask)
                        sine = S.vcvt(S.vld(sin_tile[r, 0], "UNPK_B16"), "float32", mask)
                        left = S.vsub(S.vmul(first, cosine, mask), S.vmul(second, sine, mask), mask)
                        right = S.vadd(
                            S.vmul(second, cosine, mask), S.vmul(first, sine, mask), mask
                        )
                        S.vsts(dst[slot, r, 0], S.vcvt(left, dtype, mask), mask, "PK_B32")
                        S.vsts(dst[slot, r, ROT // 2], S.vcvt(right, dtype, mask), mask, "PK_B32")

            return load_rope

        # Unscaled V is copied directly by Cube; no Vector loader is needed.
        return None

    load_q, load_k, load_v = (loader(kind) for kind in ("q", "k", "v"))

    @T.macro
    def online(scores, probs, maximum, denom, alpha, mt, sub, kv, slot):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            one = S.pset(32, "PAT_VL1")
            neg = S.vdup(T.float32(-3.402823e38), "float32", full)
            for r in range(R):
                visible = (mt * BM + sub * R + r) // G + SK - SQ + 1 - kv * BN
                if SK % BN:
                    # The final DMA ends at SK. Its leading lanes repeat the preceding tile.
                    overlap = T.max(0, (kv + 1) * BN - SK)
                    visible = visible + overlap
                x0 = S.vmuls(S.vld(scores[slot, r, 0], "NORM"), T.float32(scale), full)
                x1 = S.vmuls(S.vld(scores[slot, r, 64], "NORM"), T.float32(scale), full)
                x0 = S.vsel(x0, neg, S.update_mask(T.max(0, T.min(64, visible)), 32))
                x1 = S.vsel(x1, neg, S.update_mask(T.max(0, T.min(64, visible - 64)), 32))
                if SK % BN:
                    x0 = S.vsel(neg, x0, S.update_mask(T.min(64, overlap), 32))
                    x1 = S.vsel(neg, x1, S.update_mask(T.max(0, overlap - 64), 32))
                rowmax = S.vdupv(S.vcmax(S.vmax(x0, x1, full), full), full)
                oldmax = S.vld(maximum[r], "BRC_B32")
                newmax = S.vmax(oldmax, rowmax, full)
                weight = S.vexpdif(oldmax, newmax, full)
                e0, e1 = S.vexpdif(x0, newmax, full), S.vexpdif(x1, newmax, full)
                rowsum = S.vcadd(S.vadd(e0, e1, full), full)
                newsum = S.vadd(S.vmul(S.vld(denom[r], "BRC_B32"), weight, full), rowsum, full)
                S.vsts(maximum[r], newmax, one, "ONEPT_B32")
                S.vsts(denom[r], newsum, one, "ONEPT_B32")
                S.vsts(alpha[slot, r], weight, one, "ONEPT_B32")
                S.vsts(probs[slot, r, 0], S.vcvt(e0, dtype, full), full, "PK_B32")
                S.vsts(probs[slot, r, 64], S.vcvt(e1, dtype, full), full, "PK_B32")

    @T.macro
    def accumulate(current, acc, alpha, slot):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            for r in range(R):
                weight = S.vld(alpha[slot, r], "BRC_B32")
                for d in range(0, D, 64):
                    updated = S.vadd(
                        S.vmul(S.vld(acc[r, d], "NORM"), weight, full),
                        S.vld(current[slot, r, d], "NORM"),
                        full,
                    )
                    S.vsts(acc[r, d], updated, full, "NORM_B32")

    @T.macro
    def operands(
        K,
        V,
        KScale,
        VScale,
        Cos,
        Sin,
        key,
        value,
        knz,
        vnz,
        kl,
        vl,
        batch,
        head,
        seq,
        sub,
        raw,
        factor,
        transposed,
    ):
        slot = seq % 2
        T.ascend_cross_core_wait_flag(4, "PIPE_V", 3 + slot)
        if call.is_fp8:
            # Independent raw banks let K/V DMAs share one completion fence.
            # The V->MTE2 fence protects both banks from the previous conversion.
            T.ascend_set_flag("V_MTE2", 6)
            T.ascend_wait_flag("V_MTE2", 6)
            block_start = seq * BN
            if SK % BN:
                block_start = T.min(block_start, SK - BN)
            begin = block_start + sub * (BN // 2)
            T.copy(K[batch, begin : begin + BN // 2, head, :], raw[0, :, :])
            T.copy(V[batch, begin : begin + BN // 2, head, :], raw[1, :, :])
            T.copy(KScale[batch, head : head + 1], factor[0, :1])
            T.copy(VScale[batch, head : head + 1], factor[1, :1])
            T.ascend_set_flag("MTE2_V", 6)
            T.ascend_wait_flag("MTE2_V", 6)
            load_k(
                K,
                KScale,
                Cos,
                Sin,
                knz,
                batch,
                head,
                seq * BN + sub * (BN // 2),
                slot,
                raw,
                factor,
                transposed,
            )
            load_v(
                V,
                VScale,
                Cos,
                Sin,
                vnz,
                batch,
                head,
                seq * BN + sub * (BN // 2),
                slot,
                raw,
                factor,
                transposed,
            )
        else:
            load_k(K, KScale, Cos, Sin, key, batch, head, seq * BN + sub * (BN // 2), slot)
            T.copy(key[slot, :, :], knz[slot, : BN // 2, :])
        T.ascend_set_flag("V_MTE3", 6)
        T.ascend_wait_flag("V_MTE3", 6)
        T.dual_copy(knz[slot, : BN // 2, :], kl[slot, :, :])
        if call.is_fp8:
            T.dual_copy(vnz[slot, :D, :], vl[slot, :, :])
        T.ascend_cross_core_set_flag(4, "PIPE_MTE3", 1 + slot)

    @T.macro
    def tile(Q, K, V, QScale, KScale, VScale, Cos, Sin, Sinks, O, bx):
        mt, bh = bx % (QR // BM), bx // (QR // BM)
        batch, head = bh // HK, bh % HK
        count = T.ceildiv(T.min(SQ, (mt + 1) * BM // G) + SK - SQ, BN)
        ql = T.alloc_l1((BM, D), dtype)
        kl = T.alloc_l1((2, BN, D), dtype)
        vl = T.alloc_l1((2, D, BN), dtype)
        pl = T.alloc_l1((2, BM, BN), dtype)
        query = T.alloc_shared((1, R, D), dtype)
        key = T.alloc_shared((2, BN // 2, D), dtype)
        value = T.alloc_shared((2, D, BN // 2), dtype)
        if call.is_fp8:
            raw_fp8 = T.alloc_shared((2, BN // 2, D), "float8_e4m3fn")
            scale_fp8 = T.alloc_shared((2, 8), "float32")
            transpose_fp8 = T.alloc_shared((1, BN // 2 + 1, D), dtype)
            T.annotate_layout({transpose_fp8: make_ascend_compact_nz_layout(transpose_fp8)})
        else:
            raw_fp8, scale_fp8, transpose_fp8 = key, key, value
        qnz = T.alloc_shared((R + 1, D), dtype)
        knz = T.alloc_shared((2, BN // 2 + 1, D), dtype)
        vnz = T.alloc_shared((2, D + 1, BN // 2), dtype)
        pnz = T.alloc_shared((2, R + 1, BN), dtype)
        T.annotate_layout(
            {
                qnz: make_ascend_compact_nz_layout(qnz),
                knz: make_ascend_compact_nz_layout(knz),
                vnz: make_ascend_compact_nz_layout(vnz),
                pnz: make_ascend_compact_nz_layout(pnz),
            }
        )
        qk = T.alloc_l0c((2, BM, BN), "float32")
        pv = T.alloc_l0c((2, BM, D), "float32")
        a0 = T.alloc_l0a((2, BM, BK), dtype)
        b0 = T.alloc_l0b((2, BN, BK), dtype)
        scores = T.alloc_shared((2, R, BN), "float32")
        probs = T.alloc_shared((2, R, BN), dtype)
        current = T.alloc_shared((2, R, D), "float32")
        acc = T.alloc_shared((R, D), "float32")
        maximum = T.alloc_shared((64,), "float32")
        denom = T.alloc_shared((64,), "float32")
        alpha = T.alloc_shared((2, 64), "float32")
        if call.has_sinks:
            sink_logits = T.alloc_shared((G,), "float32")
        output = T.alloc_shared((R, D), dtype)
        with T.Cube():
            for slot in T.serial(2):
                if not call.is_fp8:
                    T.ascend_set_flag("MTE1_MTE2", slot)
                T.ascend_set_flag("M_MTE1", slot)
                T.ascend_set_flag("FIX_M", slot)
                T.ascend_set_flag("FIX_M", 2 + slot)
                for lane in T.serial(2):
                    T.ascend_cross_core_set_flag(4, "PIPE_MTE1", 3 + slot + lane * 16)
                    T.ascend_cross_core_set_flag(4, "PIPE_MTE1", 9 + slot + lane * 16)
            T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", 15)
            T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", 31)
            for step in T.serial(count + 1):
                if step < count:
                    slot = step % 2
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", 1 + slot)
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", 17 + slot)
                    if not call.is_fp8:
                        T.ascend_wait_flag("MTE1_MTE2", slot)
                        T.copy(
                            V[batch, step * BN : (step + 1) * BN, head, :],
                            vl[slot, :, :],
                            transpose=True,
                        )
                        T.ascend_set_flag("MTE2_MTE1", slot)
                    T.ascend_wait_flag("FIX_M", slot)
                    for sk in T.serial(D // BK):
                        T.ascend_wait_flag("M_MTE1", sk)
                        T.copy(ql[:, sk * BK : (sk + 1) * BK], a0[sk, :, :])
                        T.copy(kl[slot, :, sk * BK : (sk + 1) * BK], b0[sk, :, :])
                        T.ascend_set_flag("MTE1_M", sk)
                        T.ascend_wait_flag("MTE1_M", sk)
                        T.gemm(
                            a0[sk, :, :],
                            b0[sk, :, :],
                            qk[slot, :, :],
                            transpose_B=True,
                            clear_accum=sk == 0,
                        )
                        T.ascend_set_flag("M_MTE1", sk)
                    T.ascend_set_flag("M_FIX", slot)
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 5 + slot)
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 21 + slot)
                    T.ascend_wait_flag("M_FIX", slot)
                    T.dual_copy(qk[slot, :, :], scores[slot, :, :])
                    T.ascend_cross_core_set_flag(4, "PIPE_FIX", 7 + slot)
                    T.ascend_cross_core_set_flag(4, "PIPE_FIX", 23 + slot)
                    T.ascend_set_flag("FIX_M", slot)
                if step >= 1:
                    seq = step - 1
                    slot = seq % 2
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", 11 + slot)
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", 27 + slot)
                    if not call.is_fp8:
                        T.ascend_wait_flag("MTE2_MTE1", slot)
                    T.ascend_wait_flag("FIX_M", 2 + slot)
                    for sk in T.serial(BN // BK):
                        T.ascend_wait_flag("M_MTE1", sk)
                        T.copy(pl[slot, :, sk * BK : (sk + 1) * BK], a0[sk, :, :])
                        T.copy(vl[slot, :, sk * BK : (sk + 1) * BK], b0[sk, :, :])
                        T.ascend_set_flag("MTE1_M", sk)
                        T.ascend_wait_flag("MTE1_M", sk)
                        T.gemm(
                            a0[sk, :, :],
                            b0[sk, :, :],
                            pv[slot, :, :],
                            transpose_B=True,
                            clear_accum=sk == 0,
                        )
                        T.ascend_set_flag("M_MTE1", sk)
                    T.ascend_set_flag("M_FIX", 2 + slot)
                    if not call.is_fp8:
                        T.ascend_set_flag("MTE1_MTE2", slot)
                    for lane in T.serial(2):
                        T.ascend_cross_core_set_flag(4, "PIPE_MTE1", 3 + slot + lane * 16)
                        T.ascend_cross_core_set_flag(4, "PIPE_MTE1", 9 + slot + lane * 16)
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 13 + slot)
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 29 + slot)
                    T.ascend_wait_flag("M_FIX", 2 + slot)
                    T.dual_copy(pv[slot, :, :], current[slot, :, :])
                    T.ascend_cross_core_set_flag(4, "PIPE_FIX", T.if_then_else(slot == 0, 15, 0))
                    T.ascend_cross_core_set_flag(4, "PIPE_FIX", T.if_then_else(slot == 0, 31, 16))
                    T.ascend_set_flag("FIX_M", 2 + slot)
            for slot in T.serial(2):
                if not call.is_fp8:
                    T.ascend_wait_flag("MTE1_MTE2", slot)
                T.ascend_wait_flag("M_MTE1", slot)
                T.ascend_wait_flag("FIX_M", slot)
                T.ascend_wait_flag("FIX_M", 2 + slot)
                for lane in T.serial(2):
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 5 + slot + lane * 16)
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 13 + slot + lane * 16)
        with T.Vector() as sub:
            if call.is_fp8:
                load_q(
                    Q,
                    QScale,
                    Cos,
                    Sin,
                    query,
                    batch,
                    head,
                    mt * BM + sub * R,
                    0,
                    raw_fp8,
                    scale_fp8,
                    transpose_fp8,
                )
            else:
                load_q(Q, QScale, Cos, Sin, query, batch, head, mt * BM + sub * R, 0)
            T.copy(query[0, :, :], qnz[:R, :])
            T.ascend_set_flag("V_MTE3", 6)
            T.ascend_wait_flag("V_MTE3", 6)
            T.dual_copy(qnz[:R, :], ql)
            T.ascend_cross_core_set_flag(4, "PIPE_MTE3", 15)
            if call.has_sinks:
                T.copy(Sinks[head * G : (head + 1) * G], sink_logits)
                T.ascend_set_flag("MTE2_V", 6)
                T.ascend_wait_flag("MTE2_V", 6)
            with T.SimdVF():
                full = S.pset(32, "PAT_ALL")
                zero = S.vdup(T.float32(0), "float32", full)
                neg = S.vdup(T.float32(-3.402823e38), "float32", full)
                if call.has_sinks:
                    # Treat the per-query-head sink as a zero-valued token.
                    # A finite ceiling gives +inf sink unit mass throughout the recurrence.
                    ceiling = S.vdup(T.float32(3.402823e38), "float32", full)
                    one = S.pset(32, "PAT_VL1")
                    for r in range(R):
                        sink = S.vmin(S.vld(sink_logits[r % G], "BRC_B32"), ceiling, full)
                        S.vsts(maximum[r], sink, one, "ONEPT_B32")
                    S.vsts(denom[0], S.vdup(T.float32(1), "float32", full), full, "NORM_B32")
                else:
                    S.vsts(maximum[0], neg, full, "NORM_B32")
                    S.vsts(denom[0], zero, full, "NORM_B32")
                for r in range(R):
                    for d in range(0, D, 64):
                        S.vsts(acc[r, d], zero, full, "NORM_B32")
            for slot in T.serial(2):
                T.ascend_cross_core_set_flag(4, "PIPE_V", 5 + slot)
                T.ascend_cross_core_set_flag(4, "PIPE_V", 13 + slot)
            operands(
                K,
                V,
                KScale,
                VScale,
                Cos,
                Sin,
                key,
                value,
                knz,
                vnz,
                kl,
                vl,
                batch,
                head,
                0,
                sub,
                raw_fp8,
                scale_fp8,
                transpose_fp8,
            )
            for step in T.serial(count):
                if step + 1 < count:
                    operands(
                        K,
                        V,
                        KScale,
                        VScale,
                        Cos,
                        Sin,
                        key,
                        value,
                        knz,
                        vnz,
                        kl,
                        vl,
                        batch,
                        head,
                        step + 1,
                        sub,
                        raw_fp8,
                        scale_fp8,
                        transpose_fp8,
                    )
                if step >= 1:
                    previous = (step - 1) % 2
                    T.ascend_cross_core_wait_flag(4, "PIPE_V", T.if_then_else(previous == 0, 15, 0))
                    accumulate(current, acc, alpha, previous)
                    T.ascend_cross_core_set_flag(4, "PIPE_V", 13 + previous)
                slot = step % 2
                T.ascend_cross_core_wait_flag(4, "PIPE_V", 7 + slot)
                T.ascend_cross_core_wait_flag(4, "PIPE_V", 9 + slot)
                online(scores, probs, maximum, denom, alpha, mt, sub, step, slot)
                T.ascend_cross_core_set_flag(4, "PIPE_V", 5 + slot)
                T.copy(probs[slot, :, :], pnz[slot, :R, :])
                T.ascend_set_flag("V_MTE3", 6)
                T.ascend_wait_flag("V_MTE3", 6)
                T.dual_copy(pnz[slot, :R, :], pl[slot, :, :])
                T.ascend_cross_core_set_flag(4, "PIPE_MTE3", 11 + slot)
            last = (count - 1) % 2
            T.ascend_cross_core_wait_flag(4, "PIPE_V", T.if_then_else(last == 0, 15, 0))
            accumulate(current, acc, alpha, last)
            T.ascend_cross_core_set_flag(4, "PIPE_V", 13 + last)
            for slot in T.serial(2):
                T.ascend_cross_core_wait_flag(4, "PIPE_V", 3 + slot)
                T.ascend_cross_core_wait_flag(4, "PIPE_V", 9 + slot)
            with T.SimdVF():
                full = S.pset(32, "PAT_ALL")
                for r in range(R):
                    divisor = S.vld(denom[r], "BRC_B32")
                    for d in range(0, D, 64):
                        result = S.vdiv(S.vld(acc[r, d], "NORM"), divisor, full)
                        S.vsts(output[r, d], S.vcvt(result, dtype, full), full, "PK_B32")
            T.ascend_set_flag("V_MTE3", 6)
            T.ascend_wait_flag("V_MTE3", 6)
            if grouped_io:
                for group in T.serial(R // G):
                    pos = (mt * BM + sub * R) // G + group
                    if SQ * G % BM == 0 or pos < SQ:
                        T.copy(
                            output[group * G : (group + 1) * G, :],
                            O[batch, pos, head * G : (head + 1) * G, :],
                        )
            else:
                for r in T.serial(R):
                    row = mt * BM + sub * R + r
                    if row < SQ * G:
                        T.copy(output[r, :], O[batch, row // G, head * G + row % G, :])
            T.ascend_set_flag("MTE3_V", 0)
            T.ascend_wait_flag("MTE3_V", 0)

    if call.has_sinks:

        @T.prim_func
        def fp8_sink(
            Q: T.Tensor((B, SQ, H, D), in_type),
            K: T.Tensor((B, SK, HK, D), in_type),
            V: T.Tensor((B, SK, HK, D), in_type),
            QScale: T.Tensor((B, HK), "float32"),
            KScale: T.Tensor((B, HK), "float32"),
            VScale: T.Tensor((B, HK), "float32"),
            Sinks: T.Tensor((H,), "float32"),
            O: T.Tensor((B, SQ, H, D), dtype),
        ):
            with T.Kernel(min(num_cores, B * HK * (QR // BM))) as bx:
                for work in T.serial(bx, B * HK * (QR // BM), min(num_cores, B * HK * (QR // BM))):
                    tile(Q, K, V, QScale, KScale, VScale, Q, Q, Sinks, O, work)

        return fp8_sink.with_attr("global_symbol", "main")

    @T.prim_func
    def pipelined(
        Q: T.Tensor((B, SQ, H, D), in_type),
        K: T.Tensor((B, SK, HK, D), in_type),
        V: T.Tensor((B, SK, HK, D), in_type),
        QScale: T.Tensor((B, HK), "float32"),
        KScale: T.Tensor((B, HK), "float32"),
        VScale: T.Tensor((B, HK), "float32"),
        Cos: T.Tensor((POS, ROT // 2), dtype),
        Sin: T.Tensor((POS, ROT // 2), dtype),
        O: T.Tensor((B, SQ, H, D), dtype),
    ):
        with T.Kernel(min(num_cores, B * HK * (QR // BM))) as bx:
            for work in T.serial(bx, B * HK * (QR // BM), min(num_cores, B * HK * (QR // BM))):
                tile(Q, K, V, QScale, KScale, VScale, Cos, Sin, Cos, O, work)

    return pipelined.with_attr("global_symbol", "main")


@functools.lru_cache(maxsize=128)
def prefill_query_major(call, num_cores, packed=False):
    q_len, kv_len = call.max_seqlen_q, call.seqlen_kv
    batch_size, num_heads, num_kv_heads = call.batch, call.heads, call.heads_kv
    group = num_heads // num_kv_heads
    out_dtype = str(call.dtype).split(".")[-1]
    softmax_scale = call.sm_scale
    assert call.dim in (64, 128) and out_dtype in ("float16", "bfloat16")
    assert packed or (q_len % 128 == 0 and kv_len % 128 == 0 and kv_len >= q_len)
    assert call.is_causal and call.softcap == 0
    assert not call.is_fp8 and not call.fuse_rope
    assert (packed or call.window_size_left in (-1, 127)) and call.window_size_right == -1
    window_left = call.window_size_left
    assert num_cores > 0
    BR, BC, ROWS, VL, D, D0 = 128, 128, 64, 64, call.dim, call.dim
    M_TILES = q_len // BR
    # Interleave heads so causal Q positions are balanced across cores.
    # Preserve the existing non-sink specializations exactly.
    TILE_STRIDE = num_heads if call.has_sinks else 1
    HEAD_STRIDE = 1 if call.has_sinks else M_TILES
    TASKS = batch_size * num_heads * M_TILES
    # Every launched block must own a Q tile: the bounds assumptions below
    # allow lowering to remove the zero-task path for otherwise idle blocks.
    num_cores = num_cores if packed else min(num_cores, TASKS)
    PREFIX = (kv_len - q_len) // BC + 1
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)

    DIRECT_SCORE = call.has_sinks and window_left < 0 and D == 64 and scale == 0.125
    VECTOR_UPDATE = DIRECT_SCORE and not packed and q_len == kv_len

    # Consecutive Q tasks on core cid visit head (task * num_cores + cid) % num_heads.
    # Cache that short cycle in aligned UB slots for the lifetime of the kernel.
    sink_heads_per_core = num_heads // num_cores
    cache_sinks = (
        call.has_sinks
        and D == 64
        and window_left < 0
        and num_heads % num_cores == 0
        and sink_heads_per_core in (1, 2)
    )

    @T.macro
    def softmax(
        score,
        prob,
        maximum,
        denominator,
        alpha,
        old_max,
        row_sum,
        task,
        sequence,
        sid,
        first,
        masked,
        qk_offset,
    ):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            one = S.pset(32, "PAT_VL1")
            b16 = S.pset(16, "PAT_VL128")
            ss = sequence % 3
            ts = task % 3
            dst = S.make_ubuf_ptr(
                T.access_ptr(prob[sequence % 2, 0, 0], "w", ROWS + 1, BC), out_dtype
            )
            if VECTOR_UPDATE and not first:
                saved = S.alloc_local((1,), "float32")
                rescale = S.alloc_local((1,), "float32")
            if not first:
                if VECTOR_UPDATE:
                    saved[0] = S.vld(maximum[ts, 0], "NORM")
                else:
                    S.vsts(old_max[0], S.vld(maximum[ts, 0], "NORM"), full, "NORM_B32")
            for r in range(0, ROWS, 2):
                if DIRECT_SCORE and not masked:
                    s00 = S.vld(score[ss, r, 0], "NORM")
                    s01 = S.vld(score[ss, r, VL], "NORM")
                    s10 = S.vld(score[ss, r + 1, 0], "NORM")
                    s11 = S.vld(score[ss, r + 1, VL], "NORM")
                else:
                    s00 = S.vmuls(S.vld(score[ss, r, 0], "NORM"), T.float32(scale), full)
                    s01 = S.vmuls(S.vld(score[ss, r, VL], "NORM"), T.float32(scale), full)
                    s10 = S.vmuls(S.vld(score[ss, r + 1, 0], "NORM"), T.float32(scale), full)
                    s11 = S.vmuls(S.vld(score[ss, r + 1, VL], "NORM"), T.float32(scale), full)
                if masked:
                    upper_offset = 0 if window_left >= 0 else qk_offset
                    neg = S.vdup(T.float32(float("-inf")), "float32", full)
                    masked00 = S.vsel(
                        s00,
                        neg,
                        S.update_mask(T.max(0, T.min(VL, upper_offset + sid * ROWS + r + 1)), 32),
                    )
                    masked01 = S.vsel(
                        s01,
                        neg,
                        S.update_mask(
                            T.max(0, T.min(VL, upper_offset + sid * ROWS + r + 1 - VL)), 32
                        ),
                    )
                    masked10 = S.vsel(
                        s10,
                        neg,
                        S.update_mask(T.max(0, T.min(VL, upper_offset + sid * ROWS + r + 2)), 32),
                    )
                    masked11 = S.vsel(
                        s11,
                        neg,
                        S.update_mask(
                            T.max(0, T.min(VL, upper_offset + sid * ROWS + r + 2 - VL)), 32
                        ),
                    )
                else:
                    masked00, masked01, masked10, masked11 = s00, s01, s10, s11
                if window_left >= 0 and (packed or not masked):
                    neg = S.vdup(T.float32(float("-inf")), "float32", full)
                    lo0 = (qk_offset if packed else BC) + sid * ROWS + r - window_left
                    lo1 = lo0 + 1
                    masked00 = S.vsel(neg, masked00, S.update_mask(T.max(0, T.min(VL, lo0)), 32))
                    masked01 = S.vsel(
                        neg, masked01, S.update_mask(T.max(0, T.min(VL, lo0 - VL)), 32)
                    )
                    masked10 = S.vsel(neg, masked10, S.update_mask(T.max(0, T.min(VL, lo1)), 32))
                    masked11 = S.vsel(
                        neg, masked11, S.update_mask(T.max(0, T.min(VL, lo1 - VL)), 32)
                    )
                if not (DIRECT_SCORE and not masked):
                    S.vsts(score[ss, r, 0], masked00, full, "NORM_B32")
                    S.vsts(score[ss, r, VL], masked01, full, "NORM_B32")
                    S.vsts(score[ss, r + 1, 0], masked10, full, "NORM_B32")
                    S.vsts(score[ss, r + 1, VL], masked11, full, "NORM_B32")
                max0 = S.vcmax(S.vmax(masked00, masked01, full), full)
                max1 = S.vcmax(S.vmax(masked10, masked11, full), full)
                if window_left >= 0:
                    # A boundary KV block may be empty for some query rows.
                    # Keep its maximum finite so exp(-inf - max) is zero.
                    floor = S.vdup(T.float32(-3.402823e38), "float32", full)
                    max0 = S.vmax(max0, floor, full)
                    max1 = S.vmax(max1, floor, full)
                S.vsts(maximum[ts, r], max0, one, "ONEPT_B32")
                S.vsts(maximum[ts, r + 1], max1, one, "ONEPT_B32")
            S.mem_bar("VST_VLD")
            if DIRECT_SCORE and not masked:
                # Scale all 64 row maxima in one vector before the exp pass.
                scaled = S.vmuls(S.vld(maximum[ts, 0], "NORM"), T.float32(scale), full)
                if not first:
                    if VECTOR_UPDATE:
                        scaled = S.vmax(scaled, saved[0], full)
                        rescale[0] = S.vexpdif(saved[0], scaled, full)
                    else:
                        scaled = S.vmax(scaled, S.vld(old_max[0], "NORM"), full)
                S.vsts(maximum[ts, 0], scaled, full, "NORM_B32")
                S.mem_bar("VST_VLD")
            elif not first:
                if VECTOR_UPDATE:
                    merged = S.vmax(S.vld(maximum[ts, 0], "NORM"), saved[0], full)
                    rescale[0] = S.vexpdif(saved[0], merged, full)
                else:
                    merged = S.vmax(S.vld(maximum[ts, 0], "NORM"), S.vld(old_max[0], "NORM"), full)
                S.vsts(maximum[ts, 0], merged, full, "NORM_B32")
                S.mem_bar("VST_VLD")
            if VECTOR_UPDATE and not first and not masked:
                # Interleave the two rows without extending the loop body.
                for r in range(0, ROWS, 2):
                    m0 = S.vld(maximum[ts, r], "BRC_B32")
                    x00, x01 = S.vld2(score[ss, r, 0], "DINTLV_B32")
                    x00 = S.vmuls(x00, T.float32(scale), full)
                    x01 = S.vmuls(x01, T.float32(scale), full)
                    m1 = S.vld(maximum[ts, r + 1], "BRC_B32")
                    x10, x11 = S.vld2(score[ss, r + 1, 0], "DINTLV_B32")
                    e00 = S.vexpdif(x00, m0, full)
                    e01 = S.vexpdif(x01, m0, full)
                    x10 = S.vmuls(x10, T.float32(scale), full)
                    x11 = S.vmuls(x11, T.float32(scale), full)
                    sum0 = S.vcadd(S.vadd(e00, e01, full), full)
                    S.vsts(row_sum[r], sum0, one, "ONEPT_B32")
                    e10 = S.vexpdif(x10, m1, full)
                    e11 = S.vexpdif(x11, m1, full)
                    even0 = S.vcvt(e00, out_dtype, full, sat=False, part=0)
                    odd0 = S.vcvt(e01, out_dtype, full, sat=False, part=1)
                    bits0 = S.vor(  # codespell:ignore vor
                        T.reinterpret(even0, "uint16x128"), T.reinterpret(odd0, "uint16x128"), b16
                    )
                    dst = S.vsstb(
                        T.reinterpret(bits0, f"{out_dtype}x128"),
                        dst,
                        T.int32(((ROWS + 1) << 16) | 1),
                        b16,
                        update=True,
                    )
                    sum1 = S.vcadd(S.vadd(e10, e11, full), full)
                    S.vsts(row_sum[r + 1], sum1, one, "ONEPT_B32")
                    even1 = S.vcvt(e10, out_dtype, full, sat=False, part=0)
                    odd1 = S.vcvt(e11, out_dtype, full, sat=False, part=1)
                    bits1 = S.vor(  # codespell:ignore vor
                        T.reinterpret(even1, "uint16x128"), T.reinterpret(odd1, "uint16x128"), b16
                    )
                    dst = S.vsstb(
                        T.reinterpret(bits1, f"{out_dtype}x128"),
                        dst,
                        T.int32(((ROWS + 1) << 16) | 1),
                        b16,
                        update=True,
                    )
            else:
                for r in range(0, ROWS, 2):
                    m0 = S.vld(maximum[ts, r], "BRC_B32")
                    m1 = S.vld(maximum[ts, r + 1], "BRC_B32")
                    x00, x01 = S.vld2(score[ss, r, 0], "DINTLV_B32")
                    x10, x11 = S.vld2(score[ss, r + 1, 0], "DINTLV_B32")
                    if DIRECT_SCORE and not masked:
                        x00 = S.vmuls(x00, T.float32(scale), full)
                        x01 = S.vmuls(x01, T.float32(scale), full)
                        x10 = S.vmuls(x10, T.float32(scale), full)
                        x11 = S.vmuls(x11, T.float32(scale), full)
                    e00 = S.vexpdif(x00, m0, full)
                    e01 = S.vexpdif(x01, m0, full)
                    e10 = S.vexpdif(x10, m1, full)
                    e11 = S.vexpdif(x11, m1, full)
                    sum0 = S.vcadd(S.vadd(e00, e01, full), full)
                    sum1 = S.vcadd(S.vadd(e10, e11, full), full)
                    S.vsts(row_sum[r], sum0, one, "ONEPT_B32")
                    S.vsts(row_sum[r + 1], sum1, one, "ONEPT_B32")
                    even0 = S.vcvt(e00, out_dtype, full, sat=False, part=0)
                    even1 = S.vcvt(e10, out_dtype, full, sat=False, part=0)
                    odd0 = S.vcvt(e01, out_dtype, full, sat=False, part=1)
                    odd1 = S.vcvt(e11, out_dtype, full, sat=False, part=1)
                    bits0 = S.vor(  # codespell:ignore vor
                        T.reinterpret(even0, "uint16x128"), T.reinterpret(odd0, "uint16x128"), b16
                    )
                    bits1 = S.vor(  # codespell:ignore vor
                        T.reinterpret(even1, "uint16x128"), T.reinterpret(odd1, "uint16x128"), b16
                    )
                    dst = S.vsstb(
                        T.reinterpret(bits0, f"{out_dtype}x128"),
                        dst,
                        T.int32(((ROWS + 1) << 16) | 1),
                        b16,
                        update=True,
                    )
                    dst = S.vsstb(
                        T.reinterpret(bits1, f"{out_dtype}x128"),
                        dst,
                        T.int32(((ROWS + 1) << 16) | 1),
                        b16,
                        update=True,
                    )
            S.mem_bar("VST_VLD")
            if first:
                S.vsts(denominator[ts, 0], S.vld(row_sum[0], "NORM"), full, "NORM_B32")
            else:
                if VECTOR_UPDATE:
                    a = rescale[0]
                else:
                    a = S.vexpdif(S.vld(old_max[0], "NORM"), S.vld(maximum[ts, 0], "NORM"), full)
                S.vsts(alpha[ss, 0], a, full, "NORM_B32")
                updated = S.vadd(
                    S.vmul(S.vld(denominator[ts, 0], "NORM"), a, full),
                    S.vld(row_sum[0], "NORM"),
                    full,
                )
                S.vsts(denominator[ts, 0], updated, full, "NORM_B32")

    @T.macro
    def update_output(
        pv,
        accum,
        output,
        alpha,
        denominator,
        maximum,
        sink_ub,
        normalizer,
        task,
        sequence,
        first,
        last,
    ):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            if last and call.has_sinks:
                # Compute one coefficient per row, not 64 identical lanes per
                # output row. row_sum is scratch: PV is consumed before the
                # next softmax writes it in the persistent vector pipeline.
                m = S.vld(maximum[task % 3, 0], "NORM")
                d = S.vld(denominator[task % 3, 0], "NORM")
                sink = S.vmin(
                    S.vld(
                        sink_ub[(task % sink_heads_per_core) * 8 if cache_sinks else 0], "BRC_B32"
                    ),
                    S.vdup(T.float32(3.402823e38), "float32", full),
                    full,
                )
                merged = S.vmax(m, sink, full)
                weight = S.vexpdif(m, merged, full)
                divisor = S.vadd(S.vmul(d, weight, full), S.vexpdif(sink, merged, full), full)
                S.vsts(normalizer[0], S.vdiv(weight, divisor, full), full, "NORM_B32")
                S.mem_bar("VST_VLD")
            for r in range(ROWS):
                for c in range(0, D0, VL):
                    current = S.vld(pv[sequence % 2, r, c], "NORM")
                    if first:
                        updated = current
                    else:
                        a = S.vld(alpha[sequence % 3, r], "BRC_B32")
                        prev = S.vld(accum[r, c], "NORM")
                        if VECTOR_UPDATE and last:
                            result = S.alloc_local((1,), "float32")
                            result[0] = prev
                            S.vmadd(result[0], a, current, full)
                            updated = result[0]
                        else:
                            updated = S.vadd(S.vmul(a, prev, full), current, full)
                    if last:
                        if call.has_sinks:
                            normalized = S.vmul(updated, S.vld(normalizer[r], "BRC_B32"), full)
                        else:
                            d = S.vld(denominator[task % 3, r], "BRC_B32")
                            normalized = S.vdiv(updated, d, full)
                        bf16 = S.vcvt(normalized, out_dtype, full)
                        S.vsts(output[r, c], bf16, full, "PK_B32")
                    else:
                        S.vsts(accum[r, c], updated, full, "NORM_B32")

    window_blocks = (1 + (window_left + BC - 1) // BC) if packed and window_left >= 0 else 2

    @T.macro
    def execute(Q, K, V, Sinks, O, LSE, cid, q_origin, kv_origin,
                q_len, kv_len, M_TILES, HEAD_STRIDE, TASKS, PREFIX):
        T.assume(q_len > 0)
        T.assume(kv_len > 0)
        T.assume(batch_size > 0)
        T.assume(num_heads > 0)
        T.assume(q_len % 128 == 0)
        T.assume(kv_len % 128 == 0)
        T.assume(kv_len >= q_len)
        q_l1 = T.alloc_l1((2, BR, D), out_dtype)
        k_l1 = T.alloc_l1((3, BC, D), out_dtype)
        v_l1 = T.alloc_l1((3, D0, BC), out_dtype)
        p_l1 = T.alloc_l1((2, BR, BC), out_dtype)
        # Shared L0 rings have fixed 32 KiB slots for BOTH GEMMs.
        a_l0 = T.alloc_l0a((2, BR, BC), out_dtype)
        b_l0 = T.alloc_l0b((2, BC, BC), out_dtype)
        qk_l0 = T.alloc_l0c((2, BR, BC), "float32")
        pv_l0 = T.alloc_l0c((2, BR, D0), "float32")
        score = T.alloc_shared((3, ROWS, BC), "float32")
        prob = T.alloc_shared((2, ROWS + 1, BC), out_dtype)
        T.annotate_layout({prob: make_ascend_compact_nz_layout(prob)})
        pv = T.alloc_shared((2, ROWS, D0), "float32")
        accum = T.alloc_shared((ROWS, D0), "float32")
        maximum = T.alloc_shared((3, ROWS), "float32")
        denominator = T.alloc_shared((3, ROWS), "float32")
        alpha = T.alloc_shared((3, ROWS), "float32")
        old_max = T.alloc_shared((ROWS,), "float32")
        row_sum = T.alloc_shared((ROWS,), "float32")
        output = T.alloc_shared((ROWS, D0), out_dtype)
        if call.return_lse:
            row_lse = T.alloc_shared((ROWS,), "float32")
        sink_ub = T.alloc_shared((8 * sink_heads_per_core if cache_sinks else 8,), "float32")

        tasks = T.ceildiv(TASKS - cid, num_cores)
        total = T.alloc_var("int32", init=0)
        for task in T.serial(tasks):
            total += (
                (task * num_cores + cid) // TILE_STRIDE % M_TILES
                + PREFIX
                - (
                    T.max(0, (task * num_cores + cid) // TILE_STRIDE % M_TILES + PREFIX - window_blocks)
                    if window_left >= 0
                    else 0
                )
            )
        T.assume(total >= 0)
        T.assume(total <= T.ceildiv(TASKS, num_cores) * (kv_len // BC))

        with T.Cube():
            for e in T.serial(8):
                T.ascend_set_flag("MTE1_MTE2", e)
            for e in T.serial(4):
                T.ascend_set_flag("FIX_M", e)
            for e in T.serial(2):
                T.ascend_set_flag("M_MTE1", e)
                T.ascend_cross_core_set_flag(4, "PIPE_MTE1", 5 + e * 16)
                T.ascend_cross_core_set_flag(4, "PIPE_MTE1", 12 + e * 16)
            tk = T.alloc_var("int32", init=0)
            kk = T.alloc_var("int32", init=0)
            tq = T.alloc_var("int32", init=0)
            kq = T.alloc_var("int32", init=0)
            tv = T.alloc_var("int32", init=0)
            kv = T.alloc_var("int32", init=0)
            l0_sequence = T.alloc_var("int32", init=0)

            # t: load K/Q; t-1: QK; t-2: load V / QK fix;
            # t-4: PV; t-5: PV fix. Drain once for ALL Q tasks.
            for step in T.serial(total + 5):
                # Feed older PV before prefetching future Q/K tiles.
                if call.has_sinks and window_left < 0:  # noqa: SIM102 - Static specialization guard.
                    if step >= 2 and step < total + 2:
                        g = step - 2
                        tile = tv * num_cores + cid
                        T.assume(0 <= tile < TASKS)
                        T.assume(0 <= kv < kv_len // BC)
                        batch = tile // (num_heads * M_TILES)
                        head = tile // HEAD_STRIDE % num_heads
                        T.ascend_wait_flag("MTE1_MTE2", g % 3)
                        T.copy(
                            V[
                                batch,
                                kv_origin + ((
                                    kv
                                    + (
                                        T.max(0, tile // TILE_STRIDE % M_TILES + PREFIX - window_blocks)
                                        if window_left >= 0
                                        else 0
                                    )
                                )
                                * BC) : kv_origin + ((
                                    kv
                                    + (
                                        T.max(0, tile // TILE_STRIDE % M_TILES + PREFIX - window_blocks)
                                        if window_left >= 0
                                        else 0
                                    )
                                    + 1
                                )
                                * BC),
                                head // group,
                                :,
                            ],
                            v_l1[g % 3, :, :],
                            transpose=True,
                        )
                        T.ascend_set_flag("MTE2_MTE1", g % 3)
                        kv += 1
                        if kv == (
                            tile // TILE_STRIDE % M_TILES
                            + PREFIX
                            - (
                                T.max(0, tile // TILE_STRIDE % M_TILES + PREFIX - window_blocks)
                                if window_left >= 0
                                else 0
                            )
                        ):
                            kv = 0
                            tv += 1

                if step < total:
                    tile = tk * num_cores + cid
                    # Cursor invariants: the stream contains exactly the
                    # valid Q tasks and their visible KV tiles. Expose the
                    # bounds so OOB lowering need not add dynamic padding.
                    T.assume(0 <= tile < TASKS)
                    T.assume(0 <= kk < kv_len // BC)
                    batch = tile // (num_heads * M_TILES)
                    head = tile // HEAD_STRIDE % num_heads
                    mt = tile // TILE_STRIDE % M_TILES
                    if kk == 0:
                        T.ascend_wait_flag("MTE1_MTE2", 6 + tk % 2)
                        T.copy(Q[batch, q_origin + (mt * BR) : q_origin + ((mt + 1) * BR), head, :], q_l1[tk % 2, :, :])
                        T.ascend_set_flag("MTE2_MTE1", 6 + tk % 2)
                    T.ascend_wait_flag("MTE1_MTE2", 3 + step % 3)
                    T.copy(
                        K[
                            batch,
                            kv_origin + ((
                                kk
                                + (
                                    T.max(0, tile // TILE_STRIDE % M_TILES + PREFIX - window_blocks)
                                    if window_left >= 0
                                    else 0
                                )
                            )
                            * BC) : kv_origin + ((
                                kk
                                + (
                                    T.max(0, tile // TILE_STRIDE % M_TILES + PREFIX - window_blocks)
                                    if window_left >= 0
                                    else 0
                                )
                                + 1
                            )
                            * BC),
                            head // group,
                            :,
                        ],
                        k_l1[step % 3, :, :],
                    )
                    T.ascend_set_flag("MTE2_MTE1", 3 + step % 3)
                    kk += 1
                    if kk == (
                        tile // TILE_STRIDE % M_TILES
                        + PREFIX
                        - (
                            T.max(0, tile // TILE_STRIDE % M_TILES + PREFIX - window_blocks)
                            if window_left >= 0
                            else 0
                        )
                    ):
                        kk = 0
                        tk += 1

                if step >= 1 and step < total + 1:
                    g = step - 1
                    tile = tq * num_cores + cid
                    T.ascend_wait_flag("FIX_M", 2 + g % 2)
                    if kq == 0:
                        T.ascend_wait_flag("MTE2_MTE1", 6 + tq % 2)
                    T.ascend_wait_flag("MTE2_MTE1", 3 + g % 3)
                    for sk in T.serial(1):
                        ls = T.floormod(l0_sequence + sk, 2)
                        T.assume(0 <= ls < 2)
                        T.ascend_wait_flag("M_MTE1", ls)
                        T.copy(q_l1[tq % 2, :, sk * D : (sk + 1) * D], a_l0[ls, :, :D])
                        T.copy(k_l1[g % 3, :, sk * D : (sk + 1) * D], b_l0[ls, :, :D])
                        T.ascend_set_flag("MTE1_M", ls)
                        T.ascend_wait_flag("MTE1_M", ls)
                        T.gemm(
                            a_l0[ls, :, :D],
                            b_l0[ls, :, :D],
                            qk_l0[g % 2, :, :],
                            transpose_B=True,
                            clear_accum=sk == 0,
                        )
                        T.ascend_set_flag("M_MTE1", ls)
                    l0_sequence += 1
                    T.ascend_set_flag("M_FIX", 2 + g % 2)
                    T.ascend_set_flag("MTE1_MTE2", 3 + g % 3)
                    kq += 1
                    if kq == (
                        tile // TILE_STRIDE % M_TILES
                        + PREFIX
                        - (
                            T.max(0, tile // TILE_STRIDE % M_TILES + PREFIX - window_blocks)
                            if window_left >= 0
                            else 0
                        )
                    ):
                        T.ascend_set_flag("MTE1_MTE2", 6 + tq % 2)
                        kq = 0
                        tq += 1

                if not call.has_sinks or window_left >= 0:  # noqa: SIM102 - Static specialization guard.
                    if step >= 2 and step < total + 2:
                        g = step - 2
                        tile = tv * num_cores + cid
                        T.assume(0 <= tile < TASKS)
                        T.assume(0 <= kv < kv_len // BC)
                        batch = tile // (num_heads * M_TILES)
                        head = tile // HEAD_STRIDE % num_heads
                        T.ascend_wait_flag("MTE1_MTE2", g % 3)
                        T.copy(
                            V[
                                batch,
                                kv_origin + ((
                                    kv
                                    + (
                                        T.max(0, tile // TILE_STRIDE % M_TILES + PREFIX - window_blocks)
                                        if window_left >= 0
                                        else 0
                                    )
                                )
                                * BC) : kv_origin + ((
                                    kv
                                    + (
                                        T.max(0, tile // TILE_STRIDE % M_TILES + PREFIX - window_blocks)
                                        if window_left >= 0
                                        else 0
                                    )
                                    + 1
                                )
                                * BC),
                                head // group,
                                :,
                            ],
                            v_l1[g % 3, :, :],
                            transpose=True,
                        )
                        T.ascend_set_flag("MTE2_MTE1", g % 3)
                        kv += 1
                        if kv == (
                            tile // TILE_STRIDE % M_TILES
                            + PREFIX
                            - (
                                T.max(0, tile // TILE_STRIDE % M_TILES + PREFIX - window_blocks)
                                if window_left >= 0
                                else 0
                            )
                        ):
                            kv = 0
                            tv += 1

                # Publish ready QK scores before waiting on older PV work.
                if call.has_sinks and window_left < 0:  # noqa: SIM102 - Static specialization guard.
                    if step >= 2 and step < total + 2:
                        g = step - 2
                        T.ascend_cross_core_wait_flag(4, "PIPE_FIX", g % 3)
                        T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 16 + g % 3)
                        T.ascend_wait_flag("M_FIX", 2 + g % 2)
                        T.dual_copy(qk_l0[g % 2, :, :], score[g % 3, :, :])
                        T.ascend_cross_core_set_flag(4, "PIPE_FIX", 6 + g % 3)
                        T.ascend_cross_core_set_flag(4, "PIPE_FIX", 22 + g % 3)
                        T.ascend_set_flag("FIX_M", 2 + g % 2)

                if step >= 4 and step < total + 4:
                    g = step - 4
                    ready = T.if_then_else(g % 2 == 0, 11, 13)
                    free = T.if_then_else(g % 2 == 0, 5, 12)
                    T.ascend_wait_flag("FIX_M", g % 2)
                    T.ascend_wait_flag("MTE2_MTE1", g % 3)
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", ready)
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", ready + 16)
                    ls = T.floormod(l0_sequence, 2)
                    T.assume(0 <= ls < 2)
                    T.ascend_wait_flag("M_MTE1", ls)
                    T.copy(p_l1[g % 2, :, :], a_l0[ls, :, :])
                    T.copy(v_l1[g % 3, :, :], b_l0[ls, :D0, :])
                    T.ascend_set_flag("MTE1_M", ls)
                    T.ascend_wait_flag("MTE1_M", ls)
                    T.gemm(
                        a_l0[ls, :, :],
                        b_l0[ls, :D0, :],
                        pv_l0[g % 2, :, :],
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.ascend_set_flag("M_MTE1", ls)
                    l0_sequence += 1
                    T.ascend_set_flag("M_FIX", g % 2)
                    T.ascend_cross_core_set_flag(4, "PIPE_MTE1", free)
                    T.ascend_cross_core_set_flag(4, "PIPE_MTE1", free + 16)
                    T.ascend_set_flag("MTE1_MTE2", g % 3)

                if step >= 5:
                    g = step - 5
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 3 + g % 2)
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 19 + g % 2)
                    T.ascend_wait_flag("M_FIX", g % 2)
                    T.dual_copy(pv_l0[g % 2, :, :], pv[g % 2, :, :])
                    T.ascend_cross_core_set_flag(4, "PIPE_FIX", 9 + g % 2)
                    T.ascend_cross_core_set_flag(4, "PIPE_FIX", 25 + g % 2)
                    T.ascend_set_flag("FIX_M", g % 2)

                if not call.has_sinks or window_left >= 0:  # noqa: SIM102 - Static specialization guard.
                    if step >= 2 and step < total + 2:
                        g = step - 2
                        T.ascend_cross_core_wait_flag(4, "PIPE_FIX", g % 3)
                        T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 16 + g % 3)
                        T.ascend_wait_flag("M_FIX", 2 + g % 2)
                        T.dual_copy(qk_l0[g % 2, :, :], score[g % 3, :, :])
                        T.ascend_cross_core_set_flag(4, "PIPE_FIX", 6 + g % 3)
                        T.ascend_cross_core_set_flag(4, "PIPE_FIX", 22 + g % 3)
                        T.ascend_set_flag("FIX_M", 2 + g % 2)

            for e in T.serial(8):
                T.ascend_wait_flag("MTE1_MTE2", e)
            for e in T.serial(4):
                T.ascend_wait_flag("FIX_M", e)
            for sub in T.serial(2):
                T.ascend_wait_flag("M_MTE1", sub)
                for e in T.serial(5):
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", e + sub * 16)

        with T.Vector() as sid:
            for e in T.serial(5):
                T.ascend_cross_core_set_flag(4, "PIPE_V", e)
            for e in T.serial(3):
                T.ascend_set_flag("MTE3_V", e)
            if cache_sinks:
                for slot in T.unroll(sink_heads_per_core):
                    prefetch_head = cid + slot * num_cores
                    T.copy(
                        Sinks[prefetch_head : prefetch_head + 1],
                        sink_ub[slot * 8 : slot * 8 + 1],
                    )
                T.ascend_set_flag("MTE2_V", 0)
                T.ascend_wait_flag("MTE2_V", 0)
            st = T.alloc_var("int32", init=0)
            skv = T.alloc_var("int32", init=0)
            output_task = T.alloc_var("int32", init=0)
            okv = T.alloc_var("int32", init=0)

            # Consume g=t-3 BEFORE producing t: alpha has three slots.
            # Separate Q-task cursors keep the previous denominator live.
            for step in T.serial(total + 3):
                if step >= 3:
                    g = step - 3
                    tile = output_task * num_cores + cid
                    last = (
                        okv
                        == (
                            tile // TILE_STRIDE % M_TILES
                            + PREFIX
                            - (
                                T.max(0, tile // TILE_STRIDE % M_TILES + PREFIX - window_blocks)
                                if window_left >= 0
                                else 0
                            )
                        )
                        - 1
                    )
                    T.ascend_cross_core_wait_flag(4, "PIPE_V", 9 + g % 2)
                    if last:
                        T.ascend_wait_flag("MTE3_V", 2)
                        if call.has_sinks and not cache_sinks:
                            head = tile // HEAD_STRIDE % num_heads
                            T.ascend_set_flag("V_MTE2", 0)
                            T.ascend_wait_flag("V_MTE2", 0)
                            T.copy(Sinks[head : head + 1], sink_ub[:1])
                            T.ascend_set_flag("MTE2_V", 0)
                            T.ascend_wait_flag("MTE2_V", 0)
                        if okv == 0:
                            update_output(
                                pv,
                                accum,
                                output,
                                alpha,
                                denominator,
                                maximum,
                                sink_ub,
                                row_sum,
                                output_task,
                                g,
                                True,
                                True,
                            )
                        else:
                            update_output(
                                pv,
                                accum,
                                output,
                                alpha,
                                denominator,
                                maximum,
                                sink_ub,
                                row_sum,
                                output_task,
                                g,
                                False,
                                True,
                            )
                    elif okv == 0:
                        update_output(
                            pv,
                            accum,
                            output,
                            alpha,
                            denominator,
                            maximum,
                            sink_ub,
                            row_sum,
                            output_task,
                            g,
                            True,
                            False,
                        )
                    else:
                        update_output(
                            pv,
                            accum,
                            output,
                            alpha,
                            denominator,
                            maximum,
                            sink_ub,
                            row_sum,
                            output_task,
                            g,
                            False,
                            False,
                        )
                    T.ascend_cross_core_set_flag(4, "PIPE_V", 3 + g % 2)
                    if last:
                        T.assume(0 <= tile < TASKS)
                        batch = tile // (num_heads * M_TILES)
                        head = tile // HEAD_STRIDE % num_heads
                        mt = tile // TILE_STRIDE % M_TILES
                        if call.return_lse:
                            with T.SimdVF():
                                full = S.pset(32)
                                m = S.vld(maximum[output_task % 3, 0], "NORM")
                                d = S.vld(denominator[output_task % 3, 0], "NORM")
                                S.vsts(row_lse[0], S.vadd(m, S.vln(d, full), full), full, "NORM_B32")
                        T.ascend_set_flag("V_MTE3", 2)
                        T.ascend_wait_flag("V_MTE3", 2)
                        T.copy(
                            output,
                            O[
                                batch,
                                q_origin + (mt * BR + sid * ROWS) : q_origin + (mt * BR + (sid + 1) * ROWS),
                                head,
                                :,
                            ],
                            l2_cache_ctrl=0,
                        )
                        if call.return_lse:
                            T.copy(row_lse, LSE[batch, head, q_origin + (mt*BR+sid*ROWS):q_origin + (mt*BR+(sid+1)*ROWS)])
                        T.ascend_set_flag("MTE3_V", 2)
                        okv = 0
                        output_task += 1
                    else:
                        okv += 1

                if step < total:
                    tile = st * num_cores + cid
                    last = (
                        skv
                        == (
                            tile // TILE_STRIDE % M_TILES
                            + PREFIX
                            - (
                                T.max(0, tile // TILE_STRIDE % M_TILES + PREFIX - window_blocks)
                                if window_left >= 0
                                else 0
                            )
                        )
                        - 1
                    )
                    T.ascend_wait_flag("MTE3_V", step % 2)
                    T.ascend_cross_core_wait_flag(4, "PIPE_V", 6 + step % 3)
                    if skv == 0:
                        if last:
                            softmax(
                                score,
                                prob,
                                maximum,
                                denominator,
                                alpha,
                                old_max,
                                row_sum,
                                st,
                                step,
                                sid,
                                True,
                                True,
                                (
                                    tile // TILE_STRIDE % M_TILES
                                    + PREFIX
                                    - 1
                                    - skv
                                    - (
                                        T.max(0, tile // TILE_STRIDE % M_TILES + PREFIX - window_blocks)
                                        if window_left >= 0
                                        else 0
                                    )
                                )
                                * BC,
                            )
                        else:
                            softmax(
                                score,
                                prob,
                                maximum,
                                denominator,
                                alpha,
                                old_max,
                                row_sum,
                                st,
                                step,
                                sid,
                                True,
                                False,
                                (
                                    tile // TILE_STRIDE % M_TILES
                                    + PREFIX
                                    - 1
                                    - skv
                                    - (
                                        T.max(0, tile // TILE_STRIDE % M_TILES + PREFIX - window_blocks)
                                        if window_left >= 0
                                        else 0
                                    )
                                )
                                * BC,
                            )
                    elif last:
                        softmax(
                            score,
                            prob,
                            maximum,
                            denominator,
                            alpha,
                            old_max,
                            row_sum,
                            st,
                            step,
                            sid,
                            False,
                            True,
                            (
                                tile // TILE_STRIDE % M_TILES
                                + PREFIX
                                - 1
                                - skv
                                - (
                                    T.max(0, tile // TILE_STRIDE % M_TILES + PREFIX - window_blocks)
                                    if window_left >= 0
                                    else 0
                                )
                            )
                            * BC,
                        )
                    else:
                        softmax(
                            score,
                            prob,
                            maximum,
                            denominator,
                            alpha,
                            old_max,
                            row_sum,
                            st,
                            step,
                            sid,
                            False,
                            False,
                            (
                                tile // TILE_STRIDE % M_TILES
                                + PREFIX
                                - 1
                                - skv
                                - (
                                    T.max(0, tile // TILE_STRIDE % M_TILES + PREFIX - window_blocks)
                                    if window_left >= 0
                                    else 0
                                )
                            )
                            * BC,
                        )
                    T.ascend_set_flag("V_MTE3", step % 2)
                    T.ascend_cross_core_set_flag(4, "PIPE_V", step % 3)
                    T.ascend_cross_core_wait_flag(
                        4, "PIPE_MTE3", T.if_then_else(step % 2 == 0, 5, 12)
                    )
                    T.ascend_wait_flag("V_MTE3", step % 2)
                    T.dual_copy(prob[step % 2, :ROWS, :], p_l1[step % 2, :, :])
                    T.ascend_cross_core_set_flag(
                        4, "PIPE_MTE3", T.if_then_else(step % 2 == 0, 11, 13)
                    )
                    T.ascend_set_flag("MTE3_V", step % 2)
                    if last:
                        skv = 0
                        st += 1
                    else:
                        skv += 1

            for e in T.serial(3):
                T.ascend_wait_flag("MTE3_V", e)
            T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", 5)
            T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", 12)

    if packed:
        all_q = T.dynamic("all_q")
        all_k = T.dynamic("all_k")

        @T.prim_func
        def ragged(
            Q: T.Tensor((1, all_q, num_heads, D), out_dtype),
            K: T.Tensor((1, all_k, num_kv_heads, D), out_dtype),
            V: T.Tensor((1, all_k, num_kv_heads, D0), out_dtype),
            Sinks: T.Tensor((num_heads,), "float32"),
            O: T.Tensor((1, all_q, num_heads, D0), out_dtype),
            Cq: T.Tensor((call.batch + 1,), "int32"),
            Ck: T.Tensor((call.batch + 1,), "int32"),
        ):
            with T.Kernel(num_cores * call.batch) as block:
                req = block // num_cores
                cid = block % num_cores
                qstart = Cq[req]
                kstart = Ck[req]
                sq = Cq[req + 1] - qstart
                sk = Ck[req + 1] - kstart
                mt = sq // BR
                jobs = num_heads * mt
                if sq > 0 and sk >= sq and sq % BR == 0 and sk % BC == 0 and cid < jobs:
                    execute(Q, K, V, Sinks, O, None, cid, qstart, kstart, sq, sk,
                            mt, 1 if call.has_sinks else mt, jobs, (sk - sq) // BC + 1)
        return ragged

    @T.prim_func
    def main(
        Q: T.Tensor((batch_size, q_len, num_heads, D), out_dtype),
        K: T.Tensor((batch_size, kv_len, num_kv_heads, D), out_dtype),
        V: T.Tensor((batch_size, kv_len, num_kv_heads, D0), out_dtype),
        Sinks: T.Tensor((num_heads,), "float32"),
        O: T.Tensor((batch_size, q_len, num_heads, D0), out_dtype),
        LSE: T.Tensor((batch_size, num_heads, q_len), "float32") if call.return_lse else None,
    ):
        with T.Kernel(num_cores) as cid:
            execute(Q, K, V, Sinks, O, LSE, cid, 0, 0, q_len, kv_len, M_TILES,
                    HEAD_STRIDE, TASKS, PREFIX)

    return main


@functools.lru_cache(maxsize=128)
def pack_operand(call, kind, cube_cores):
    """Pack BSHD into independent KV groups, applying scale then rounded RoPE."""
    H, HK, D = call.heads, call.heads_kv, call.dim
    B, S = T.dynamic("batch"), T.dynamic("length")
    G = H // HK
    is_q = kind == "q"
    rotate = call.fuse_rope and kind != "v"
    N = T.ceildiv(S * G, 32) * 32 if is_q else T.ceildiv(S, 128) * 128
    NH = H if is_q else HK
    R = call.rotary_dim if rotate else 2
    P = T.dynamic("positions") if rotate else 1
    interleaved = call.rope_layout == "interleaved"
    out_type = str(call.dtype).split(".")[-1]
    in_type = "uint8" if call.is_fp8 else out_type
    grid = 2 * cube_cores

    def read_value(x, scale, b, pos, ih, col, h):
        value = T.Cast("float32", x[b, pos, ih, col])
        if call.is_fp8:
            # Decode the E4M3FN byte before applying the KV-head scale.
            raw = T.Cast("int32", x[b, pos, ih, col])
            exponent = (raw >> 3) & 15
            magnitude = T.if_then_else(
                exponent == 0,
                (raw & 7) * (2.0**-9),
                ((raw & 7) + 8) * T.exp2(T.Cast("float32", exponent - 10)),
            )
            signed = T.if_then_else((raw & 128) != 0, -magnitude, magnitude)
            value = T.if_then_else(
                (raw & 127) == 127, T.reinterpret(T.uint32(0x7FC00000), "float32"), signed
            )
            value = T.Cast("float32", T.Cast(out_type, value * scale[b, h]))
        return value

    @T.prim_func
    def main(
        X: T.Tensor((B, S, NH, D), in_type),
        Scale: T.Tensor((B, HK), "float32"),
        Cos: T.Tensor((P, R // 2), out_type),
        Sin: T.Tensor((P, R // 2), out_type),
        position_offset: T.int32,
        Y: T.Tensor((B * HK, N, D), out_type),
    ):
        with T.Kernel(grid) as bx:
            for group in T.serial(B * HK):
                bh = T.Cast("int64", group)
                b = bh // HK
                h = group % HK
                for row in T.serial(bx, N, grid):
                    pos = row // G if is_q else row
                    ih = h * G + row % G if is_q else h
                    with T.SimtVF(threads=128):
                        for d in T.Parallel(D):
                            value = T.alloc_var("float32", init=0)
                            if pos < S:
                                value = read_value(X, Scale, b, pos, ih, d, h)
                                if rotate:  # noqa: SIM102 - eliminate the optional table access at specialization time
                                    if d < R:
                                        pair = d // 2 if interleaved else d % (R // 2)
                                        first = d % 2 == 0 if interleaved else d < R // 2
                                        other = (
                                            T.if_then_else(first, d + 1, d - 1)
                                            if interleaved
                                            else (d + R // 2) % R
                                        )
                                        partner = read_value(X, Scale, b, pos, ih, other, h)
                                        position = pos + position_offset if is_q else pos
                                        value = value * T.Cast(
                                            "float32", Cos[position, pair]
                                        ) + T.if_then_else(first, -partner, partner) * T.Cast(
                                            "float32", Sin[position, pair]
                                        )
                            Y[bh, row, d] = value

    return main


@functools.lru_cache(maxsize=128)
def attention_partial(call, splits):
    """Cube QK/PV on padded tiles; sq/sk retain the original mask boundaries."""
    D = call.dim
    G = call.heads // call.heads_kv
    BH, QR, KR = T.dynamic("groups"), T.dynamic("query_rows"), T.dynamic("kv_rows")
    BM, BN, R = 32, 128, 16
    NS = splits
    MT = QR // BM
    CHUNK = T.ceildiv(KR // BN, NS)
    grid = BH * MT * NS
    dtype = str(call.dtype).split(".")[-1]
    scale = D**-0.5 if call.sm_scale is None else call.sm_scale
    cap = call.softcap
    causal = call.is_causal
    wl, wr = call.window_size_left, call.window_size_right

    @T.prim_func
    def main(
        Q: T.Tensor((BH, QR, D), dtype),
        K: T.Tensor((BH, KR, D), dtype),
        V: T.Tensor((BH, KR, D), dtype),
        SQ: T.int32,
        SK: T.int32,
        Partial: T.Tensor((BH, NS, QR, D), "float32"),
        Stats: T.Tensor((BH, NS, QR, 2), "float32"),
    ):
        with T.MixedKernel(grid) as (bx, sub):
            split = bx % NS
            mt = bx // NS % MT
            bh = bx // (NS * MT)
            ql = T.alloc_l1((BM, D), dtype)
            kl = T.alloc_l1((BN, D), dtype)
            vl = T.alloc_l1((D, BN), dtype)
            pl = T.alloc_l1((BM, BN), dtype)
            qk = T.alloc_l0c((BM, BN), "float32")
            pv = T.alloc_l0c((BM, D), "float32")
            scores = T.alloc_shared((R, BN), "float32")
            probs = T.alloc_shared((R, BN), dtype)
            current = T.alloc_shared((R, D), "float32")
            acc = T.alloc_shared((R, D), "float32")
            maximum = T.alloc_shared((R,), "float32")
            denom = T.alloc_shared((R,), "float32")
            alpha = T.alloc_shared((R,), "float32")
            stats = T.alloc_shared((R, 2), "float32")
            T.copy(Q[bh, mt * BM : (mt + 1) * BM, :], ql)
            with T.SimtVF(threads=128):
                for r, d in T.Parallel(R, D):
                    acc[r, d] = 0
                for r in T.Parallel(R):
                    maximum[r] = -1e30
                    denom[r] = 0
            for it in T.serial(CHUNK):
                kv = split * CHUNK + it
                if kv * BN < SK:
                    T.copy(K[bh, kv * BN : (kv + 1) * BN, :], kl)
                    T.gemm(ql, kl, qk, transpose_B=True, clear_accum=True)
                    T.dual_copy(qk, scores)
                    with T.SimtVF(threads=128):
                        for r, j in T.Parallel(R, BN):
                            qi = (mt * BM + sub * R + r) // G
                            pos = qi + SK - SQ
                            key = kv * BN + j
                            visible = key < SK and qi < SQ
                            if causal:
                                visible = visible and key <= pos
                            if wl >= 0:
                                visible = visible and key >= pos - wl
                            if wr >= 0:
                                visible = visible and key <= pos + wr
                            z = scores[r, j] * scale
                            if cap > 0:
                                z = cap * T.tanh(z / cap)
                            scores[r, j] = T.if_then_else(visible, z, -T.infinity("float32"))
                    # One SIMD reduction owns each row. Max, sum and rescaling must
                    # use that same row when merging successive KV tiles.
                    with T.SimdVF():
                        full = S.pset(32)
                        one = S.pset(32, "PAT_VL1")
                        for r in T.serial(R):
                            lo = S.vld(scores[r, 0])
                            hi = S.vld(scores[r, 64])
                            old = S.vld(maximum[r], "BRC_B32")
                            peak = S.vdupv(S.vcmax(S.vmax(lo, hi, full), full), full)
                            peak = S.vmax(old, peak, full)
                            adjust = S.vexpdif(old, peak, full)
                            p0 = S.vexpdif(lo, peak, full)
                            p1 = S.vexpdif(hi, peak, full)
                            summed = S.vcadd(S.vadd(p0, p1, full), full)
                            total = S.vadd(
                                S.vmul(S.vld(denom[r], "BRC_B32"), adjust, full), summed, one
                            )
                            S.vsts(maximum[r], peak, one, "ONEPT_B32")
                            S.vsts(denom[r], total, one, "ONEPT_B32")
                            S.vsts(alpha[r], adjust, one, "ONEPT_B32")
                            S.vsts(
                                probs[r, 0],
                                S.vcvt(p0, dtype, round="ROUND_R", sat=False, part=0),
                                full,
                                "PK_B32",
                            )
                            S.vsts(
                                probs[r, 64],
                                S.vcvt(p1, dtype, round="ROUND_R", sat=False, part=0),
                                full,
                                "PK_B32",
                            )
                    T.dual_copy(probs, pl)
                    T.copy(V[bh, kv * BN : (kv + 1) * BN, :], vl, transpose=True)
                    T.gemm(pl, vl, pv, transpose_B=True, clear_accum=True)
                    T.dual_copy(pv, current)
                    with T.SimtVF(threads=128):
                        for r, d in T.Parallel(R, D):
                            acc[r, d] = acc[r, d] * alpha[r] + current[r, d]
            with T.SimtVF(threads=128):
                for r in T.Parallel(R):
                    stats[r, 0] = maximum[r]
                    stats[r, 1] = denom[r]
            T.dual_copy(acc, Partial[bh, split, mt * BM : (mt + 1) * BM, :])
            T.dual_copy(stats, Stats[bh, split, mt * BM : (mt + 1) * BM, :])

    return main


@functools.lru_cache(maxsize=128)
def attention_merge(call, cube_cores, splits):
    B, SQ, QR = T.dynamic("batch"), T.dynamic("query_length"), T.dynamic("query_rows")
    H, HK, D = call.heads, call.heads_kv, call.dim
    G, NS = H // HK, splits
    dtype = str(call.dtype).split(".")[-1]
    grid = 2 * cube_cores

    @T.prim_func
    def main(
        Partial: T.Tensor((B * HK, NS, QR, D), "float32"),
        Stats: T.Tensor((B * HK, NS, QR, 2), "float32"),
        Sinks: T.Tensor((H,), "float32"),
        O: T.Tensor((B, SQ, H, D), dtype),
        LSE: T.Tensor((B, H, SQ), "float32") if call.return_lse else None,
    ):
        with T.Kernel(grid) as bx:
            for batch in T.serial(B):
                b = T.Cast("int64", batch)
                for task in T.serial(bx, SQ * H, grid):
                    h = task % H
                    q = task // H
                    bh = b * HK + h // G
                    row = q * G + h % G
                    with T.SimtVF(threads=128):
                        for d in T.Parallel(D):
                            m = T.alloc_var("float32", init=-1e30)
                            for s in T.serial(NS):
                                m = T.max(m, Stats[bh, s, row, 0])
                            if call.has_sinks:
                                m = T.max(m, Sinks[h])
                            numerator = T.alloc_var("float32", init=0)
                            denominator = T.alloc_var("float32", init=0)
                            if call.has_sinks:
                                denominator = T.exp(Sinks[h] - m)
                            for s in T.serial(NS):
                                weight = T.exp(Stats[bh, s, row, 0] - m)
                                numerator += Partial[bh, s, row, d] * weight
                                denominator += Stats[bh, s, row, 1] * weight
                            if call.has_sinks:
                                O[b, q, h, d] = T.if_then_else(
                                    Sinks[h] == T.infinity("float32"), 0, numerator / denominator
                                )
                            elif call.return_lse:
                                O[b, q, h, d] = T.if_then_else(denominator > 0, numerator / denominator, 0)
                                if d == 0:
                                    LSE[b, h, q] = T.if_then_else(
                                        denominator > 0, m + T.log(denominator), -T.infinity("float32")
                                    )
                            else:
                                O[b, q, h, d] = numerator / denominator

    return main


class AscendGQADenseKernel(Kernel, GQADenseFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def refusal(cls, call):
        if call.dtype not in (torch.float16, torch.bfloat16):
            return "requires float16 or bfloat16 output"
        if call.dim not in (64, 128, 256):
            return "requires head dimension 64, 128 or 256"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call):
        cores = device_cube_cores(call)
        variant, options = select_kernel(call, cores)
        identity = call
        if variant == "prefill-preprocess" and call.fuse_rope:
            identity = replace(call, max_position=call.seqlen_kv)
        elif variant == "packed-attention":
            # The packed kernels read sizes at launch. Only the split schedule and
            # the optional contiguous MQA view distinguish builds across requests.
            static = replace(call, batch=0, max_seqlen_q=0, seqlen_kv=0, max_position=0)
            identity = (static, configuration(call, cores)[2], options["kv_view"])
        return (cores, identity), lambda: cls(call, cores)

    def __init__(self, call, cube_cores=None):
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        if cube_cores is None:
            cube_cores = device_cube_cores(call)
        variant, options = select_kernel(call, cube_cores)
        if variant == "empty":
            self.config = dict(variant=variant, cube_cores=cube_cores)
            return
        qr, kr, splits = configuration(call, cube_cores)
        config = dict(variant=variant, cube_cores=cube_cores, kv_splits=splits, **options)
        if variant != "packed-attention":
            config.update(query_rows=qr, kv_rows=kr)
        if variant == "decode-multibuffer":
            config.update(multibuffer=True, balanced_vectors=True, kv_splits=options["splits"])
        self.config = config
        self.fused = None
        self.grid_fused = options.get("grid_fuse", False)
        self.pack = None
        self.kv_view = options.get("kv_view", False)
        if variant == "prefill-preprocess":
            self.fused = tilelang.compile(
                prefill_preprocess(call, **options),
                target="ascend",
                out_idx=-1,
                pass_configs=preprocess_pass_configs(call),
            )
            return
        if variant == "prefill-query-major":
            self.fused = tilelang.compile(
                prefill_query_major(call, **options),
                target="ascend",
                out_idx=[4,5] if call.return_lse else -1,
                pass_configs=pass_configs(),
            )
            return
        if variant == "decode-multibuffer":
            assert options["fuse_output"] or self.grid_fused
            self.fused = tilelang.compile(
                decode_multibuffer(call, **options),
                target="ascend",
                # Allocate scratch tensors with the output, then execute one
                # kernel containing both split attention and normalization.
                out_idx=[3, 4, 5] if self.grid_fused else -1,
                pass_configs=pass_configs(),
            )
            return
        operands = ("q",) if self.kv_view else ("q", "k", "v")
        self.pack = tuple(
            tilelang.compile(pack_operand(call, kind, cube_cores), target="ascend", out_idx=[])
            for kind in operands
        )
        self.partial = tilelang.compile(
            attention_partial(call, splits), target="ascend", out_idx=[5, 6]
        )
        self.merge = tilelang.compile(
            attention_merge(call, cube_cores, splits), target="ascend", out_idx=[]
        )

    def forward(
        self,
        q,
        k,
        v,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        rope_cos=None,
        rope_sin=None,
        sinks=None,
    ):
        self._require_devices(q=q, k=k, v=v)
        call = self.call
        if self.config["cube_cores"] > device_cube_cores(call):
            raise RuntimeError("cached GQA grid exceeds the current Cube limit")
        variant = self.config["variant"]
        if variant == "empty":
            output = torch.empty_like(q, dtype=call.dtype)
            if call.return_lse:
                return output, q.new_empty((q.shape[0], call.heads, q.shape[1]), dtype=torch.float32)
            return output
        if variant == "decode-multibuffer":
            if call.has_sinks:
                return self.fused(q, k, v, sinks)
            result = self.fused(q, k, v)
            return result[-1] if self.grid_fused else result
        if variant == "prefill-preprocess" and call.has_sinks:
            return self.fused(q, k, v, q_scale, k_scale, v_scale, sinks)
        if sinks is None and variant != "prefill-preprocess":
            sinks = torch.empty((call.heads,), dtype=torch.float32, device=q.device)
        if variant == "prefill-query-major":
            return self.fused(q, k, v, sinks)

        batch, sq = q.shape[:2]
        sk = k.shape[1]
        if q_scale is None:
            q_scale = k_scale = v_scale = torch.empty(
                (batch, call.heads_kv), dtype=torch.float32, device=q.device
            )
        dummy_rope = torch.empty((1, 1), dtype=call.dtype, device=q.device)
        if rope_cos is None:
            rope_cos = rope_sin = dummy_rope
        if variant == "prefill-preprocess":
            if call.fuse_rope:
                rope_cos, rope_sin = rope_cos[:sk], rope_sin[:sk]
            return self.fused(q, k, v, q_scale, k_scale, v_scale, rope_cos, rope_sin)

        operands = (q, k, v)
        if call.is_fp8:
            operands = tuple(x.view(torch.uint8) for x in operands)
        group = call.heads // call.heads_kv
        query_rows = (sq * group + 31) // 32 * 32
        groups = batch * call.heads_kv
        query = torch.empty((groups, query_rows, call.dim), dtype=call.dtype, device=q.device)
        self.pack[0](operands[0], q_scale, rope_cos, rope_sin, sk - sq, query)
        if self.kv_view:
            key, value = (x.view(batch, sk, call.dim) for x in (k, v))
        else:
            kv_rows = (sk + 127) // 128 * 128
            key = torch.empty((groups, kv_rows, call.dim), dtype=call.dtype, device=q.device)
            value = torch.empty_like(key)
            self.pack[1](operands[1], k_scale, rope_cos, rope_sin, 0, key)
            self.pack[2](operands[2], v_scale, dummy_rope, dummy_rope, 0, value)
        partial, stats = self.partial(query, key, value, sq, sk)
        output = torch.empty(q.shape, dtype=call.dtype, device=q.device)
        if call.return_lse:
            lse = torch.empty((batch, call.heads, sq), dtype=torch.float32, device=q.device)
            self.merge(partial, stats, sinks, output, lse)
            return output, lse
        self.merge(partial, stats, sinks, output)
        return output


register_kernel_type("GQADenseFwdOp", "ascend", AscendGQADenseKernel)
register_kernel_type("GQADenseWithLSEFwdOp", "ascend", AscendGQADenseKernel)
