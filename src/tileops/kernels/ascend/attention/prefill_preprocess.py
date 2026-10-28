"""Tile-local operand preprocessing, online attention and output in one launch."""

import functools

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout

from tileops._csrc import csrc_include
from tileops.kernels.ascend.attention.common import pass_configs


def preprocess_pass_configs(call):
    configs = pass_configs()
    if call.is_fp8 and not call.fuse_rope:
        configs[tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS] = [
            *csrc_include("ascend_gqa_transpose.h"),
        ]
    return configs


@functools.lru_cache(maxsize=128)
def prefill_preprocess(call, query_block=32, l0_k=64):
    """Fuse the aligned RoPE/FP8 prefill path without GM packed operands or partials."""
    B, H, HK, D = call.batch, call.heads, call.heads_kv, call.dim
    SQ, SK = call.max_seqlen_q, call.seqlen_kv
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
    POS = call.max_position if call.fuse_rope else 1
    scale = D**-0.5 if call.sm_scale is None else call.sm_scale
    assert D == 128 and SK % BN == 0 and SQ > 1
    assert SQ * G % BM == 0 and BM % G == 0 and SK >= SQ
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
                            T.copy(
                                X[batch, start // G + group, head * G : (head + 1) * G, :],
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
                x0 = S.vmuls(S.vld(scores[slot, r, 0], "NORM"), T.float32(scale), full)
                x1 = S.vmuls(S.vld(scores[slot, r, 64], "NORM"), T.float32(scale), full)
                x0 = S.vsel(x0, neg, S.update_mask(T.max(0, T.min(64, visible)), 32))
                x1 = S.vsel(x1, neg, S.update_mask(T.max(0, T.min(64, visible - 64)), 32))
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
            begin = seq * BN + sub * (BN // 2)
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
        with T.Kernel(B * HK * (QR // BM)) as bx:
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
                        T.ascend_cross_core_set_flag(
                            4, "PIPE_FIX", T.if_then_else(slot == 0, 15, 0)
                        )
                        T.ascend_cross_core_set_flag(
                            4, "PIPE_FIX", T.if_then_else(slot == 0, 31, 16)
                        )
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
                with T.SimdVF():
                    full = S.pset(32, "PAT_ALL")
                    zero = S.vdup(T.float32(0), "float32", full)
                    neg = S.vdup(T.float32(-3.402823e38), "float32", full)
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
                        T.ascend_cross_core_wait_flag(
                            4, "PIPE_V", T.if_then_else(previous == 0, 15, 0)
                        )
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
                        T.copy(
                            output[group * G : (group + 1) * G, :],
                            O[
                                batch,
                                (mt * BM + sub * R) // G + group,
                                head * G : (head + 1) * G,
                                :,
                            ],
                        )
                else:
                    for r in T.serial(R):
                        row = mt * BM + sub * R + r
                        if row < SQ * G:
                            T.copy(output[r, :], O[batch, row // G, head * G + row % G, :])

    return pipelined.with_attr("global_symbol", "main")
