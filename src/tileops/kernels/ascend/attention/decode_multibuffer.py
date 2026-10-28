"""Explicit multi-buffer decode pipeline, adapted from the FIA prefill schedule.

K/V: three L1 slots; P and shared L0A/B: two slots; score: three UB
slots; PV: two UB slots; alpha: three slots for delayed output rescaling.
Query and online-softmax state stay live for one complete split.
"""

import functools
import math

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout

from tileops._csrc import csrc_include
from tileops.kernels.ascend.attention.common import decode_merge_body, pass_configs
from tileops.kernels.ascend.attention.config import configuration


def grid_pass_configs():
    """Content-address the native barrier header in both compiler caches."""
    return {
        **pass_configs(),
        tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: [
            *csrc_include("ascend_gqa_grid_sync.h"),
            # Bisheng's device frontend rejects libstdc++'s constrained pair
            # assignment overloads. Use its legacy library branch, retaining
            # C++20 for TileLang's std::bit_cast constants.
            "-U__cpp_concepts",
        ],
    }


@functools.lru_cache(maxsize=128)
def decode_multibuffer(
    call,
    kv_block=128,
    fuse_output=False,
    query_block=32,
    grid_fuse=False,
    cores=36,
    splits=None,
    l0_k=64,
    output_block=None,
    kv_head_first=False,
    merge_rows=1,
    persistent=False,
    kv_phase=0,
):
    """Build decode with local or cross-block output normalization.

    ``fuse_output`` normalizes a single KV split within its tile; ``grid_fuse``
    normalizes multiple splits after a global barrier in the same launch.
    """
    B, H, HK, D = call.batch, call.heads, call.heads_kv, call.dim
    SK, G = call.seqlen_kv, H // HK
    QR, _, NS = configuration(call)
    NS = NS if splits is None else splits
    assert NS > 0 and cores > 0
    assert not grid_fuse or not fuse_output
    assert not fuse_output or NS == 1
    assert call.max_seqlen_q == 1
    assert SK % 128 == 0 and D in (64, 128, 256)
    assert not call.is_fp8 and not call.fuse_rope
    assert call.softcap >= 0
    assert call.window_size_left == call.window_size_right == -1
    assert kv_block in (64, 128, 256) and SK % kv_block == 0
    assert query_block in (32, 64) or query_block == 128 and D == 64 and kv_block == 128
    # The logical partial buffers remain padded to 32 rows. For wider tiles, write
    # only valid heads so a final half tile cannot overrun that padding.
    BM, BN, R = query_block, kv_block, query_block // 2
    VN = D if output_block is None else output_block
    assert VN in (64, 128, 256) and D % VN == 0
    # Bound the resident Q, three K/V versions and two probability versions.
    assert (BM * D + 6 * BN * D + 2 * BM * BN) * 2 <= 512 * 1024
    QT = math.ceil(G / BM)
    KV_TILES = SK // BN
    assert NS <= KV_TILES, "all split partitions must be nonempty"
    dtype = str(call.dtype).split(".")[-1]
    scale = D**-0.5 if call.sm_scale is None else call.sm_scale

    BK = l0_k
    assert BK in (64, 128) and D % BK == 0 and BN % BK == 0
    assert 2 * max(BN, VN) * BK * 2 <= 65536

    def vector_rows(qt):
        # Keep the physical 16-row halves required by Cube/NZ, but distribute
        # useful heads over both V cores even when G is smaller than 16.
        return T.ceildiv(T.min(BM, G - qt * BM), 2)

    def cap_scores(x, mask):
        if not call.softcap:
            return x
        # tanh(z) = 2 / (1 + exp(-2*z)) - 1; no inf/inf at saturation.
        denom = S.vadds(
            S.vexp(S.vmuls(x, T.float32(-2.0 / call.softcap), mask), mask), T.float32(1), mask
        )
        value = S.vadds(
            S.vdiv(S.vdup(T.float32(2), "float32", mask), denom, mask), T.float32(-1), mask
        )
        return S.vmuls(value, T.float32(call.softcap), mask)

    @T.macro
    def softmax(scores, probs, maximum, denom, alpha, seq, qt, sub):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            one = S.pset(32, "PAT_VL1")
            for r in range(T.max(0, T.min(vector_rows(qt), G - qt * BM - sub * vector_rows(qt)))):
                x0 = cap_scores(
                    S.vmuls(S.vld(scores[seq % 3, r, 0], "NORM"), T.float32(scale), full), full
                )
                if BN == 64:
                    max_values = x0
                else:
                    x1 = cap_scores(
                        S.vmuls(S.vld(scores[seq % 3, r, 64], "NORM"), T.float32(scale), full), full
                    )
                    if BN == 128:
                        max_values = S.vmax(x0, x1, full)
                    else:
                        x2 = cap_scores(
                            S.vmuls(S.vld(scores[seq % 3, r, 128], "NORM"), T.float32(scale), full),
                            full,
                        )
                        x3 = cap_scores(
                            S.vmuls(S.vld(scores[seq % 3, r, 192], "NORM"), T.float32(scale), full),
                            full,
                        )
                        max_values = S.vmax(S.vmax(x0, x1, full), S.vmax(x2, x3, full), full)
                rowmax = S.vdupv(S.vcmax(max_values, full), full)
                oldmax = S.vld(maximum[r], "BRC_B32")
                newmax = S.vmax(oldmax, rowmax, full)
                weight = S.vexpdif(oldmax, newmax, full)
                e0 = S.vexpdif(x0, newmax, full)
                S.vsts(probs[seq % 2, r, 0], S.vcvt(e0, dtype, full), full, "PK_B32")
                if BN == 64:
                    sum_values = e0
                else:
                    e1 = S.vexpdif(x1, newmax, full)
                    S.vsts(probs[seq % 2, r, 64], S.vcvt(e1, dtype, full), full, "PK_B32")
                    if BN == 128:
                        sum_values = S.vadd(e0, e1, full)
                    else:
                        e2, e3 = (
                            S.vexpdif(x2, newmax, full),
                            S.vexpdif(x3, newmax, full),
                        )
                        S.vsts(probs[seq % 2, r, 128], S.vcvt(e2, dtype, full), full, "PK_B32")
                        S.vsts(probs[seq % 2, r, 192], S.vcvt(e3, dtype, full), full, "PK_B32")
                        sum_values = S.vadd(S.vadd(e0, e1, full), S.vadd(e2, e3, full), full)
                rowsum = S.vcadd(sum_values, full)
                oldsum = S.vld(denom[r], "BRC_B32")
                newsum = S.vadd(S.vmul(oldsum, weight, full), rowsum, full)
                S.vsts(maximum[r], newmax, one, "ONEPT_B32")
                S.vsts(denom[r], newsum, one, "ONEPT_B32")
                S.vsts(alpha[seq % 3, r], weight, one, "ONEPT_B32")

    @T.macro
    def update(current, acc, alpha, seq, qt, sub):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            for r in range(T.max(0, T.min(vector_rows(qt), G - qt * BM - sub * vector_rows(qt)))):
                weight = S.vld(alpha[seq % 3, r], "BRC_B32")
                for d in range(0, D, 64):
                    old = S.vld(acc[r, d], "NORM")
                    value = S.vld(
                        current[seq % 2, r, d] if VN == D else current[seq % 2, d // VN, r, d % VN],
                        "NORM",
                    )
                    updated = S.vadd(S.vmul(old, weight, full), value, full)
                    S.vsts(acc[r, d], updated, full, "NORM_B32")

    @T.macro
    def tile(Q, K, V, Partial, Stats, O, bx):
        if kv_head_first:
            # Neighboring blocks visit adjacent BSND KV heads at the same split.
            bh = bx // (NS * QT * HK) * HK + bx % HK
            qt, split = bx // HK % QT, bx // (HK * QT) % NS
        else:
            split, qt, bh = bx % NS, bx // NS % QT, bx // (NS * QT)
        batch, head = bh // HK, bh % HK
        # Floor boundaries cover every KV tile exactly once and differ by
        # at most one tile, even when the split count does not divide KV.
        first_kv = split * KV_TILES // NS
        count = (split + 1) * KV_TILES // NS - first_kv
        # Q is invariant within a split; it needs no rotating version.
        ql = T.alloc_l1((BM, D), dtype)
        query = T.alloc_shared((R, D), dtype)
        kl = T.alloc_l1((3, BN, D), dtype)
        vl = T.alloc_l1((3, D, BN), dtype)
        pl = T.alloc_l1((2, BM, BN), dtype)
        qk = T.alloc_l0c((2, BM, BN), "float32")
        pv = T.alloc_l0c((2, BM, D) if VN == D else (2, D // VN, BM, VN), "float32")
        scores = T.alloc_shared((3, R, BN), "float32")
        probs = T.alloc_shared((2, R, BN), dtype)
        # Make ND->NZ conversion storage explicit: implicit scratch would
        # be single-buffered and escape the hand-written event schedule.
        query_nz = T.alloc_shared((R + 1, D), dtype)
        prob_nz = T.alloc_shared((2, R + 1, BN), dtype)
        T.annotate_layout(
            {
                query_nz: make_ascend_compact_nz_layout(query_nz),
                prob_nz: make_ascend_compact_nz_layout(prob_nz),
            }
        )
        current = T.alloc_shared((2, R, D) if VN == D else (2, D // VN, R, VN), "float32")
        acc = T.alloc_shared((R, D), "float32")
        maximum = T.alloc_shared((64,), "float32")
        denom = T.alloc_shared((64,), "float32")
        alpha = T.alloc_shared((3, 64), "float32")
        stats = T.alloc_shared((R, 2), "float32")
        output = T.alloc_shared((R, D), dtype)
        a_l0 = T.alloc_l0a((2, BM, BK), dtype)
        b_l0 = T.alloc_l0b((2, max(BN, VN), BK), dtype)

        with T.Cube():
            for e in T.serial(6):
                T.ascend_set_flag("MTE1_MTE2", e)
            for e in T.serial(4):
                T.ascend_set_flag("FIX_M", e)
            for e in T.serial(2):
                T.ascend_set_flag("M_MTE1", e)
                T.ascend_cross_core_set_flag(4, "PIPE_MTE1", 5 + e * 16)
                T.ascend_cross_core_set_flag(4, "PIPE_MTE1", 12 + e * 16)
            T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", 14)
            T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", 30)
            l0_sequence = T.alloc_var("int32", init=0)
            # t: K load; t-1: QK; t-2: V load and QK fix;
            # t-4: PV; t-5: PV fix. The epilogue drains every token.
            for step in T.serial(count + 5):
                if step < count:
                    # Rotate complete splits to stagger KV requests across heads.
                    kv = first_kv + ((step + bh * kv_phase) % count if kv_phase else step)
                    T.assume(0 <= kv < SK // BN)
                    T.ascend_wait_flag("MTE1_MTE2", 3 + step % 3)
                    T.copy(K[batch, kv * BN : (kv + 1) * BN, head, :], kl[step % 3, :, :])
                    T.ascend_set_flag("MTE2_MTE1", 3 + step % 3)
                if step >= 1 and step < count + 1:
                    g = step - 1
                    T.ascend_wait_flag("FIX_M", 2 + g % 2)
                    T.ascend_wait_flag("MTE2_MTE1", 3 + g % 3)
                    for sk in T.serial(D // BK):
                        ls = T.floormod(l0_sequence + sk, 2)
                        T.assume(0 <= ls < 2)
                        T.ascend_wait_flag("M_MTE1", ls)
                        T.copy(ql[:, sk * BK : (sk + 1) * BK], a_l0[ls, :, :])
                        T.copy(kl[g % 3, :, sk * BK : (sk + 1) * BK], b_l0[ls, :BN, :])
                        T.ascend_set_flag("MTE1_M", ls)
                        T.ascend_wait_flag("MTE1_M", ls)
                        T.gemm(
                            a_l0[ls, :, :],
                            b_l0[ls, :BN, :],
                            qk[g % 2, :, :],
                            transpose_B=True,
                            clear_accum=sk == 0,
                        )
                        T.ascend_set_flag("M_MTE1", ls)
                    l0_sequence += D // BK
                    T.ascend_set_flag("M_FIX", 2 + g % 2)
                    T.ascend_set_flag("MTE1_MTE2", 3 + g % 3)
                if step >= 2 and step < count + 2:
                    g = step - 2
                    kv = first_kv + ((g + bh * kv_phase) % count if kv_phase else g)
                    T.assume(0 <= kv < SK // BN)
                    T.ascend_wait_flag("MTE1_MTE2", g % 3)
                    T.copy(
                        V[batch, kv * BN : (kv + 1) * BN, head, :],
                        vl[g % 3, :, :],
                        transpose=True,
                    )
                    T.ascend_set_flag("MTE2_MTE1", g % 3)
                if step >= 4 and step < count + 4:
                    g = step - 4
                    ready = T.if_then_else(g % 2 == 0, 11, 13)
                    free = T.if_then_else(g % 2 == 0, 5, 12)
                    T.ascend_wait_flag("FIX_M", g % 2)
                    T.ascend_wait_flag("MTE2_MTE1", g % 3)
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", ready)
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", ready + 16)
                    # Split PV columns so D256 can use BK128 with two L0B slots.
                    for dn in T.serial(D // VN):
                        for sk in T.serial(BN // BK):
                            ls = T.floormod(l0_sequence + sk, 2)
                            T.assume(0 <= ls < 2)
                            T.ascend_wait_flag("M_MTE1", ls)
                            T.copy(pl[g % 2, :, sk * BK : (sk + 1) * BK], a_l0[ls, :, :])
                            T.copy(
                                vl[g % 3, dn * VN : (dn + 1) * VN, sk * BK : (sk + 1) * BK],
                                b_l0[ls, :VN, :],
                            )
                            T.ascend_set_flag("MTE1_M", ls)
                            T.ascend_wait_flag("MTE1_M", ls)
                            T.gemm(
                                a_l0[ls, :, :],
                                b_l0[ls, :VN, :],
                                pv[g % 2, :, :] if VN == D else pv[g % 2, dn, :, :],
                                transpose_B=True,
                                clear_accum=sk == 0,
                            )
                            T.ascend_set_flag("M_MTE1", ls)
                        l0_sequence += BN // BK
                    T.ascend_set_flag("M_FIX", g % 2)
                    T.ascend_cross_core_set_flag(4, "PIPE_MTE1", free)
                    T.ascend_cross_core_set_flag(4, "PIPE_MTE1", free + 16)
                    T.ascend_set_flag("MTE1_MTE2", g % 3)
                if step >= 5:
                    g = step - 5
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 3 + g % 2)
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 19 + g % 2)
                    T.ascend_wait_flag("M_FIX", g % 2)
                    if VN == D:
                        T.dual_copy(pv[g % 2, :, :], current[g % 2, :, :])
                    else:
                        for dn in T.serial(D // VN):
                            T.dual_copy(pv[g % 2, dn, :, :], current[g % 2, dn, :, :])
                    T.ascend_cross_core_set_flag(4, "PIPE_FIX", 9 + g % 2)
                    T.ascend_cross_core_set_flag(4, "PIPE_FIX", 25 + g % 2)
                    T.ascend_set_flag("FIX_M", g % 2)
                if step >= 2 and step < count + 2:
                    g = step - 2
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", g % 3)
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 16 + g % 3)
                    T.ascend_wait_flag("M_FIX", 2 + g % 2)
                    T.dual_copy(qk[g % 2, :, :], scores[g % 3, :, :])
                    T.ascend_cross_core_set_flag(4, "PIPE_FIX", 6 + g % 3)
                    T.ascend_cross_core_set_flag(4, "PIPE_FIX", 22 + g % 3)
                    T.ascend_set_flag("FIX_M", 2 + g % 2)
            for e in T.serial(6):
                T.ascend_wait_flag("MTE1_MTE2", e)
            for e in T.serial(4):
                T.ascend_wait_flag("FIX_M", e)
            for lane in T.serial(2):
                T.ascend_wait_flag("M_MTE1", lane)
                for e in T.serial(5):
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", e + lane * 16)

        with T.Vector() as sub:
            with T.SimdVF():
                full = S.pset(32, "PAT_ALL")
                zero = S.vdup(T.float32(0), "float32", full)
                neg = S.vdup(T.float32(-3.402823e38), "float32", full)
                S.vsts(maximum[0], neg, full, "NORM_B32")
                S.vsts(denom[0], zero, full, "NORM_B32")
                for r in range(R):
                    for d in range(0, D, 64):
                        S.vsts(acc[r, d], zero, full, "NORM_B32")
                        S.vsts(query[r, d], S.vcvt(zero, dtype, full), full, "PK_B32")
                    for c in range(0, BN, 64):
                        for slot in range(2):
                            S.vsts(probs[slot, r, c], S.vcvt(zero, dtype, full), full, "PK_B32")
            T.ascend_set_flag("V_MTE2", 0)
            T.ascend_wait_flag("V_MTE2", 0)
            if qt * BM + sub * vector_rows(qt) < G:
                T.copy(
                    Q[
                        batch,
                        0,
                        head * G + qt * BM + sub * vector_rows(qt) : head * G
                        + T.min(qt * BM + (sub + 1) * vector_rows(qt), G),
                        :,
                    ],
                    query[: T.min(vector_rows(qt), G - qt * BM - sub * vector_rows(qt)), :],
                )
            T.ascend_set_flag("MTE2_V", 0)
            T.ascend_wait_flag("MTE2_V", 0)
            T.copy(query, query_nz[:R, :])
            T.ascend_set_flag("V_MTE3", 2)
            T.ascend_wait_flag("V_MTE3", 2)
            T.dual_copy(query_nz[:R, :], ql)
            T.ascend_cross_core_set_flag(4, "PIPE_MTE3", 14)
            for e in T.serial(5):
                T.ascend_cross_core_set_flag(4, "PIPE_V", e)
            for e in T.serial(2):
                T.ascend_set_flag("MTE3_V", e)
            for step in T.serial(count + 3):
                # Consume alpha[t-3] before softmax overwrites its slot.
                if step >= 3:
                    g = step - 3
                    T.ascend_cross_core_wait_flag(4, "PIPE_V", 9 + g % 2)
                    update(current, acc, alpha, g, qt, sub)
                    T.ascend_cross_core_set_flag(4, "PIPE_V", 3 + g % 2)
                if step < count:
                    T.ascend_wait_flag("MTE3_V", step % 2)
                    T.ascend_cross_core_wait_flag(4, "PIPE_V", 6 + step % 3)
                    softmax(scores, probs, maximum, denom, alpha, step, qt, sub)
                    T.copy(probs[step % 2, :, :], prob_nz[step % 2, :R, :])
                    T.ascend_set_flag("V_MTE3", step % 2)
                    T.ascend_cross_core_set_flag(4, "PIPE_V", step % 3)
                    T.ascend_cross_core_wait_flag(
                        4, "PIPE_MTE3", T.if_then_else(step % 2 == 0, 5, 12)
                    )
                    T.ascend_wait_flag("V_MTE3", step % 2)
                    T.dual_copy(prob_nz[step % 2, :R, :], pl[step % 2, :, :])
                    T.ascend_cross_core_set_flag(
                        4, "PIPE_MTE3", T.if_then_else(step % 2 == 0, 11, 13)
                    )
                    T.ascend_set_flag("MTE3_V", step % 2)
            for e in T.serial(2):
                T.ascend_wait_flag("MTE3_V", e)
            T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", 5)
            T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", 12)
            if fuse_output:
                with T.SimdVF():
                    full = S.pset(32, "PAT_ALL")
                    for r in range(
                        T.max(0, T.min(vector_rows(qt), G - qt * BM - sub * vector_rows(qt)))
                    ):
                        divisor = S.vld(denom[r], "BRC_B32")
                        for d in range(0, D, 64):
                            value = S.vdiv(S.vld(acc[r, d], "NORM"), divisor, full)
                            S.vsts(output[r, d], S.vcvt(value, dtype, full), full, "PK_B32")
                T.ascend_set_flag("V_MTE3", 2)
                T.ascend_wait_flag("V_MTE3", 2)
                if qt * BM + sub * vector_rows(qt) < G:
                    T.copy(
                        output[: T.min(vector_rows(qt), G - qt * BM - sub * vector_rows(qt)), :],
                        O[
                            batch,
                            0,
                            head * G + qt * BM + sub * vector_rows(qt) : head * G
                            + T.min(qt * BM + (sub + 1) * vector_rows(qt), G),
                            :,
                        ],
                    )
            else:
                with T.SimdVF():
                    one = S.pset(32, "PAT_VL1")
                    for r in range(R):
                        S.vsts(stats[r, 0], S.vld(maximum[r], "BRC_B32"), one, "ONEPT_B32")
                        S.vsts(stats[r, 1], S.vld(denom[r], "BRC_B32"), one, "ONEPT_B32")
                T.ascend_set_flag("V_MTE3", 2)
                T.ascend_wait_flag("V_MTE3", 2)
                # Restore contiguous logical head order for the existing
                # merge kernel. Copy only valid rows; full physical halves
                # would overlap or leave the second V's heads in padding.
                start = qt * BM + sub * vector_rows(qt)
                valid = T.max(0, T.min(vector_rows(qt), G - start))
                if valid > 0:
                    T.copy(acc[:valid, :], Partial[bh, split, start : start + valid, :])
                    T.copy(stats[:valid, :], Stats[bh, split, start : start + valid, :])
            T.ascend_set_flag("MTE3_V", 2)
            T.ascend_wait_flag("MTE3_V", 2)

    merge = decode_merge_body(call, NS, merge_rows)

    @T.macro
    def body(Q, K, V, Partial, Stats, O):
        with T.Kernel(cores if grid_fuse or persistent else B * HK * NS * QT) as bx:
            if grid_fuse:
                for work in T.serial(bx, B * HK * NS * QT, cores):
                    tile(Q, K, V, Partial, Stats, O, work)
                # Every AIC and both AIVs participate, including idle blocks.
                # The tile epilogue drains all local and cross-core event tokens.
                T.call_extern("handle", "tileops::gqa_grid_sync")
                with T.Vector() as sub:
                    merge(Partial, Stats, O, bx * 2 + sub, cores * 2)
            elif persistent:
                for work in T.serial(bx, B * HK * NS * QT, cores):
                    tile(Q, K, V, Partial, Stats, O, work)
            else:
                tile(Q, K, V, Partial, Stats, O, bx)

    if grid_fuse:

        @T.prim_func
        def main(
            Q: T.Tensor((B, 1, H, D), dtype),
            K: T.Tensor((B, SK, HK, D), dtype),
            V: T.Tensor((B, SK, HK, D), dtype),
            Partial: T.Tensor((B * HK, NS, QR, D), "float32"),
            Stats: T.Tensor((B * HK, NS, QR, 2), "float32"),
            O: T.Tensor((B, 1, H, D), dtype),
        ):
            T.func_attr({"ascend.batch_schedule": True})
            body(Q, K, V, Partial, Stats, O)
    elif fuse_output:

        @T.prim_func
        def main(
            Q: T.Tensor((B, 1, H, D), dtype),
            K: T.Tensor((B, SK, HK, D), dtype),
            V: T.Tensor((B, SK, HK, D), dtype),
            O: T.Tensor((B, 1, H, D), dtype),
        ):
            body(Q, K, V, O, O, O)
    else:

        @T.prim_func
        def main(
            Q: T.Tensor((B, 1, H, D), dtype),
            K: T.Tensor((B, SK, HK, D), dtype),
            V: T.Tensor((B, SK, HK, D), dtype),
            Partial: T.Tensor((B * HK, NS, QR, D), "float32"),
            Stats: T.Tensor((B * HK, NS, QR, 2), "float32"),
        ):
            body(Q, K, V, Partial, Stats, Partial)

    return main
