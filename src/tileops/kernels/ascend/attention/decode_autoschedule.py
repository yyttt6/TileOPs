"""Compiler-scheduled decode with direct BSND reads and SIMD online softmax.

Retains the validated single-buffer schedule for untuned aligned shapes.
KV partitioning is independent of the scheduling choice in decode_multibuffer.
"""

import functools
import math

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from tileops.kernels.ascend.attention.config import configuration


@functools.lru_cache(maxsize=128)
def decode_autoschedule(call):
    B, H, HK, D = call.batch, call.heads, call.heads_kv, call.dim
    SK, G = call.seqlen_kv, H // HK
    QR, KR, NS = configuration(call)
    assert call.max_seqlen_q == 1
    assert SK % 128 == 0 and D in (64, 128)
    assert not call.is_fp8 and not call.fuse_rope and not call.softcap
    assert call.window_size_left == call.window_size_right == -1
    BM, BN, R = 32, 128, 16
    QT = QR // BM
    CHUNK = math.ceil((KR // BN) / NS)
    dtype = str(call.dtype).split(".")[-1]
    scale = D**-0.5 if call.sm_scale is None else call.sm_scale

    @T.prim_func
    def main(
        Q: T.Tensor((B, 1, H, D), dtype),
        K: T.Tensor((B, SK, HK, D), dtype),
        V: T.Tensor((B, SK, HK, D), dtype),
        Partial: T.Tensor((B * HK, NS, QR, D), "float32"),
        Stats: T.Tensor((B * HK, NS, QR, 2), "float32"),
    ):
        with T.MixedKernel(B * HK * NS * QT) as (bx, sub):
            split, qt, bh = bx % NS, bx // NS % QT, bx // (NS * QT)
            batch, head = bh // HK, bh % HK
            ql = T.alloc_l1((BM, D), dtype)
            query = T.alloc_shared((R, D), dtype)
            kl = T.alloc_l1((BN, D), dtype)
            vl = T.alloc_l1((D, BN), dtype)
            pl = T.alloc_l1((BM, BN), dtype)
            qk = T.alloc_l0c((BM, BN), "float32")
            pv = T.alloc_l0c((BM, D), "float32")
            scores = T.alloc_shared((R, BN), "float32")
            probs = T.alloc_shared((R, BN), dtype)
            current = T.alloc_shared((R, D), "float32")
            acc = T.alloc_shared((R, D), "float32")
            maximum = T.alloc_shared((64,), "float32")
            denom = T.alloc_shared((64,), "float32")
            alpha = T.alloc_shared((64,), "float32")
            stats = T.alloc_shared((R, 2), "float32")
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
                        S.vsts(probs[r, c], S.vcvt(zero, dtype, full), full, "PK_B32")
            if qt * BM + sub * R < G:
                T.copy(
                    Q[
                        batch,
                        0,
                        head * G + qt * BM + sub * R : head * G + T.min(qt * BM + (sub + 1) * R, G),
                        :,
                    ],
                    query[: T.min(R, G - qt * BM - sub * R), :],
                )
            T.dual_copy(query, ql)
            for it in T.serial(CHUNK):
                kv = split * CHUNK + it
                if kv * BN < SK:
                    T.copy(K[batch, kv * BN : (kv + 1) * BN, head, :], kl)
                    T.gemm(ql, kl, qk, transpose_B=True, clear_accum=True)
                    T.dual_copy(qk, scores)
                    with T.SimdVF():
                        full = S.pset(32, "PAT_ALL")
                        one = S.pset(32, "PAT_VL1")
                        for r in range(T.max(0, T.min(R, G - qt * BM - sub * R))):
                            x0 = S.vmuls(S.vld(scores[r, 0], "NORM"), T.float32(scale), full)
                            x1 = S.vmuls(S.vld(scores[r, 64], "NORM"), T.float32(scale), full)
                            rowmax = S.vdupv(S.vcmax(S.vmax(x0, x1, full), full), full)
                            oldmax = S.vld(maximum[r], "BRC_B32")
                            newmax = S.vmax(oldmax, rowmax, full)
                            weight = S.vexpdif(oldmax, newmax, full)
                            e0, e1 = S.vexpdif(x0, newmax, full), S.vexpdif(x1, newmax, full)
                            rowsum = S.vcadd(S.vadd(e0, e1, full), full)
                            oldsum = S.vld(denom[r], "BRC_B32")
                            newsum = S.vadd(S.vmul(oldsum, weight, full), rowsum, full)
                            S.vsts(maximum[r], newmax, one, "ONEPT_B32")
                            S.vsts(denom[r], newsum, one, "ONEPT_B32")
                            S.vsts(alpha[r], weight, one, "ONEPT_B32")
                            S.vsts(probs[r, 0], S.vcvt(e0, dtype, full), full, "PK_B32")
                            S.vsts(probs[r, 64], S.vcvt(e1, dtype, full), full, "PK_B32")
                    T.dual_copy(probs, pl)
                    T.copy(V[batch, kv * BN : (kv + 1) * BN, head, :], vl, transpose=True)
                    T.gemm(pl, vl, pv, transpose_B=True, clear_accum=True)
                    T.dual_copy(pv, current)
                    with T.SimdVF():
                        full = S.pset(32, "PAT_ALL")
                        for r in range(T.max(0, T.min(R, G - qt * BM - sub * R))):
                            weight = S.vld(alpha[r], "BRC_B32")
                            for d in range(0, D, 64):
                                old = S.vld(acc[r, d], "NORM")
                                value = S.vld(current[r, d], "NORM")
                                updated = S.vadd(S.vmul(old, weight, full), value, full)
                                S.vsts(acc[r, d], updated, full, "NORM_B32")
            with T.SimdVF():
                one = S.pset(32, "PAT_VL1")
                for r in range(R):
                    S.vsts(stats[r, 0], S.vld(maximum[r], "BRC_B32"), one, "ONEPT_B32")
                    S.vsts(stats[r, 1], S.vld(denom[r], "BRC_B32"), one, "ONEPT_B32")
            T.dual_copy(acc, Partial[bh, split, qt * BM : (qt + 1) * BM, :])
            T.dual_copy(stats, Stats[bh, split, qt * BM : (qt + 1) * BM, :])

    return main
