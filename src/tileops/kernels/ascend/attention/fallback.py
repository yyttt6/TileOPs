"""Packed dense GQA for unaligned lengths and general attention semantics."""

import functools
import math

import tilelang.ascend.language as T

from tileops.kernels.ascend.attention.config import configuration


@functools.lru_cache(maxsize=128)
def pack_operand(call, kind):
    """Pack BSHD into independent KV groups, applying scale then rounded RoPE."""
    B, H, HK, D = call.batch, call.heads, call.heads_kv, call.dim
    SQ, SK = call.max_seqlen_q, call.seqlen_kv
    G = H // HK
    QR, KR, _ = configuration(call)
    is_q = kind == "q"
    rotate = call.fuse_rope and kind != "v"
    N = QR if is_q else KR
    S = SQ if is_q else SK
    NH = H if is_q else HK
    R = call.rotary_dim if rotate else 2
    P = call.max_position if rotate else 1
    interleaved = call.rope_layout == "interleaved"
    out_type = str(call.dtype).split(".")[-1]
    in_type = "uint8" if call.is_fp8 else out_type
    total = B * HK * N * D
    block = 256
    grid = min(64, math.ceil(total / block))

    @T.prim_func
    def main(
        X: T.Tensor((B, S, NH, D), in_type),
        Scale: T.Tensor((B, HK), "float32"),
        Cos: T.Tensor((P, R // 2), out_type),
        Sin: T.Tensor((P, R // 2), out_type),
        Y: T.Tensor((B * HK, N, D), out_type),
    ):
        with T.Kernel(grid) as bx:
            for tile in T.serial(T.ceildiv(total, grid * block)):
                with T.SimtVF(threads=128):
                    for lane in T.Parallel(block):
                        idx = (tile * grid + bx) * block + lane
                        if idx < total:
                            d = idx % D
                            row = idx // D % N
                            bh = idx // (D * N)
                            b = bh // HK
                            h = bh % HK
                            pos = row // G if is_q else row
                            ih = h * G + row % G if is_q else h
                            value = T.alloc_var("float32", init=0)
                            if pos < S:
                                value = T.Cast("float32", X[b, pos, ih, d])
                                if call.is_fp8:
                                    # Decode E4M3FN bytes explicitly: the pinned CANN
                                    # scalar fp8->float conversion produced zeros on 950DT.
                                    raw = T.Cast("int32", X[b, pos, ih, d])
                                    exponent = (raw >> 3) & 15
                                    value = T.if_then_else(
                                        exponent == 0,
                                        (raw & 7) * (2.0**-9),
                                        ((raw & 7) + 8) * T.exp2(T.Cast("float32", exponent - 10)),
                                    )
                                    value = T.if_then_else((raw & 128) != 0, -value, value)
                                    value = T.if_then_else(
                                        (raw & 127) == 127,
                                        T.reinterpret(T.uint32(0x7FC00000), "float32"),
                                        value,
                                    )
                                    value = T.Cast("float32", T.Cast(out_type, value * Scale[b, h]))
                                if rotate:  # noqa: SIM102 - eliminate the optional table access at specialization time
                                    if d < R:
                                        pair = d // 2 if interleaved else d % (R // 2)
                                        first = d % 2 == 0 if interleaved else d < R // 2
                                        other = (
                                            T.if_then_else(first, d + 1, d - 1)
                                            if interleaved
                                            else (d + R // 2) % R
                                        )
                                        partner = T.Cast("float32", X[b, pos, ih, other])
                                        if call.is_fp8:
                                            other_raw = T.Cast("int32", X[b, pos, ih, other])
                                            other_exponent = (other_raw >> 3) & 15
                                            partner = T.if_then_else(
                                                other_exponent == 0,
                                                (other_raw & 7) * (2.0**-9),
                                                ((other_raw & 7) + 8)
                                                * T.exp2(T.Cast("float32", other_exponent - 10)),
                                            )
                                            partner = T.if_then_else(
                                                (other_raw & 128) != 0, -partner, partner
                                            )
                                            partner = T.if_then_else(
                                                (other_raw & 127) == 127,
                                                T.reinterpret(T.uint32(0x7FC00000), "float32"),
                                                partner,
                                            )
                                            partner = T.Cast(
                                                "float32", T.Cast(out_type, partner * Scale[b, h])
                                            )
                                        position = pos + SK - SQ if is_q else pos
                                        value = value * T.Cast(
                                            "float32", Cos[position, pair]
                                        ) + T.if_then_else(first, -partner, partner) * T.Cast(
                                            "float32", Sin[position, pair]
                                        )
                            Y[bh, row, d] = value

    return main


@functools.lru_cache(maxsize=128)
def attention_partial(call):
    """Cube QK/PV with per-row FP32 online softmax; no quadratic GM scores."""
    B, HK, D = call.batch, call.heads_kv, call.dim
    G = call.heads // HK
    SQ, SK = call.max_seqlen_q, call.seqlen_kv
    QR, KR, NS = configuration(call)
    BM, BN, R = 32, 128, 16
    MT = QR // BM
    CHUNK = math.ceil((KR // BN) / NS)
    grid = B * HK * MT * NS
    dtype = str(call.dtype).split(".")[-1]
    scale = D**-0.5 if call.sm_scale is None else call.sm_scale
    cap = call.softcap
    causal = call.is_causal
    wl, wr = call.window_size_left, call.window_size_right

    @T.prim_func
    def main(
        Q: T.Tensor((B * HK, QR, D), dtype),
        K: T.Tensor((B * HK, KR, D), dtype),
        V: T.Tensor((B * HK, KR, D), dtype),
        Partial: T.Tensor((B * HK, NS, QR, D), "float32"),
        Stats: T.Tensor((B * HK, NS, QR, 2), "float32"),
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
                        logits = T.alloc_fragment((R, BN), "float32")
                        mx = T.alloc_reducer((R,), "float32", op="max")
                        T.reducer_init(mx)
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
                            logits[r, j] = T.if_then_else(visible, z, -T.infinity("float32"))
                            T.reducer_update(mx[r], logits[r, j])
                        newmax = T.alloc_fragment((R,), "float32")
                        T.finalize_reducer(mx, newmax)
                        for r in T.Parallel(R):
                            newmax[r] = T.max(maximum[r], newmax[r])
                            alpha[r] = T.exp(maximum[r] - newmax[r])
                            maximum[r] = newmax[r]
                        sm = T.alloc_reducer((R,), "float32", op="sum")
                        T.reducer_init(sm)
                        for r, j in T.Parallel(R, BN):
                            logits[r, j] = T.exp(logits[r, j] - newmax[r])
                            probs[r, j] = logits[r, j]
                            T.reducer_update(sm[r], logits[r, j])
                        sums = T.alloc_fragment((R,), "float32")
                        T.finalize_reducer(sm, sums)
                        for r in T.Parallel(R):
                            denom[r] = denom[r] * alpha[r] + sums[r]
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
def attention_merge(call, splits=None):
    B, H, HK, D = call.batch, call.heads, call.heads_kv, call.dim
    SQ = call.max_seqlen_q
    G = H // HK
    QR, _, NS = configuration(call)
    NS = NS if splits is None else splits
    assert NS > 0
    dtype = str(call.dtype).split(".")[-1]
    total = B * SQ * H * D
    grid = min(64, math.ceil(total / 256))

    @T.prim_func
    def main(
        Partial: T.Tensor((B * HK, NS, QR, D), "float32"),
        Stats: T.Tensor((B * HK, NS, QR, 2), "float32"),
        O: T.Tensor((B, SQ, H, D), dtype),
    ):
        with T.Kernel(grid) as bx:
            for tile in T.serial(T.ceildiv(total, grid * 256)):
                with T.SimtVF(threads=128):
                    for lane in T.Parallel(256):
                        idx = (tile * grid + bx) * 256 + lane
                        if idx < total:
                            d = idx % D
                            h = idx // D % H
                            q = idx // (D * H) % SQ
                            b = idx // (D * H * SQ)
                            bh = b * HK + h // G
                            row = q * G + h % G
                            m = T.alloc_var("float32", init=-1e30)
                            for s in T.serial(NS):
                                m = T.max(m, Stats[bh, s, row, 0])
                            numerator = T.alloc_var("float32", init=0)
                            denominator = T.alloc_var("float32", init=0)
                            for s in T.serial(NS):
                                weight = T.exp(Stats[bh, s, row, 0] - m)
                                numerator += Partial[bh, s, row, d] * weight
                                denominator += Stats[bh, s, row, 1] * weight
                            O[b, q, h, d] = numerator / denominator

    return main
