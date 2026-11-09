"""KDA chunk-local triangular solve and mixed Cube/Vector state scan."""

import functools

import tilelang
import tilelang.ascend.language as T

from tileops._kernels.ascend.linear_attention.kda import _pick
from tileops._kernels.ascend.linear_attention.kda_gate import log_decay


def _geometry(B, S, C, offsets):
    starts = offsets[:-1] if offsets else (0,) * B
    ends = offsets[1:] if offsets else (S,) * B
    counts = tuple((e - s + C - 1) // C for s, e in zip(starts, ends, strict=True))
    cumulative = [0]
    for n in counts:
        cumulative.append(cumulative[-1] + n)
    return starts, ends, counts, tuple(cumulative)


def _sequence(chunk, cumulative):
    result = T.int32(len(cumulative) - 2)
    for i in range(len(cumulative) - 3, -1, -1):
        result = T.if_then_else(chunk < cumulative[i + 1], T.int32(i), result)
    return result


# Each chunk forms the causal Gram matrix L in FP32 and solves
# W = (I + L)^-1 (beta * k * exp(prefix)), U = (I + L)^-1 (beta * v).
# Causal decay gaps are exponentiated directly, without reciprocal decay factors.
@functools.lru_cache(maxsize=64)
def prepare(
    B,
    S,
    H,
    HV,
    K,
    V,
    dtype,
    scale,
    norm,
    raw,
    beta_sigmoid,
    negative,
    lower_bound,
    has_bias,
    offsets=(),
    C=32,
):
    starts, ends, counts, cumulative = _geometry(B, S, C, offsets)
    N = (S + C - 1) // C
    J = cumulative[-1] * HV
    grid = min(64, J)
    packed = bool(offsets)

    @T.prim_func
    def main(
        Q: T.Tensor((B, S, H, K), dtype),
        KK: T.Tensor((B, S, H, K), dtype),
        VV: T.Tensor((B, S, HV, V), dtype),
        G: T.Tensor((B, S, HV, K), dtype),
        Beta: T.Tensor((B, S, HV), dtype),
        ALog: T.Tensor((HV,), "float32"),
        Bias: T.Tensor((HV * K,), "float32"),
        W: T.Tensor((J, C, K), dtype),
        U: T.Tensor((J, C, V), dtype),
        QE: T.Tensor((J, C, K), dtype),
        KE: T.Tensor((J, C, K), dtype),
        Att: T.Tensor((J, C, C), dtype),
        D: T.Tensor((J, K), "float32"),
    ):
        with T.Kernel(grid) as bx:
            seq = T.alloc_var("int32")
            n = T.alloc_var("int32")
            first = T.alloc_var("int32")
            last = T.alloc_var("int32")
            q = T.alloc_shared((C, K), "float32")
            k = T.alloc_shared((C, K), "float32")
            gp = T.alloc_shared((C, K), "float32")
            v = T.alloc_shared((C, V), "float32")
            beta = T.alloc_shared((C,), "float32")
            inv = T.alloc_shared((C, 2), "float32")
            gram = T.alloc_shared((C, C), "float32")
            att = T.alloc_shared((C, C), "float32")
            w = T.alloc_shared((C, K), "float32")
            u = T.alloc_shared((C, V), "float32")
            for step in T.serial((J + grid - 1) // grid):
                job = step * grid + bx
                if job < J:
                    chunk = job // HV
                    seq = _sequence(chunk, cumulative) if packed else chunk // N
                    b = 0 if packed else seq
                    n = chunk - _pick(seq, cumulative[:-1]) if packed else chunk % N
                    first = _pick(seq, starts) if packed else 0
                    last = _pick(seq, ends) if packed else S
                    h = job % HV
                    qh = h // (HV // H)
                    with T.SimtVF(threads=256):
                        for t, i in T.Parallel(C, K):
                            q[t, i] = 0.0
                            k[t, i] = 0.0
                            gp[t, i] = 0.0
                            if first + n * C + t < last:
                                q[t, i] = Q[b, first + n * C + t, qh, i]
                                k[t, i] = KK[b, first + n * C + t, qh, i]
                                gate = T.cast(G[b, first + n * C + t, h, i], "float32")
                                if raw:
                                    gate = log_decay(
                                        gate,
                                        ALog[h],
                                        Bias[h * K + i] if has_bias else T.float32(0),
                                        lower_bound,
                                    )
                                gp[t, i] = gate
                        for t, i in T.Parallel(C, V):
                            v[t, i] = T.if_then_else(
                                first + n * C + t < last,
                                T.cast(VV[b, first + n * C + t, h, i], "float32"),
                                T.float32(0),
                            )
                        for t in T.Parallel(C):
                            beta[t] = 0.0
                            if first + n * C + t < last:
                                bb = T.cast(Beta[b, first + n * C + t, h], "float32")
                                if beta_sigmoid:
                                    bb = (2.0 if negative else 1.0) / (1.0 + T.exp(-bb))
                                beta[t] = bb
                    if norm:
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            one = T.simd.pset(32, "PAT_VL1")
                            for t in T.serial(C):
                                sq = T.simd.alloc_local(1, "float32")
                                sk = T.simd.alloc_local(1, "float32")
                                sq[0] = T.simd.vdup(T.float32(0), "float32", full)
                                sk[0] = T.simd.vdup(T.float32(0), "float32", full)
                                for part in T.serial(K // 64):
                                    qv = T.simd.vld(q[t, part * 64])
                                    kv = T.simd.vld(k[t, part * 64])
                                    sq[0] = T.simd.vadd(sq[0], T.simd.vmul(qv, qv, full), full)
                                    sk[0] = T.simd.vadd(sk[0], T.simd.vmul(kv, kv, full), full)
                                T.simd.vsts(
                                    inv[t, 0], T.simd.vcadd(sq[0], full), one, dist="ONEPT_B32"
                                )
                                T.simd.vsts(
                                    inv[t, 1], T.simd.vcadd(sk[0], full), one, dist="ONEPT_B32"
                                )
                        with T.SimtVF(threads=32):
                            for t in T.Parallel(C):
                                inv[t, 0] = T.rsqrt(inv[t, 0] + 1e-6)
                                inv[t, 1] = T.rsqrt(inv[t, 1] + 1e-6)
                    with T.SimtVF(threads=256):
                        for t, i in T.Parallel(C, K):
                            q[t, i] = q[t, i] * (inv[t, 0] if norm else T.float32(1)) * scale
                            k[t, i] = k[t, i] * (inv[t, 1] if norm else T.float32(1))
                    with T.SimtVF(threads=128):
                        for i in T.Parallel(K):
                            total = T.alloc_local((1,), "float32")
                            total[0] = 0.0
                            for t in T.serial(C):
                                total[0] = total[0] + gp[t, i]
                                gp[t, i] = total[0]
                    with T.SimtVF(threads=256):
                        for t, j in T.Parallel(C, C):
                            gram[t, j] = 0.0
                            att[t, j] = 0.0
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.pset(32, "PAT_VL1")
                        for t in T.serial(C):
                            for j in T.serial(t + 1):
                                gg = T.simd.alloc_local(1, "float32")
                                aa = T.simd.alloc_local(1, "float32")
                                gg[0] = T.simd.vdup(T.float32(0), "float32", full)
                                aa[0] = T.simd.vdup(T.float32(0), "float32", full)
                                for part in T.serial(K // 64):
                                    qt = T.simd.vld(q[t, part * 64])
                                    kt = T.simd.vld(k[t, part * 64])
                                    kj = T.simd.vld(k[j, part * 64])
                                    gt = T.simd.vld(gp[t, part * 64])
                                    gj = T.simd.vld(gp[j, part * 64])
                                    weight = T.simd.vmul(
                                        T.simd.vexp(T.simd.vsub(gt, gj, full), full), kj, full
                                    )
                                    aa[0] = T.simd.vadd(aa[0], T.simd.vmul(qt, weight, full), full)
                                    gg[0] = T.simd.vadd(gg[0], T.simd.vmul(kt, weight, full), full)
                                T.simd.vsts(
                                    att[t, j], T.simd.vcadd(aa[0], full), one, dist="ONEPT_B32"
                                )
                                if j < t:
                                    bv = T.simd.vld(beta[t], dist="BRC_B32")
                                    value = T.simd.vmul(T.simd.vcadd(gg[0], full), bv, one)
                                    T.simd.vsts(gram[t, j], value, one, dist="ONEPT_B32")
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        for part in T.serial(K // 64):
                            for t in T.serial(C):
                                value = T.simd.alloc_local(1, "float32")
                                bv = T.simd.vld(beta[t], dist="BRC_B32")
                                kv = T.simd.vld(k[t, part * 64])
                                gate = T.simd.vexp(T.simd.vld(gp[t, part * 64]), full)
                                value[0] = T.simd.vmul(T.simd.vmul(bv, kv, full), gate, full)
                                for j in T.serial(t):
                                    gv = T.simd.vld(gram[t, j], dist="BRC_B32")
                                    prev = T.simd.vld(w[j, part * 64])
                                    value[0] = T.simd.vsub(
                                        value[0], T.simd.vmul(gv, prev, full), full
                                    )
                                T.simd.vsts(w[t, part * 64], value[0], full)
                                # Later rows read this newly solved row inside the same VF.
                                T.simd.mem_bar("VST_VLD")
                        for part in T.serial(V // 64):
                            for t in T.serial(C):
                                value = T.simd.alloc_local(1, "float32")
                                bv = T.simd.vld(beta[t], dist="BRC_B32")
                                vv = T.simd.vld(v[t, part * 64])
                                value[0] = T.simd.vmul(bv, vv, full)
                                for j in T.serial(t):
                                    gv = T.simd.vld(gram[t, j], dist="BRC_B32")
                                    prev = T.simd.vld(u[j, part * 64])
                                    value[0] = T.simd.vsub(
                                        value[0], T.simd.vmul(gv, prev, full), full
                                    )
                                T.simd.vsts(u[t, part * 64], value[0], full)
                                # Later rows read this newly solved row inside the same VF.
                                T.simd.mem_bar("VST_VLD")
                    with T.SimtVF(threads=256):
                        for t, i in T.Parallel(C, K):
                            W[job, t, i] = w[t, i]
                            QE[job, t, i] = q[t, i] * T.exp(gp[t, i])
                            KE[job, t, i] = k[t, i] * T.exp(gp[C - 1, i] - gp[t, i])
                        for t, i in T.Parallel(C, V):
                            U[job, t, i] = u[t, i]
                        for t, j in T.Parallel(C, C):
                            Att[job, t, j] = att[t, j]
                        for i in T.Parallel(K):
                            D[job, i] = T.exp(gp[C - 1, i])

    return tilelang.compile(main, target="ascend", out_idx=[7, 8, 9, 10, 11, 12])


# FP32 carried state is updated by S <- D * S + KE.T @ (U - W @ S).
# Two vector subcores split each Cube tile. C=32 gives 16 rows per subcore,
# meeting the ND-to-NZ copy alignment required by this toolchain.
@functools.lru_cache(maxsize=64)
def scan(B, S, H, K, V, dtype, initial, offsets=(), C=32, VT=64):
    starts, ends, counts, cumulative = _geometry(B, S, C, offsets)
    NS = len(starts)
    J = cumulative[-1] * H
    tiles = V // VT
    jobs = NS * H * tiles
    grid = min(32, jobs)
    R = K // 2
    RC = C // 2
    packed = bool(offsets)

    @T.prim_func
    def main(
        W: T.Tensor((J, C, K), dtype),
        U: T.Tensor((J, C, V), dtype),
        QE: T.Tensor((J, C, K), dtype),
        KE: T.Tensor((J, C, K), dtype),
        Att: T.Tensor((J, C, C), dtype),
        D: T.Tensor((J, K), "float32"),
        H0: T.Tensor((NS, H, K, V), "float32"),
        O: T.Tensor((B, S, H, V), dtype),
        HT: T.Tensor((NS, H, K, V), "float32"),
    ):
        with T.MixedKernel(grid) as (bx, sid):
            first = T.alloc_var("int32")
            last = T.alloc_var("int32")
            count = T.alloc_var("int32")
            chunk_start = T.alloc_var("int32")
            state = T.alloc_shared((R, VT), "float32")
            native = T.alloc_shared((R, VT), dtype)
            decay = T.alloc_shared((R,), "float32")
            projection = T.alloc_shared((RC, VT), "float32")
            readout = T.alloc_shared((RC, VT), "float32")
            u = T.alloc_shared((RC, VT), dtype)
            residual = T.alloc_shared((RC, VT), dtype)
            intra = T.alloc_shared((RC, VT), "float32")
            update = T.alloc_shared((R, VT), "float32")
            h1 = T.alloc_l1((K, VT), dtype)
            w1 = T.alloc_l1((C, K), dtype)
            k1 = T.alloc_l1((K, C), dtype)
            r1 = T.alloc_l1((C, VT), dtype)
            at1 = T.alloc_l1((C, C), dtype)
            wa = T.alloc_l0a((C, K), dtype)
            hb = T.alloc_l0b((K, VT), dtype)
            ka = T.alloc_l0a((K, C), dtype)
            rb = T.alloc_l0b((C, VT), dtype)
            ata = T.alloc_l0a((C, C), dtype)
            p0 = T.alloc_l0c((C, VT), "float32")
            s0 = T.alloc_l0c((K, VT), "float32")
            for step in T.serial((jobs + grid - 1) // grid):
                job = step * grid + bx
                if job < jobs:
                    seq = job // (H * tiles)
                    b = 0 if packed else seq
                    h = job // tiles % H
                    col = job % tiles * VT
                    first = _pick(seq, starts) if packed else 0
                    last = _pick(seq, ends) if packed else S
                    count = _pick(seq, counts) if packed else (S + C - 1) // C
                    chunk_start = (
                        _pick(seq, cumulative[:-1]) if packed else seq * ((S + C - 1) // C)
                    )
                    if initial:
                        T.dual_copy(H0[seq, h, :, col : col + VT], state)
                    else:
                        with T.SimtVF(threads=256):
                            for i, v in T.Parallel(R, VT):
                                state[i, v] = 0.0
                    for n in T.serial(count):
                        j = (chunk_start + n) * H + h
                        with T.SimtVF(threads=256):
                            for i, v in T.Parallel(R, VT):
                                native[i, v] = state[i, v]
                        T.dual_copy(native, h1)
                        T.copy(h1, hb)
                        T.copy(W[j, :, :], w1)
                        T.copy(w1, wa)
                        T.gemm(wa, hb, p0, clear_accum=True)
                        T.dual_copy(p0, projection)
                        T.copy(QE[j, :, :], w1)
                        T.copy(w1, wa)
                        T.gemm(wa, hb, p0, clear_accum=True)
                        T.dual_copy(p0, readout)
                        T.dual_copy(U[j, :, col : col + VT], u)
                        with T.SimtVF(threads=256):
                            for i, v in T.Parallel(RC, VT):
                                residual[i, v] = T.cast(u[i, v], "float32") - projection[i, v]
                        T.dual_copy(residual, r1)
                        T.copy(r1, rb)
                        T.copy(Att[j, :, :], at1)
                        T.copy(at1, ata)
                        T.gemm(ata, rb, p0, clear_accum=True)
                        T.dual_copy(p0, intra)
                        with T.SimtVF(threads=256):
                            for i, v in T.Parallel(RC, VT):
                                if first + n * C + sid * RC + i < last:
                                    O[b, first + n * C + sid * RC + i, h, col + v] = (
                                        readout[i, v] + intra[i, v]
                                    )
                        T.copy(KE[j, :, :], k1, transpose=True)
                        T.copy(k1, ka)
                        T.gemm(ka, rb, s0, clear_accum=True)
                        T.dual_copy(s0, update)
                        with T.SimtVF(threads=128):
                            for i in T.Parallel(R):
                                decay[i] = D[j, sid * R + i]
                        with T.SimtVF(threads=256):
                            for i, v in T.Parallel(R, VT):
                                state[i, v] = decay[i] * state[i, v] + update[i, v]
                    T.dual_copy(state, HT[seq, h, :, col : col + VT])

    return tilelang.compile(main, target="ascend", out_idx=[7, 8])


def run(c, q, k, v, g, beta, state, alog, bias, offsets=()):
    dtype = str(c.dtype).removeprefix("torch.")
    prep = prepare(
        c.batch,
        c.seq_len,
        c.heads,
        c.value_heads,
        c.dim_k,
        c.dim_v,
        dtype,
        c.scale,
        c.l2norm,
        c.gate_in_kernel,
        c.beta_sigmoid,
        c.allow_neg_eigval,
        c.lower_bound,
        c.has_dt_bias,
        offsets,
    )
    parts = prep(q, k, v, g, beta, alog, bias)
    return scan(
        c.batch, c.seq_len, c.value_heads, c.dim_k, c.dim_v, dtype, c.has_initial_state, offsets
    )(*parts, state)
