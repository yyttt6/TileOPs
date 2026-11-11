"""C64 SIMD pack for non-normalized inputs; retain qualified norm path."""

import tilelang.ascend.language as T


def pack(B, S, H, HV, K, V, dtype, scale, initial, norm, raw_gate, beta_sigmoid, negative, C=64):
    assert not norm and HV % H == 0
    assert C == 64 and K in (64, 128) and V in (64, 128)
    N = (S + C - 1) // C
    J = B * N * HV
    group = HV // H
    JG = B * N * H
    grid = min(64, J)

    @T.prim_func
    def main(
        Q: T.Tensor((B, S, H, K), dtype),
        KK: T.Tensor((B, S, H, K), dtype),
        VV: T.Tensor((B, S, HV, V), dtype),
        G: T.Tensor((B, S, HV), dtype),
        Beta: T.Tensor((B, S, HV), dtype),
        H0: T.Tensor((B, HV, K, V), "float32"),
        ALog: T.Tensor((HV,), "float32"),
        Bias: T.Tensor((HV,), "float32"),
        QP: T.Tensor((JG, C, K), dtype),
        KP: T.Tensor((JG, C, K), dtype),
        VP: T.Tensor((J, C, V), dtype),
        BP: T.Tensor((J, C), "float32"),
        GP: T.Tensor((J, C), "float32"),
        QG: T.Tensor((J, C, K), dtype),
        KG: T.Tensor((J, C, K), dtype),
        KE: T.Tensor((J, C, K), dtype),
    ):
        with T.Kernel(grid) as bx:
            qi = T.alloc_shared((C, K), dtype)
            ki = T.alloc_shared((C, K), dtype)
            qn = T.alloc_shared((C, K), dtype)
            qg = T.alloc_shared((C, K), dtype)
            kg = T.alloc_shared((C, K), dtype)
            ke = T.alloc_shared((C, K), dtype)
            vn = T.alloc_shared((C, V), dtype)
            gp = T.alloc_shared((C,), "float32")
            bp = T.alloc_shared((C,), "float32")
            for step in T.serial((J + grid - 1) // grid):
                job = step * grid + bx
                if job < J:
                    b = job // (N * HV)
                    n = job // HV % N
                    h = job % HV
                    qh = h // (HV // H)
                    valid = T.min(C, S - n * C)
                    if valid < C:
                        with T.SimtVF(threads=256):
                            for t, i in T.Parallel(C, K):
                                qi[t, i] = 0.0
                                ki[t, i] = 0.0
                            for t, i in T.Parallel(C, V):
                                vn[t, i] = 0.0
                    T.copy(Q[b, n * C : n * C + valid, qh, :], qi[:valid, :])
                    T.copy(KK[b, n * C : n * C + valid, qh, :], ki[:valid, :])
                    T.copy(VV[b, n * C : n * C + valid, h, :], vn[:valid, :])
                    T.copy(vn, VP[job, :, :])
                    with T.SimtVF(threads=32):
                        lane = T.get_thread_binding()
                        prefix = T.alloc_local((2,), "float32")
                        for part in T.unroll(2):
                            token = n * C + part * 32 + lane
                            if token < S:
                                gg = T.cast(G[b, token, h], "float32")
                                if raw_gate:
                                    x = gg + Bias[h]
                                    gg = -T.exp(ALog[h]) * T.if_then_else(
                                        x > 20.0, x, T.log1p(T.exp(x))
                                    )
                                prefix[part] = gg
                                bb = T.cast(Beta[b, token, h], "float32")
                                if beta_sigmoid:
                                    bb = (2.0 if negative else 1.0) / (1.0 + T.exp(-bb))
                                bp[part * 32 + lane] = bb
                            else:
                                prefix[part] = 0.0
                                bp[part * 32 + lane] = 0.0
                            for shift in T.unroll(5):
                                offset = 1 << shift
                                other = T.call_extern(  # lint: extern-target allow: Ascend shuffle
                                    "float32", "asc_shfl", prefix[part], T.max(0, lane - offset), 32
                                )
                                prefix[part] = prefix[part] + T.if_then_else(
                                    lane >= offset, other, T.float32(0)
                                )
                        prefix[1] = prefix[
                            1
                        ] + T.call_extern(  # lint: extern-target allow: Ascend shuffle
                            "float32", "asc_shfl", prefix[0], 31, 32
                        )
                        gp[lane] = prefix[0]
                        gp[32 + lane] = prefix[1]
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        end = T.simd.vld(gp[C - 1], dist="BRC_B32")
                        scaling = T.simd.vdup(T.float32(scale), "float32")
                        for row in T.serial(C):
                            current = T.simd.vld(gp[row], dist="BRC_B32")
                            eg = T.simd.vexp(current, mask)
                            ek = T.simd.vexp(T.simd.vsub(end, current, mask), mask)
                            for piece in T.serial(K // 64):
                                qq = T.simd.vcvt(
                                    T.simd.vld(qi[row, piece * 64], dist="UNPK_B16"),
                                    "float32",
                                    part=0,
                                )
                                kk = T.simd.vcvt(
                                    T.simd.vld(ki[row, piece * 64], dist="UNPK_B16"),
                                    "float32",
                                    part=0,
                                )
                                qq = T.simd.vmul(qq, scaling, mask)
                                T.simd.vsts(
                                    qn[row, piece * 64],
                                    T.simd.vcvt(qq, dtype, sat=False),
                                    mask,
                                    dist="PK_B32",
                                )
                                T.simd.vsts(
                                    qg[row, piece * 64],
                                    T.simd.vcvt(T.simd.vmul(qq, eg, mask), dtype, sat=False),
                                    mask,
                                    dist="PK_B32",
                                )
                                T.simd.vsts(
                                    kg[row, piece * 64],
                                    T.simd.vcvt(T.simd.vmul(kk, eg, mask), dtype, sat=False),
                                    mask,
                                    dist="PK_B32",
                                )
                                T.simd.vsts(
                                    ke[row, piece * 64],
                                    T.simd.vcvt(T.simd.vmul(kk, ek, mask), dtype, sat=False),
                                    mask,
                                    dist="PK_B32",
                                )
                    if job % group == 0:
                        T.copy(qn, QP[job // group, :, :])
                        T.copy(ki, KP[job // group, :, :])
                    T.copy(qg, QG[job, :, :])
                    T.copy(kg, KG[job, :, :])
                    T.copy(ke, KE[job, :, :])
                    T.copy(gp, GP[job, :])
                    T.copy(bp, BP[job, :])

    return main
