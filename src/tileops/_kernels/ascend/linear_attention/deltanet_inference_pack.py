"""Strided DMA packing for the dense chunked delta rule."""

import tilelang.ascend.language as T


def pack(B, S, H, K, V, dtype, scale, initial, norm, C=32):
    N = (S + C - 1) // C
    J = B * N * H
    grid = min(64, J)

    @T.prim_func
    def main(
        Q: T.Tensor((B, S, H, K), dtype),
        KK: T.Tensor((B, S, H, K), dtype),
        VV: T.Tensor((B, S, H, V), dtype),
        Beta: T.Tensor((B, S, H), dtype),
        H0: T.Tensor((B, H, K, V), "float32"),
        QP: T.Tensor((J, C, K), dtype),
        KP: T.Tensor((J, C, K), dtype),
        VP: T.Tensor((J, C, V), dtype),
        BP: T.Tensor((J, C), "float32"),
        HP: T.Tensor((B * H, K, V), dtype),
    ):
        with T.Kernel(grid) as bx:
            q = T.alloc_shared((C, K), "float32")
            k = T.alloc_shared((C, K), "float32")
            qin = T.alloc_shared((C, K), dtype)
            kin = T.alloc_shared((C, K), dtype)
            qn = T.alloc_shared((C, K), dtype)
            kn = T.alloc_shared((C, K), dtype)
            vn = T.alloc_shared((C, V), dtype)
            inv = T.alloc_shared((C, 2), "float32")
            hs = T.alloc_shared((K, V), "float32")
            hn = T.alloc_shared((K, V), dtype)
            for step in T.serial((J + grid - 1) // grid):
                j = step * grid + bx
                if j < J:
                    b = j // (N * H)
                    n = j // H % N
                    h = j % H
                    valid = T.min(C, S - n * C)
                    if valid < C:
                        with T.SimtVF(threads=256):
                            for t, i in T.Parallel(C, K):
                                qin[t, i] = 0.0
                                kin[t, i] = 0.0
                            for t, i in T.Parallel(C, V):
                                vn[t, i] = 0.0
                    T.copy(Q[b, n * C : n * C + valid, h, :], qin[:valid, :])
                    T.copy(KK[b, n * C : n * C + valid, h, :], kin[:valid, :])
                    T.copy(VV[b, n * C : n * C + valid, h, :], vn[:valid, :])
                    T.copy(vn, VP[j, :, :])
                    with T.SimtVF(threads=256):
                        for t, i in T.Parallel(C, K):
                            q[t, i] = qin[t, i]
                            k[t, i] = kin[t, i]
                        for t in T.Parallel(C):
                            BP[j, t] = T.if_then_else(
                                n * C + t < S,
                                T.cast(Beta[b, n * C + t, h], "float32"),
                                T.float32(0),
                            )
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
                                    qi = T.simd.vld(q[t, part * 64])
                                    ki = T.simd.vld(k[t, part * 64])
                                    sq[0] = T.simd.vadd(sq[0], T.simd.vmul(qi, qi, full), full)
                                    sk[0] = T.simd.vadd(sk[0], T.simd.vmul(ki, ki, full), full)
                                T.simd.vsts(
                                    inv[t, 0], T.simd.vcadd(sq[0], full), one, dist="ONEPT_B32"
                                )
                                T.simd.vsts(
                                    inv[t, 1], T.simd.vcadd(sk[0], full), one, dist="ONEPT_B32"
                                )
                        with T.SimtVF(threads=256):
                            for t in T.Parallel(C):
                                inv[t, 0] = T.rsqrt(inv[t, 0] + 1e-6)
                                inv[t, 1] = T.rsqrt(inv[t, 1] + 1e-6)
                    with T.SimtVF(threads=256):
                        for t, i in T.Parallel(C, K):
                            qn[t, i] = q[t, i] * (inv[t, 0] if norm else T.float32(1)) * scale
                            kn[t, i] = k[t, i] * (inv[t, 1] if norm else T.float32(1))
                    T.copy(qn, QP[j, :, :])
                    T.copy(kn, KP[j, :, :])
                    if initial:  # noqa: SIM102
                        if n == 0:
                            T.copy(H0[b, h, :, :], hs)
                            with T.SimtVF(threads=256):
                                for i, v in T.Parallel(K, V):
                                    hn[i, v] = hs[i, v]
                            T.copy(hn, HP[b * H + h, :, :])

    return main
