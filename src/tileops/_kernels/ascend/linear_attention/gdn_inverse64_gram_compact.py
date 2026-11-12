"""Fused raw Gram products and the 64-row gated triangular inverse."""

import tilelang.ascend.language as T


def inverse(J, K, dtype, C=64, group=1):
    assert C == 64 and group > 0 and J % group == 0
    grid = min(32, J)

    @T.prim_func
    def main(
        Q: T.Tensor((J // group, 64, K), dtype),
        KK: T.Tensor((J // group, 64, K), dtype),
        Beta: T.Tensor((J, 64), "float32"),
        Prefix: T.Tensor((J, 64), "float32"),
        B: T.Tensor((J, 64, 64), dtype),
        Att: T.Tensor((J, 64, 64), dtype),
    ):
        with T.MixedKernel(grid) as (bx, sid):
            kl1 = T.alloc_l1((64, K), dtype)
            ql1 = T.alloc_l1((64, K), dtype)
            ka = T.alloc_l0a((64, K), dtype)
            qa = T.alloc_l0a((64, K), dtype)
            kb = T.alloc_l0b((64, K), dtype)
            gramcc = T.alloc_l0c((64, 64), "float32")
            qkcc = T.alloc_l0c((64, 64), "float32")
            raw = T.alloc_shared((32, 64), "float32")
            qk = T.alloc_shared((32, 64), "float32")
            beta = T.alloc_shared((64,), "float32")
            g = T.alloc_shared((64,), "float32")
            weighted = T.alloc_shared((32, 64), "float32")
            att = T.alloc_shared((32, 64), dtype)
            gram = T.alloc_shared((32, 32), "float32")
            diagonal = T.alloc_shared((32, 32), dtype)
            cross = T.alloc_shared((32, 32), dtype)
            native = T.alloc_shared((16, 32), dtype)
            product = T.alloc_shared((16, 32), "float32")
            ds = T.alloc_l1((64, 32), dtype)
            xs = T.alloc_l1((64, 32), dtype)
            product_l1 = T.alloc_l1((32, 32), dtype)
            aa = T.alloc_l0a((32, 32), dtype)
            bb = T.alloc_l0b((32, 32), dtype)
            cc = T.alloc_l0c((32, 32), "float32")
            for step in T.serial((J + grid - 1) // grid):
                job = step * grid + bx
                if job < J:
                    T.copy(KK[job // group, :, :], kl1)
                    T.copy(Q[job // group, :, :], ql1)
                    T.copy(kl1, ka)
                    T.copy(kl1, kb)
                    T.copy(ql1, qa)
                    T.gemm(ka, kb, gramcc, transpose_B=True, clear_accum=True)
                    T.dual_copy(gramcc, raw)
                    T.gemm(qa, kb, qkcc, transpose_B=True, clear_accum=True)
                    T.dual_copy(qkcc, qk)
                    T.copy(Beta[job, :], beta)
                    T.copy(Prefix[job, :], g)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        zero = T.simd.vdup(T.float32(0), "float32")
                        gates = T.simd.vld(g[0])
                        betas = T.simd.vld(beta[0])
                        for r in T.serial(32):
                            current = T.simd.vld(g[sid * 32 + r], dist="BRC_B32")
                            decay = T.simd.vexp(T.simd.vsub(current, gates, mask), mask)
                            attention = T.simd.vmul(T.simd.vld(qk[r, 0]), decay, mask)
                            causal = T.simd.update_mask(sid * 32 + r + 1, 32)
                            attention = T.simd.vsel(attention, zero, causal)
                            T.simd.vsts(
                                att[r, 0],
                                T.simd.vcvt(attention, dtype, sat=False),
                                mask,
                                dist="PK_B32",
                            )
                            value = T.simd.vmul(
                                T.simd.vmul(T.simd.vld(raw[r, 0]), decay, mask), betas, mask
                            )
                            T.simd.vsts(weighted[r, 0], value, mask)
                    T.copy(att, Att[job, sid * 32 : (sid + 1) * 32, :])
                    with T.SimtVF(threads=512):
                        for r, j in T.Parallel(32, 32):
                            gram[r, j] = T.if_then_else(
                                j < r, weighted[r, sid * 32 + j], T.float32(0)
                            )
                            cross[r, j] = T.if_then_else(sid == 1, -weighted[r, j], T.float32(0))
                    with T.SimtVF(threads=1024):
                        tid = T.get_thread_binding()
                        i = tid // 32
                        j = tid % 32
                        value = T.alloc_local((1,), "float32")
                        value[0] = T.if_then_else(
                            i == j, T.float32(1), T.if_then_else(i > j, -gram[i, j], T.float32(0))
                        )
                        for pivot in T.unroll(31):
                            u = 31 - pivot
                            if i > u:
                                coefficient = (
                                    T.call_extern(  # lint: extern-target allow: Ascend shuffle
                                        "float32", "asc_shfl", value[0], u, 32
                                    )
                                )
                                value[0] = T.if_then_else(
                                    j < u, value[0] - coefficient * gram[u, j], value[0]
                                )
                        diagonal[i, j] = value[0]
                        B[job, sid * 32 + i, sid * 32 + j] = value[0]
                    T.dual_copy(diagonal, ds)
                    T.dual_copy(cross, xs)
                    T.copy(ds[32:64, :], aa)
                    T.copy(xs[32:64, :], bb)
                    T.gemm(aa, bb, cc, clear_accum=True)
                    T.dual_copy(cc, product)
                    with T.SimtVF(threads=512):
                        for i, j in T.Parallel(16, 32):
                            native[i, j] = product[i, j]
                    T.dual_copy(native, product_l1)
                    T.copy(product_l1, aa)
                    T.copy(ds[0:32, :], bb)
                    T.gemm(aa, bb, cc, clear_accum=True)
                    T.dual_copy(cc, product)
                    with T.SimtVF(threads=512):
                        for i, j in T.Parallel(16, 32):
                            native[i, j] = product[i, j]
                    T.dual_copy(native, B[job, 32:64, 0:32])
                    with T.SimtVF(threads=512):
                        for i, j in T.Parallel(16, 32):
                            native[i, j] = 0.0
                    T.dual_copy(native, B[job, 0:32, 32:64])

    return main
