"""Task-local W/U epilogue with DMA beta and SIMD row scaling."""

import tilelang
import tilelang.ascend.language as T


def weighted_wu(C, K, V, dtype):
    J = T.dynamic("jobs")
    R = C // 2
    grid = 32

    @T.prim_func
    def main(
        A: T.Tensor((J, C, C), dtype),
        KIn: T.Tensor((J, C, K), dtype),
        VIn: T.Tensor((J, C, V), dtype),
        Beta: T.Tensor((J, C), "float32"),
        W: T.Tensor((J, C, K), dtype),
        U: T.Tensor((J, C, V), dtype),
    ):
        with T.MixedKernel(grid) as (bx, sid):
            a = T.alloc_l1((C, C), dtype)
            k = T.alloc_l1((K, C), dtype)
            v = T.alloc_l1((V, C), dtype)
            aa = T.alloc_l0a((C, C), dtype)
            kk = T.alloc_l0b((K, C), dtype)
            vv = T.alloc_l0b((V, C), dtype)
            wk = T.alloc_l0c((C, K), "float32")
            uv = T.alloc_l0c((C, V), "float32")
            wf = T.alloc_shared((R, K), "float32")
            uf = T.alloc_shared((R, V), "float32")
            wo = T.alloc_shared((R, K), dtype)
            uo = T.alloc_shared((R, V), dtype)
            beta_ub = T.alloc_shared((C,), "float32")
            for step in T.Pipelined((J + grid - 1) // grid, num_stages=2):
                j = step * grid + bx
                if j < J:
                    T.copy(Beta[j, :], beta_ub)
                    T.copy(A[j, :, :], a)
                    T.copy(a, aa)
                    T.copy(KIn[j, :, :], k, transpose=True)
                    T.copy(k, kk)
                    T.gemm(aa, kk, wk, transpose_B=True, clear_accum=True)
                    T.dual_copy(wk, wf)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        for i in T.serial(R):
                            scale = T.simd.vld(beta_ub[i + sid * R], dist="BRC_B32")
                            for piece in T.serial(K // 64):
                                value = T.simd.vmul(T.simd.vld(wf[i, piece * 64]), scale, mask)
                                T.simd.vsts(
                                    wo[i, piece * 64],
                                    T.simd.vcvt(value, dtype, sat=False),
                                    mask,
                                    dist="PK_B32",
                                )
                    T.dual_copy(wo, W[j, :, :])
                    T.copy(VIn[j, :, :], v, transpose=True)
                    T.copy(v, vv)
                    T.gemm(aa, vv, uv, transpose_B=True, clear_accum=True)
                    T.dual_copy(uv, uf)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        for i in T.serial(R):
                            scale = T.simd.vld(beta_ub[i + sid * R], dist="BRC_B32")
                            for piece in T.serial(V // 64):
                                value = T.simd.vmul(T.simd.vld(uf[i, piece * 64]), scale, mask)
                                T.simd.vsts(
                                    uo[i, piece * 64],
                                    T.simd.vcvt(value, dtype, sat=False),
                                    mask,
                                    dist="PK_B32",
                                )
                    T.dual_copy(uo, U[j, :, :])

    return tilelang.compile(main, target="ascend", out_idx=[4, 5])
