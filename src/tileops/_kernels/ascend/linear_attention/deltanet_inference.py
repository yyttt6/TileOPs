"""Ungated chunk recurrence with FP32 state and accurate Q/K normalization."""

import functools

import tilelang
import tilelang.ascend.language as T

from tileops._kernels.ascend.linear_attention.deltanet_chunked import matmul
from tileops._kernels.ascend.linear_attention.deltanet_inference_inverse import inverse
from tileops._kernels.ascend.linear_attention.deltanet_inference_pack import pack
from tileops._kernels.ascend.linear_attention.deltanet_pairs import build as pair_matmul


def prim(B, N, H, K, V, dtype, initial, VT=64, C=64, tokens=None, exact=False):
    tokens = N * C if tokens is None else tokens
    full_length = tokens if exact else T.dynamic("full_length")
    assert V % VT == 0
    J = B * N * H
    tiles = V // VT
    jobs = B * H * tiles
    grid = min(32, jobs)
    R = K // 2
    RC = C // 2

    @T.prim_func
    def main(
        W: T.Tensor((J, C, K), dtype),
        U: T.Tensor((J, C, V), dtype),
        KE: T.Tensor((J, C, K), dtype),
        H0: T.Tensor((B, H, K, V), "float32"),
        QE: T.Tensor((J, C, K), dtype),
        Att: T.Tensor((J, C, C), dtype),
        O: T.Tensor((B, full_length, H, V), dtype),
        offset: T.int32,
        HT: T.Tensor((B, H, K, V), "float32"),
    ):
        with T.MixedKernel(grid) as (bx, sid):
            state = T.alloc_shared((R, VT), "float32")
            native = T.alloc_shared((R, VT), dtype)
            projection = T.alloc_shared((RC, VT), "float32")
            u = T.alloc_shared((RC, VT), dtype)
            readout = T.alloc_shared((RC, VT), "float32")
            bias = T.alloc_shared((RC, VT), "float32")
            residual = T.alloc_shared((RC, VT), dtype)
            update = T.alloc_shared((R, VT), "float32")
            out = T.alloc_shared((RC, VT), dtype)
            h1 = T.alloc_l1((K, VT), dtype)
            w1 = T.alloc_l1((C, K), dtype)
            a1 = T.alloc_l1((C, C), dtype)
            aa = T.alloc_l0a((C, C), dtype)
            k1 = T.alloc_l1((K, C), dtype)
            r1 = T.alloc_l1((C, VT), dtype)
            wa = T.alloc_l0a((C, K), dtype)
            hb = T.alloc_l0b((K, VT), dtype)
            ka = T.alloc_l0a((K, C), dtype)
            rb = T.alloc_l0b((C, VT), dtype)
            p0 = T.alloc_l0c((C, VT), "float32")
            s0 = T.alloc_l0c((K, VT), "float32")
            for step in T.serial((jobs + grid - 1) // grid):
                job = step * grid + bx
                if job < jobs:
                    b = job // (H * tiles)
                    h = job // tiles % H
                    col = job % tiles * VT
                    if initial:
                        T.dual_copy(H0[b, h, :, col : col + VT], state)
                    else:
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            zero = T.simd.vdup(T.float32(0), "float32")
                            for i in T.serial(R):
                                for piece in T.serial(VT // 64):
                                    T.simd.vsts(state[i, piece * 64], zero, mask)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        for i in T.serial(R):
                            for piece in T.serial(VT // 64):
                                value = T.simd.vld(state[i, piece * 64])
                                T.simd.vsts(
                                    native[i, piece * 64],
                                    T.simd.vcvt(value, dtype, sat=False),
                                    mask,
                                    dist="PK_B32",
                                )
                    for n in T.serial(N):
                        j = (b * N + n) * H + h
                        T.dual_copy(native, h1)
                        T.copy(W[j, :, :], w1)
                        T.copy(w1, wa)
                        T.copy(h1, hb)
                        T.gemm(wa, hb, p0, clear_accum=True)
                        T.dual_copy(p0, projection)
                        T.copy(QE[j, :, :], w1)
                        T.copy(w1, wa)
                        T.gemm(wa, hb, p0, clear_accum=True)
                        T.dual_copy(p0, readout)
                        T.dual_copy(U[j, :, col : col + VT], u)
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            for i in T.serial(RC):
                                for piece in T.serial(VT // 64):
                                    uv = T.simd.vcvt(
                                        T.simd.vld(u[i, piece * 64], dist="UNPK_B16"),
                                        "float32",
                                        part=0,
                                    )
                                    value = T.simd.vsub(
                                        uv, T.simd.vld(projection[i, piece * 64]), mask
                                    )
                                    T.simd.vsts(
                                        residual[i, piece * 64],
                                        T.simd.vcvt(value, dtype, sat=False),
                                        mask,
                                        dist="PK_B32",
                                    )
                        T.dual_copy(residual, r1)
                        T.copy(r1, rb)
                        T.copy(Att[j, :, :], a1)
                        T.copy(a1, aa)
                        T.gemm(aa, rb, p0, clear_accum=True)
                        T.dual_copy(p0, bias)
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            for i in T.serial(RC):
                                for piece in T.serial(VT // 64):
                                    value = T.simd.vadd(
                                        T.simd.vld(readout[i, piece * 64]),
                                        T.simd.vld(bias[i, piece * 64]),
                                        mask,
                                    )
                                    T.simd.vsts(
                                        out[i, piece * 64],
                                        T.simd.vcvt(value, dtype, sat=False),
                                        mask,
                                        dist="PK_B32",
                                    )
                        if tokens % C == 0:
                            T.copy(
                                out,
                                O[
                                    b,
                                    (0 if exact else offset) + n * C + sid * RC : (
                                        0 if exact else offset
                                    )
                                    + n * C
                                    + (sid + 1) * RC,
                                    h,
                                    col : col + VT,
                                ],
                            )
                        else:
                            if n * C + sid * RC < tokens:
                                valid = T.min(RC, tokens - n * C - sid * RC)
                                T.copy(
                                    out[:valid, :],
                                    O[
                                        b,
                                        (0 if exact else offset) + n * C + sid * RC : (
                                            0 if exact else offset
                                        )
                                        + n * C
                                        + sid * RC
                                        + valid,
                                        h,
                                        col : col + VT,
                                    ],
                                )
                        T.copy(KE[j, :, :], k1, transpose=True)
                        T.copy(k1, ka)
                        T.gemm(ka, rb, s0, clear_accum=True)
                        T.dual_copy(s0, update)
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            for i in T.serial(R):
                                for piece in T.serial(VT // 64):
                                    value = T.simd.vadd(
                                        T.simd.vld(state[i, piece * 64]),
                                        T.simd.vld(update[i, piece * 64]),
                                        mask,
                                    )
                                    T.simd.vsts(state[i, piece * 64], value, mask)
                                    T.simd.vsts(
                                        native[i, piece * 64],
                                        T.simd.vcvt(value, dtype, sat=False),
                                        mask,
                                        dist="PK_B32",
                                    )
                    T.dual_copy(state, HT[b, h, :, col : col + VT])

    return main


class SequentialDelta:
    def __init__(self, B, S, H, K, V, dtype, scale, initial, norm):
        self.N = (S + 31) // 32
        J = B * self.N * H
        C = 32
        self.pack = tilelang.compile(
            pack(B, S, H, K, V, dtype, scale, initial, norm),
            target="ascend",
            out_idx=[5, 6, 7, 8, 9],
        )
        self.gram = matmul(J, C, C, K, dtype, tb=True)
        self.inverse = tilelang.compile(inverse(J, dtype), target="ascend", out_idx=[3, 4])
        self.wu = pair_matmul(C, K, V, C, dtype)
        self.recurrence = tilelang.compile(
            prim(B, self.N, H, K, V, dtype, initial, VT=64, C=C, tokens=S, exact=True),
            target="ascend",
            out_idx=[6, 8],
        )

    def __call__(self, q, k, v, beta, h0):
        qp, kp, vp, bp, _ = self.pack(q, k, v, beta, h0)
        gram = self.gram(kp, kp, 0)
        qk = self.gram(qp, kp, 0)
        ab, attention = self.inverse(gram, qk, bp)
        w, u, _ = self.wu(ab, kp, vp, kp)
        return self.recurrence(w, u, kp, h0, qp, attention, 0)

    def get_kernel_source(self):
        return "\n".join(
            x.get_kernel_source()
            for x in [self.pack, self.gram, self.inverse, self.wu, self.recurrence]
        )


@functools.lru_cache(maxsize=64)
def build_chunked(B, S, H, K, V, dtype, scale, initial, norm):
    if K == 128 and V == 128 and S >= 128:
        return SequentialDelta(B, S, H, K, V, dtype, scale, initial, norm)
    from tileops._kernels.ascend.linear_attention.deltanet_chunked import ChunkedDelta

    result = ChunkedDelta(B, S, H, K, V, dtype, scale, initial, norm)
    if norm:
        result.pack = tilelang.compile(
            pack(B, S, H, K, V, dtype, scale, initial, norm),
            target="ascend",
            out_idx=[5, 6, 7, 8, 9],
        )
    return result
