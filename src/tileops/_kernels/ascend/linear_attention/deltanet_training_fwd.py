"""Dense DeltaNet training forward, including all six saved tensors."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch
import torch.nn.functional as F

from tileops._kernels.ascend.linear_attention.deltanet_chunked import matmul
from tileops._kernels.ascend.linear_attention.deltanet_group_scan import build_group_scan
from tileops._kernels.ascend.linear_attention.deltanet_pairs import build as paired
from tileops._kernels.ascend.linear_attention.deltanet_scan import build_scan
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.linear_attention import DeltaNetChunkFwdInterface


def inverse(J, C, dtype, output_dtype):
    grid = min(64, J)
    W = C // 32

    @T.prim_func
    def main(
        G: T.Tensor((J, C, C), "float32"),
        QK: T.Tensor((J, C, C), "float32"),
        Beta: T.Tensor((J, C), "float32"),
        AB: T.Tensor((J, C, C), dtype),
        Att: T.Tensor((J, C, C), dtype),
        A: T.Tensor((J, C, C), output_dtype),
    ):
        with T.Kernel(grid) as bx:
            gram = T.alloc_shared((C, C), "float32")
            beta = T.alloc_shared((C,), "float32")
            for step in T.serial((J + grid - 1) // grid):
                job = step * grid + bx
                if job < J:
                    T.copy(G[job, :, :], gram)
                    T.copy(Beta[job, :], beta)
                    with T.SimtVF(threads=1024):
                        tid = T.get_thread_binding()
                        lane = tid % 32
                        for rows in T.serial(W):
                            i = tid // 32 + rows * 32
                            value = T.alloc_local((W,), "float32")
                            for part in T.unroll(W):
                                j = lane + part * 32
                                value[part] = T.if_then_else(
                                    i == j,
                                    T.float32(1),
                                    T.if_then_else(i > j, -beta[i] * gram[i, j], T.float32(0)),
                                )
                            for pivot in T.serial(C - 1):
                                u = C - 1 - pivot
                                selected = T.alloc_local((1,), "float32")
                                selected[0] = 0.0
                                for part in T.unroll(W):
                                    if u >= part * 32 and u < (part + 1) * 32:
                                        selected[0] = value[part]
                                coefficient = (
                                    T.call_extern(  # lint: extern-target allow: Ascend native SIMD lowering
                                        "float32", "asc_shfl", selected[0], u % 32, 32
                                    )
                                    * beta[u]
                                )
                                for part in T.unroll(W):
                                    j = lane + part * 32
                                    value[part] = T.if_then_else(
                                        i > u and j < u,
                                        value[part] - coefficient * gram[u, j],
                                        value[part],
                                    )
                            for part in T.unroll(W):
                                j = lane + part * 32
                                AB[job, i, j] = (
                                    value[part]
                                    * beta[j]
                                    / T.if_then_else(beta[i] != 0.0, beta[i], T.float32(1))
                                )
                                A[job, i, j] = value[part]
                                Att[job, i, j] = T.if_then_else(j <= i, QK[job, i, j], T.float32(0))

    return tilelang.compile(main, target="ascend", out_idx=[3, 4, 5])


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
            for step in T.Pipelined((J + grid - 1) // grid, num_stages=2):
                j = step * grid + bx
                if j < J:
                    T.copy(A[j, :, :], a)
                    T.copy(a, aa)
                    T.copy(KIn[j, :, :], k, transpose=True)
                    T.copy(k, kk)
                    T.gemm(aa, kk, wk, transpose_B=True, clear_accum=True)
                    T.dual_copy(wk, wf)
                    with T.SimtVF(threads=256):
                        for i, t in T.Parallel(R, K):
                            wo[i, t] = wf[i, t] * Beta[j, i + sid * R]
                    T.dual_copy(wo, W[j, :, :])
                    T.copy(VIn[j, :, :], v, transpose=True)
                    T.copy(v, vv)
                    T.gemm(aa, vv, uv, transpose_B=True, clear_accum=True)
                    T.dual_copy(uv, uf)
                    with T.SimtVF(threads=256):
                        for i, t in T.Parallel(R, V):
                            uo[i, t] = uf[i, t] * Beta[j, i + sid * R]
                    T.dual_copy(uo, U[j, :, :])

    return tilelang.compile(main, target="ascend", out_idx=[4, 5])


def boundaries(BH, N, K, V, dtype):
    J = BH * N
    grid = min(64, J)

    @T.prim_func
    def main(
        Full: T.Tensor((J, K, V), "float32"),
        Before: T.Tensor((J, K, V), dtype),
        S: T.Tensor((BH, N + 1, K, V), "float32"),
    ):
        with T.Kernel(grid) as bx:
            buf = T.alloc_shared((K, V), "float32")
            native = T.alloc_shared((K, V), dtype)
            for step in T.serial((J + grid - 1) // grid):
                j = step * grid + bx
                if j < J:
                    n = j % N
                    b = j // N
                    T.copy(Full[j, :, :], buf)
                    T.copy(buf, S[b, n + 1, :, :])
                    with T.SimtVF(threads=256):
                        for i, k in T.Parallel(K, V):
                            native[i, k] = buf[i, k]
                    if n + 1 < N:
                        T.copy(native, Before[j + 1, :, :])
                    if n == 0:
                        with T.SimtVF(threads=256):
                            for i, k in T.Parallel(K, V):
                                buf[i, k] = 0.0
                                native[i, k] = 0.0
                        T.copy(buf, S[b, 0, :, :])
                        T.copy(native, Before[j, :, :])

    return tilelang.compile(main, target="ascend", out_idx=[1, 2])


def output(J, C, V, dtype):
    grid = min(64, J)

    @T.prim_func
    def main(
        X: T.Tensor((J, C, V), "float32"),
        Y: T.Tensor((J, C, V), "float32"),
        O: T.Tensor((J, C, V), dtype),
    ):
        with T.Kernel(grid) as bx:
            x = T.alloc_shared((C, V), "float32")
            y = T.alloc_shared((C, V), "float32")
            o = T.alloc_shared((C, V), dtype)
            for step in T.serial((J + grid - 1) // grid):
                j = step * grid + bx
                if j < J:
                    T.copy(X[j, :, :], x)
                    T.copy(Y[j, :, :], y)
                    with T.SimtVF(threads=256):
                        for i, k in T.Parallel(C, V):
                            o[i, k] = x[i, k] + y[i, k]
                    T.copy(o, O[j, :, :])

    return tilelang.compile(main, target="ascend", out_idx=[2])


class Training:
    def __init__(self, B, H, L, K, V, C, output_dtype):
        self.shape = B, H, L, K, V, C
        self.N = N = L // C
        self.J = J = B * H * N
        self.CP = C = (C + 31) // 32 * 32
        self.K = KP = (K + 31) // 32 * 32
        self.V = VP = (V + 31) // 32 * 32
        self.dtype = dtype = "float16" if output_dtype == "float32" else output_dtype
        self.gram = matmul(J, C, C, KP, dtype, tb=True)
        self.inv = inverse(J, C, dtype, output_dtype)
        self.wu = weighted_wu(C, KP, VP, dtype)
        self.affine = paired(KP, KP, VP, C, dtype, ta=True, first_mode=1, save_full=N == 1)
        self.qbias = paired(C, KP, VP, C, dtype, first_mode=2, second_fp32=True)
        self.group = build_group_scan(B * H, N, 1, KP, VP, dtype, False, G=4) if N >= 64 else None
        self.scan = build_scan(B * H, N, 1, KP, VP, dtype) if self.group is None else None
        self.boundaries = boundaries(B * H, N, KP, VP, dtype)
        self.readout = matmul(J, C, VP, KP, dtype)
        self.output = output(J, C, VP, output_dtype)

    def __call__(self, q, k, v, beta):
        B, H, L, K, V, C = self.shape
        dt = getattr(torch, self.dtype)
        qp, kp = (
            F.pad(x.reshape(self.J, C, K), (0, self.K - K, 0, self.CP - C)).to(dt) for x in (q, k)
        )
        vp = F.pad(v.reshape(self.J, C, V), (0, self.V - V, 0, self.CP - C)).to(dt)
        bp = F.pad(beta.reshape(self.J, C).float(), (0, self.CP - C))
        ab, att, a = self.inv(self.gram(kp, kp, 0), self.gram(qp, kp, 0), bp)
        w, u = self.wu(ab, kp, vp, bp)
        m, state, full = self.affine(kp, w, u, w)
        qe, bias, _ = self.qbias(att, w, u, qp)
        if self.group is not None:
            m, full = self.group(m, state)
        else:
            stride = 1
            while stride < self.N:
                m, state, full = self.scan(m, state, stride)
                stride *= 2
        before, s = self.boundaries(full)
        o = self.output(self.readout(qe, before, 0), bias)
        aw = a[:, :C, :C].contiguous().reshape(B, H, L, C)
        return (
            o[:, :C, :V].contiguous().reshape(B, H, L, V),
            s.reshape(B, H, self.N + 1, self.K, self.V)[..., :K, :V].contiguous(),
            aw,
            aw.clone(),
            w[:, :C, :K].to(q.dtype).contiguous().reshape(B, H, L, K),
            u[:, :C, :V].to(q.dtype).contiguous().reshape(B, H, L, V),
        )

    def get_kernel_source(self):
        return "\n".join(
            x.get_kernel_source() for x in vars(self).values() if hasattr(x, "get_kernel_source")
        )


build = functools.lru_cache(maxsize=64)(Training)


class AscendDeltaNetChunkFwdKernel(Kernel, DeltaNetChunkFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        self.compiled = build(
            call.batch,
            call.heads,
            call.seq_len,
            call.dim_k,
            call.dim_v,
            call.chunk_size,
            str(call.dtype).removeprefix("torch."),
        )

    def forward(self, q, k, v, beta):
        return self.compiled(*(x.contiguous() for x in (q, k, v, beta)))


from tileops.backend._registry import register_kernel_type  # noqa: E402

register_kernel_type("DeltaNetChunkFwdOp", "ascend", AscendDeltaNetChunkFwdKernel)
