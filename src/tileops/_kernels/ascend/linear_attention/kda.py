"""KDA recurrence correctness route with FP32 state and per-key-channel gates."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.ascend.linear_attention.kda_gate import log_decay
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.linear_attention import KDAFwdInterface
from tileops.backend._registry import register_kernel_type


def _pick(index, values):
    result = T.int32(values[-1])
    for i in range(len(values) - 2, -1, -1):
        result = T.if_then_else(index == i, T.int32(values[i]), result)
    return result


@functools.lru_cache(maxsize=128)
def recurrent(
    B,
    S,
    H,
    HV,
    K,
    V,
    dtype,
    scale,
    initial,
    norm,
    raw_gate,
    beta_sigmoid,
    negative,
    vfirst,
    offsets,
    lower_bound,
    has_bias,
):
    packed = bool(offsets)
    N = len(offsets) - 1 if packed else B
    cols = 16 if N * HV < 16 else 32
    if S == 1 and not packed and N * HV >= 32:
        cols = min(tilelang.next_power_of_2(V), 64 if N * HV < 64 else 128)
    if S == 1 and not packed:
        cols = max(64, cols)
    kp = tilelang.next_power_of_2(K)
    tiles = (V + cols - 1) // cols
    jobs = N * HV * tiles
    grid = min(64, jobs)
    starts = offsets[:-1] if packed else (0,) * N
    ends = offsets[1:] if packed else (S,) * N
    state_shape = (N, HV, V, K) if vfirst else (N, HV, K, V)

    @T.prim_func
    def main(
        Q: T.Tensor((B, S, H, K), dtype),
        KK: T.Tensor((B, S, H, K), dtype),
        VV: T.Tensor((B, S, HV, V), dtype),
        G: T.Tensor((B, S, HV, K), dtype),
        Beta: T.Tensor((B, S, HV), dtype),
        H0: T.Tensor(state_shape, "float32"),
        ALog: T.Tensor((HV,), "float32"),
        Bias: T.Tensor((HV * K,), "float32"),
        O: T.Tensor((B, S, HV, V), dtype),
        HT: T.Tensor(state_shape, "float32"),
    ):
        with T.Kernel(grid) as bx:
            q = T.alloc_shared((kp,), "float32")
            k = T.alloc_shared((kp,), "float32")
            v = T.alloc_shared((cols,), dtype)
            state = T.alloc_shared((kp, cols), "float32")
            residual = T.alloc_shared((cols,), "float32")
            out = T.alloc_shared((cols,), dtype)
            decay = T.alloc_shared((kp,), "float32")
            inv = T.alloc_shared((2,), "float32")
            parameters = T.alloc_shared((2,), "float32")
            for step in T.serial((jobs + grid - 1) // grid):
                job = step * grid + bx
                if job < jobs:
                    seq = job // (HV * tiles)
                    h = job // tiles % HV
                    qh = h // (HV // H)
                    col = job % tiles * cols
                    valid = T.min(cols, V - col)
                    b = 0 if packed else seq
                    first = _pick(seq, starts) if packed else 0
                    last = _pick(seq, ends) if packed else S
                    if initial:
                        if vfirst:
                            with T.SimtVF(threads=256):
                                for j, i in T.Parallel(cols, kp):
                                    if i < K and j < valid:
                                        state[i, j] = H0[seq, h, col + j, i]
                        else:
                            T.copy(H0[seq, h, :, col : col + valid], state[:K, :valid])
                    else:
                        with T.SimtVF(threads=256):
                            for i, j in T.Parallel(kp, cols):
                                state[i, j] = 0.0
                    for t in T.serial(first, last):
                        with T.SimtVF(threads=128):
                            for i in T.Parallel(kp):
                                q[i] = T.if_then_else(
                                    i < K, T.cast(Q[b, t, qh, i], "float32"), T.float32(0)
                                )
                                k[i] = T.if_then_else(
                                    i < K, T.cast(KK[b, t, qh, i], "float32"), T.float32(0)
                                )
                                if i < K:
                                    gate = T.cast(G[b, t, h, i], "float32")
                                    if raw_gate:
                                        gate = log_decay(
                                            gate,
                                            ALog[h],
                                            Bias[h * K + i] if has_bias else T.float32(0),
                                            lower_bound,
                                        )
                                    decay[i] = T.exp(gate)
                                else:
                                    decay[i] = 0.0
                            for _z in T.Parallel(1):
                                beta = T.cast(Beta[b, t, h], "float32")
                                if beta_sigmoid:
                                    beta = (2.0 if negative else 1.0) / (1.0 + T.exp(-beta))
                                parameters[1] = beta
                        if norm:
                            with T.SimtVF(threads=128):
                                norms = T.alloc_reducer((2,), "float32", op="sum")
                                T.reducer_init(norms)
                                for i in T.Parallel(kp):
                                    T.reducer_update(norms[0], q[i] * q[i])
                                    T.reducer_update(norms[1], k[i] * k[i])
                                sums = T.alloc_fragment((2,), "float32")
                                T.finalize_reducer(norms, sums)
                                for i in T.Parallel(2):
                                    inv[i] = T.rsqrt(sums[i] + 1e-6)
                            with T.SimtVF(threads=128):
                                for i in T.Parallel(kp):
                                    q[i] = q[i] * inv[0]
                                    k[i] = k[i] * inv[1]
                        T.copy(VV[b, t, h, col : col + valid], v[:valid])
                        if S == 1 and not packed:
                            with T.SimdVF():
                                mask = T.simd.pset(32)
                                for part in T.serial(cols // 64):
                                    projected = T.simd.alloc_local(1, "float32")
                                    projected[0] = T.simd.vdup(T.float32(0), "float32", mask)
                                    for i in T.serial(K):
                                        kv = T.simd.vld(k[i], dist="BRC_B32")
                                        decayv = T.simd.vld(decay[i], dist="BRC_B32")
                                        sv = T.simd.vld(state[i, part * 64])
                                        projected[0] = T.simd.vadd(
                                            projected[0],
                                            T.simd.vmul(T.simd.vmul(kv, decayv, mask), sv, mask),
                                            mask,
                                        )
                                    vv = T.simd.vcvt(
                                        T.simd.vld(v[part * 64], dist="UNPK_B16"),
                                        "float32",
                                        mask,
                                        part=0,
                                    )
                                    bv = T.simd.vld(parameters[1], dist="BRC_B32")
                                    delta = T.simd.vmul(
                                        bv, T.simd.vsub(vv, projected[0], mask), mask
                                    )
                                    resultv = T.simd.alloc_local(1, "float32")
                                    resultv[0] = T.simd.vdup(T.float32(0), "float32", mask)
                                    for i in T.serial(K):
                                        kv = T.simd.vld(k[i], dist="BRC_B32")
                                        qv = T.simd.vmuls(
                                            T.simd.vld(q[i], dist="BRC_B32"), T.float32(scale), mask
                                        )
                                        decayv = T.simd.vld(decay[i], dist="BRC_B32")
                                        sv = T.simd.vld(state[i, part * 64])
                                        updated = T.simd.vadd(
                                            T.simd.vmul(sv, decayv, mask),
                                            T.simd.vmul(kv, delta, mask),
                                            mask,
                                        )
                                        T.simd.vsts(state[i, part * 64], updated, mask)
                                        resultv[0] = T.simd.vadd(
                                            resultv[0], T.simd.vmul(qv, updated, mask), mask
                                        )
                                    narrow = T.simd.vcvt(resultv[0], dtype, mask, sat=False)
                                    T.simd.vsts(out[part * 64], narrow, mask, dist="PK_B32")
                        else:
                            with T.SimtVF(threads=min(1024, kp * cols)):
                                projection = T.alloc_reducer((cols,), "float32", op="sum")
                                T.reducer_init(projection)
                                for i, j in T.Parallel(kp, cols):
                                    if i < K and j < valid:
                                        T.reducer_update(
                                            projection[j], k[i] * decay[i] * state[i, j]
                                        )
                                previous = T.alloc_fragment((cols,), "float32")
                                T.finalize_reducer(projection, previous)
                                for j in T.Parallel(cols):
                                    residual[j] = parameters[1] * (
                                        T.cast(v[j], "float32") - previous[j]
                                    )
                            with T.SimtVF(threads=min(1024, kp * cols)):
                                readout = T.alloc_reducer((cols,), "float32", op="sum")
                                T.reducer_init(readout)
                                for i, j in T.Parallel(kp, cols):
                                    if i < K and j < valid:
                                        value = decay[i] * state[i, j] + k[i] * residual[j]
                                        state[i, j] = value
                                        T.reducer_update(readout[j], q[i] * scale * value)
                                output = T.alloc_fragment((cols,), "float32")
                                T.finalize_reducer(readout, output)
                                for j in T.Parallel(cols):
                                    out[j] = output[j]
                        T.copy(out[:valid], O[b, t, h, col : col + valid])
                    if vfirst:
                        with T.SimtVF(threads=256):
                            for j, i in T.Parallel(cols, kp):
                                if i < K and j < valid:
                                    HT[seq, h, col + j, i] = state[i, j]
                    else:
                        T.copy(state[:K, :valid], HT[seq, h, :, col : col + valid])

    return tilelang.compile(main, target="ascend", out_idx=[8, 9])


class AscendKDAKernel(Kernel, KDAFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call

    def forward(
        self,
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        A_log=None,
        dt_bias=None,
    ):
        c = self.call
        offsets = (
            tuple((cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.cpu()).tolist())
            if c.varlen
            else ()
        )
        state_shape = (
            (c.sequences, c.value_heads, c.dim_v, c.dim_k)
            if c.state_v_first
            else (c.sequences, c.value_heads, c.dim_k, c.dim_v)
        )
        h0 = (
            initial_state
            if initial_state is not None
            else torch.empty(state_shape, device=q.device, dtype=torch.float32)
        )
        a = (
            A_log
            if A_log is not None
            else torch.empty((c.value_heads,), device=q.device, dtype=torch.float32)
        )
        bias = (
            dt_bias
            if dt_bias is not None
            else torch.empty((c.value_heads * c.dim_k,), device=q.device, dtype=torch.float32)
        )
        if (
            c.seq_len >= 16
            and (not c.varlen or 1 <= len(offsets) - 1 <= 32)
            and not c.state_v_first
            and c.dim_k in (64, 128)
            and c.dim_v in (64, 128)
        ):
            from tileops._kernels.ascend.linear_attention.kda_chunked import run

            return run(c, q, k, v, g, beta, h0, a, bias, offsets)
        fn = recurrent(
            c.batch,
            c.seq_len,
            c.heads,
            c.value_heads,
            c.dim_k,
            c.dim_v,
            str(c.dtype).removeprefix("torch."),
            c.scale,
            c.has_initial_state,
            c.l2norm,
            c.gate_in_kernel,
            c.beta_sigmoid,
            c.allow_neg_eigval,
            c.state_v_first,
            offsets,
            c.lower_bound,
            c.has_dt_bias,
        )
        return fn(q, k, v, g, beta, h0, a, bias)


register_kernel_type("KDAFwdOp", "ascend", AscendKDAKernel)
