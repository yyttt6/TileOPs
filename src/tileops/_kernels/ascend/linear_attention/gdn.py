"""Gated delta inference: FP32 recurrent state and fused input transforms."""

import functools
from typing import ClassVar

import torch

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None
from tileops._kernels.ascend.linear_attention.deltanet import _pick
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.linear_attention import GDNFwdInterface


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
):
    packed = bool(offsets)
    N = len(offsets) - 1 if packed else B
    cols = 16 if N * HV < 16 else 32
    if S == 1 and not packed and N * HV >= 32:
        cols = min(tilelang.next_power_of_2(V), 64 if N * HV < 64 else 128)
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
        G: T.Tensor((B, S, HV), dtype),
        Beta: T.Tensor((B, S, HV), dtype),
        H0: T.Tensor(state_shape, "float32"),
        ALog: T.Tensor((HV,), "float32"),
        Bias: T.Tensor((HV,), "float32"),
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
                            for _z in T.Parallel(1):
                                gate = T.cast(G[b, t, h], "float32")
                                beta = T.cast(Beta[b, t, h], "float32")
                                if raw_gate:
                                    x = gate + Bias[h]
                                    gate = -T.exp(ALog[h]) * T.if_then_else(
                                        x > 20.0, x, T.log1p(T.exp(x))
                                    )
                                if beta_sigmoid:
                                    beta = (2.0 if negative else 1.0) / (1.0 + T.exp(-beta))
                                parameters[0] = T.exp(gate)
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
                        with T.SimtVF(threads=min(1024, kp * cols)):
                            projection = T.alloc_reducer((cols,), "float32", op="sum")
                            T.reducer_init(projection)
                            for i, j in T.Parallel(kp, cols):
                                if i < K and j < valid:
                                    T.reducer_update(projection[j], k[i] * state[i, j])
                            previous = T.alloc_fragment((cols,), "float32")
                            T.finalize_reducer(projection, previous)
                            for j in T.Parallel(cols):
                                residual[j] = parameters[1] * (
                                    T.cast(v[j], "float32") - parameters[0] * previous[j]
                                )
                        with T.SimtVF(threads=min(1024, kp * cols)):
                            readout = T.alloc_reducer((cols,), "float32", op="sum")
                            T.reducer_init(readout)
                            for i, j in T.Parallel(kp, cols):
                                if i < K and j < valid:
                                    value = parameters[0] * state[i, j] + k[i] * residual[j]
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


class AscendGDNKernel(Kernel, GDNFwdInterface):
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
        offsets = ()
        if cu_seqlens is not None:
            offsets = tuple(
                int(x)
                for x in (
                    cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.cpu()
                ).tolist()
            )
            if (
                offsets[0] != 0
                or offsets[-1] != c.seq_len
                or any(a > b for a, b in zip(offsets, offsets[1:], strict=False))
            ):
                raise ValueError("Invalid packed prefix offsets")
        if c.seq_len >= 128:
            from tileops._kernels.ascend.linear_attention.gdn_pipeline import Pipeline

            self.compiled = Pipeline(c, offsets)
            if A_log is None:
                A_log = torch.empty((c.value_heads,), dtype=torch.float32, device=q.device)
            if dt_bias is None:
                dt_bias = A_log
            return self.compiled(q, k, v, g, beta, initial_state, A_log, dt_bias)
        builder = recurrent
        if c.seq_len == 1 and c.state_v_first:
            if not offsets and c.dim_k in (64, 128):
                from tileops._kernels.ascend.linear_attention.gdn_vfirst_fused import build
            else:
                from tileops._kernels.ascend.linear_attention.gdn_vfirst import build
            builder = build
        self.compiled = builder(
            c.batch,
            c.seq_len,
            c.heads,
            c.value_heads,
            c.dim_k,
            c.dim_v,
            str(c.dtype).removeprefix("torch."),
            c.scale,
            initial_state is not None,
            c.l2norm,
            c.gate_in_kernel,
            c.beta_sigmoid,
            c.allow_neg_eigval,
            c.state_v_first,
            offsets,
        )
        if initial_state is None:
            shape = (
                (c.num_sequences, c.value_heads, c.dim_v, c.dim_k)
                if c.state_v_first
                else (c.num_sequences, c.value_heads, c.dim_k, c.dim_v)
            )
            initial_state = torch.empty(shape, dtype=torch.float32, device=q.device)
        if A_log is None:
            A_log = torch.empty((c.value_heads,), dtype=torch.float32, device=q.device)
        if dt_bias is None:
            dt_bias = A_log
        return self.compiled(q, k, v, g, beta, initial_state, A_log, dt_bias)


if T is not None:
    from tileops.backend._registry import register_kernel_type

    register_kernel_type("GDNFwdOp", "ascend", AscendGDNKernel)
