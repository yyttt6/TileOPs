"""DeltaNet inference: FP32 state, packed sequences and optional Q/K normalization."""

import functools
from typing import ClassVar

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None
import torch

from tileops._kernels.kernel_base import Kernel
from tileops._kernels.linear_attention import DeltaNetFwdInterface


def _pick(index, values):
    result = T.int32(values[-1])
    for i in range(len(values) - 2, -1, -1):
        result = T.if_then_else(index == i, T.int32(values[i]), result)
    return result


@functools.lru_cache(maxsize=128)
def build(B, S, H, K, V, dtype, scale, initial, l2norm, offsets, tuning_cols=0):
    packed = bool(offsets)
    N = len(offsets) - 1 if packed else B
    cols = 16 if N * H < 16 else 32
    if S == 1 and not packed and N * H >= 32:
        cols = min(tilelang.next_power_of_2(V), 64 if N * H < 64 else 128)
    if 1 < S < 128 and K == 128 and V == 128 and N * H >= 16:
        cols = 64
    if tuning_cols:
        cols = tuning_cols
    cache_short = not packed and 32 <= S < 128 and K == 128 and V == 128
    kp = tilelang.next_power_of_2(K)
    tiles = (V + cols - 1) // cols
    jobs = N * H * tiles
    grid = min(64, jobs)
    starts = offsets[:-1] if packed else (0,) * N
    ends = offsets[1:] if packed else (S,) * N

    @T.prim_func
    def main(
        Q: T.Tensor((B, S, H, K), dtype),
        KK: T.Tensor((B, S, H, K), dtype),
        VV: T.Tensor((B, S, H, V), dtype),
        Beta: T.Tensor((B, S, H), dtype),
        H0: T.Tensor((N, H, K, V), "float32"),
        O: T.Tensor((B, S, H, V), dtype),
        HT: T.Tensor((N, H, K, V), "float32"),
    ):
        with T.Kernel(grid) as bx:
            q = T.alloc_shared((kp,), "float32")
            k = T.alloc_shared((kp,), "float32")
            v = T.alloc_shared((cols,), dtype)
            state = T.alloc_shared((kp, cols), "float32")
            residual = T.alloc_shared((cols,), "float32")
            out = T.alloc_shared((cols,), dtype)
            inv = T.alloc_shared((2,), "float32")
            if cache_short:
                qs = T.alloc_shared((S, K), dtype)
                ks = T.alloc_shared((S, K), dtype)
                vs = T.alloc_shared((S, cols), dtype)
                bs = T.alloc_shared((S,), dtype)
            for step in T.serial((jobs + grid - 1) // grid):
                job = step * grid + bx
                if job < jobs:
                    seq = job // (H * tiles)
                    h = job // tiles % H
                    col = job % tiles * cols
                    valid = T.min(cols, V - col)
                    b = 0 if packed else seq
                    first = _pick(seq, starts)
                    last = _pick(seq, ends)
                    if cache_short:
                        T.copy(Q[b, :, h, :], qs)
                        T.copy(KK[b, :, h, :], ks)
                        T.copy(VV[b, :, h, col : col + valid], vs[:, :valid])
                        with T.SimtVF(threads=128):
                            for t in T.Parallel(S):
                                bs[t] = Beta[b, t, h]
                    if initial:
                        T.copy(H0[seq, h, :, col : col + valid], state[:K, :valid])
                    else:
                        with T.SimtVF(threads=256):
                            for i, j in T.Parallel(kp, cols):
                                state[i, j] = 0.0
                    for t in T.serial(first, last):
                        with T.SimtVF(threads=128):
                            for i in T.Parallel(kp):
                                q[i] = T.if_then_else(
                                    i < K,
                                    T.cast(qs[t, i] if cache_short else Q[b, t, h, i], "float32"),
                                    T.float32(0),
                                )
                                k[i] = T.if_then_else(
                                    i < K,
                                    T.cast(ks[t, i] if cache_short else KK[b, t, h, i], "float32"),
                                    T.float32(0),
                                )
                        if l2norm:
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
                        if not cache_short:
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
                                residual[j] = T.if_then_else(
                                    j < valid,
                                    T.cast(bs[t] if cache_short else Beta[b, t, h], "float32")
                                    * (
                                        T.cast(vs[t, j] if cache_short else v[j], "float32")
                                        - previous[j]
                                    ),
                                    T.float32(0),
                                )
                        with T.SimtVF(threads=min(1024, kp * cols)):
                            readout = T.alloc_reducer((cols,), "float32", op="sum")
                            T.reducer_init(readout)
                            for i, j in T.Parallel(kp, cols):
                                if i < K and j < valid:
                                    current = state[i, j] + k[i] * residual[j]
                                    state[i, j] = current
                                    T.reducer_update(readout[j], q[i] * current)
                            output = T.alloc_fragment((cols,), "float32")
                            T.finalize_reducer(readout, output)
                            for j in T.Parallel(cols):
                                out[j] = output[j] * scale
                        T.copy(out[:valid], O[b, t, h, col : col + valid])
                    T.copy(state[:K, :valid], HT[seq, h, :, col : col + valid])

    return tilelang.compile(main, target="ascend", out_idx=[5, 6])


class InferencePipeline:
    """Native recurrent fallback, dense chunks, and independent packed rows."""

    def __init__(self, B, S, H, K, V, dtype, scale, initial, norm, offsets):
        from tileops._kernels.ascend.linear_attention.deltanet_inference import build_chunked

        self.offsets = offsets
        self.shape = (B, S, H, K, V)
        self.equal_length = None
        if offsets and len(set(b - a for a, b in zip(offsets, offsets[1:], strict=False))) == 1:
            self.equal_length = offsets[1] - offsets[0]
            B = len(offsets) - 1
            S = self.equal_length
        lengths = (
            [b - a for a, b in zip(offsets, offsets[1:], strict=False)]
            if offsets and self.equal_length is None
            else [S]
        )
        self.kernels = []
        for length in lengths:
            batch = 1 if offsets and self.equal_length is None else B
            kernel = (
                build_chunked(batch, length, H, K, V, dtype, scale, initial, norm)
                if length >= 128
                else build(batch, length, H, K, V, dtype, scale, initial, norm, ())
            )
            self.kernels.append(kernel)

    def __call__(self, q, k, v, beta, h0):
        B, S, H, K, V = self.shape
        if self.equal_length is not None:
            rows = len(self.offsets) - 1
            length = self.equal_length
            o, ht = self.kernels[0](
                q.view(rows, length, H, K),
                k.view(rows, length, H, K),
                v.view(rows, length, H, V),
                beta.view(rows, length, H),
                h0,
            )
            return o.reshape(B, S, H, V), ht
        if self.offsets:
            outputs = []
            states = []
            for i, (start, end) in enumerate(zip(self.offsets, self.offsets[1:], strict=False)):
                o, ht = self.kernels[i](
                    q[:, start:end],
                    k[:, start:end],
                    v[:, start:end],
                    beta[:, start:end],
                    h0[i : i + 1],
                )
                outputs.append(o)
                states.append(ht)
            return torch.cat(outputs, dim=1), torch.cat(states, dim=0)
        return self.kernels[0](q, k, v, beta, h0)

    def get_kernel_source(self):
        return "\n".join(k.get_kernel_source() for k in self.kernels)


@functools.lru_cache(maxsize=128)
def select_pipeline(B, S, H, K, V, dtype, scale, initial, norm, offsets):
    if (
        S >= 32
        and K in (64, 128)
        and V in (64, 128)
        and not any(a == b for a, b in zip(offsets, offsets[1:], strict=False))
    ):
        return InferencePipeline(B, S, H, K, V, dtype, scale, initial, norm, offsets)
    return build(B, S, H, K, V, dtype, scale, initial, norm, offsets)


class AscendDeltaNetKernel(Kernel, DeltaNetFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call

    def forward(self, q, k, v, beta, initial_state=None, cu_seqlens=None, cu_seqlens_cpu=None):
        c = self.call
        offsets = ()
        if cu_seqlens is not None:
            # The interface supplies an identical CPU mirror when available.
            # Read its payload every call; never cache offsets by tensor identity.
            metadata = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.cpu()
            offsets = tuple(int(x) for x in metadata.tolist())
            if (
                offsets[0] != 0
                or offsets[-1] != c.seq_len
                or any(a > b for a, b in zip(offsets, offsets[1:], strict=False))
            ):
                raise ValueError(
                    "cu_seqlens must be nondecreasing prefix offsets covering the sequence"
                )
        self.compiled = select_pipeline(
            c.batch,
            c.seq_len,
            c.heads,
            c.dim_k,
            c.dim_v,
            str(c.dtype).removeprefix("torch."),
            c.scale,
            initial_state is not None,
            c.l2norm,
            offsets,
        )
        if initial_state is None:
            initial_state = torch.empty(
                (c.num_sequences, c.heads, c.dim_k, c.dim_v), device=q.device, dtype=torch.float32
            )
        return self.compiled(q, k, v, beta, initial_state)


if T is not None:
    from tileops.backend._registry import register_kernel_type

    register_kernel_type("DeltaNetFwdOp", "ascend", AscendDeltaNetKernel)
