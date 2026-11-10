"""GLA training forward with full-precision chunk boundary states."""

import functools
from typing import ClassVar

import torch

from tileops._kernels.ascend.linear_attention.gla import T, tilelang
from tileops._kernels.ascend.linear_attention.gla_chunked import (
    output,
    preprocessing,
    preprocessing_mixed,
    scores,
    scores_cross,
    scores_diagonal,
    summary,
)
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.linear_attention.call_spec import GLAChunkFwdInterface


def scan_states(B, N, H, K, V, dtype, initial, chunk_size):
    stride = chunk_size // 32
    count = N // stride
    cols = 8 if B * H < 16 else 32
    tiles = (V + cols - 1) // cols
    jobs = B * H * tiles
    grid = min(64, jobs)

    @T.prim_func
    def main(
        U: T.Tensor((B * N * H, K, V), "float32"),
        D: T.Tensor((B * N * H, K), "float32"),
        H0: T.Tensor((B, H, K, V), "float32"),
        HB: T.Tensor((B * N * H, K, V), dtype),
        HT: T.Tensor((B, H, K, V), "float32"),
        HS: T.Tensor((B, count + 1, H, K, V), "float32"),
    ):
        with T.Kernel(grid) as bx:  # noqa: SIM117 - TileLang DSL scopes are distinct
            with T.SimtVF(threads=256):
                state = T.alloc_fragment((K, cols), "float32")
                for step in T.serial((jobs + grid - 1) // grid):
                    job = step * grid + bx
                    if job < jobs:
                        b = job // (H * tiles)
                        h = job // tiles % H
                        col = job % tiles * cols
                        for i, j in T.Parallel(K, cols):
                            if col + j < V:
                                state[i, j] = H0[b, h, i, col + j] if initial else T.float32(0)
                        for n in T.serial(N):
                            index = (b * N + n) * H + h
                            for i, j in T.Parallel(K, cols):
                                if col + j < V:
                                    if n % stride == 0:
                                        HS[b, n // stride, h, i, col + j] = state[i, j]
                                    HB[index, i, col + j] = state[i, j]
                                    state[i, j] = state[i, j] * D[index, i] + U[index, i, col + j]
                        for i, j in T.Parallel(K, cols):
                            if col + j < V:
                                HT[b, h, i, col + j] = state[i, j]
                                HS[b, count, h, i, col + j] = state[i, j]

    return main


@functools.lru_cache(maxsize=128)
def build_recurrent_states(batch, length, heads, dk, dv, dtype, scale, initial, chunk_size):
    offsets = ()
    packed = bool(offsets)
    sequences = len(offsets) - 1 if packed else batch
    columns = 16 if sequences * heads < 16 else 32
    kp = (dk + 15) // 16 * 16
    keys = tilelang.next_power_of_2(dk)
    tiles = (dv + columns - 1) // columns
    jobs = sequences * heads * tiles
    grid = min(64, jobs)

    @T.prim_func
    def main(
        Q: T.Tensor((batch, length, heads, dk), dtype),
        K: T.Tensor((batch, length, heads, dk), dtype),
        V: T.Tensor((batch, length, heads, dv), dtype),
        G: T.Tensor((batch, length, heads, dk), dtype),
        H0: T.Tensor((sequences, heads, dk, dv), "float32"),
        O: T.Tensor((batch, length, heads, dv), dtype),
        HT: T.Tensor((sequences, heads, dk, dv), "float32"),
        HS: T.Tensor((batch, length // chunk_size + 1, heads, dk, dv), "float32"),
    ):
        with T.Kernel(grid) as bx:
            q = T.alloc_shared((kp,), dtype)
            k = T.alloc_shared((kp,), dtype)
            g = T.alloc_shared((kp,), dtype)
            v = T.alloc_shared((columns,), dtype)
            state = T.alloc_shared((kp, columns), "float32")
            out = T.alloc_shared((columns,), dtype)
            for job_index in T.serial((jobs + grid - 1) // grid):
                job = job_index * grid + bx
                if job < jobs:
                    seq = job // (heads * tiles)
                    head = job // tiles % heads
                    col = job % tiles * columns
                    valid = T.min(columns, dv - col)
                    b = 0 if packed else seq
                    if initial:
                        T.copy(H0[seq, head, :, col : col + valid], state[:dk, :valid])
                    else:
                        with T.SimtVF(threads=256):
                            for i, j in T.Parallel(kp, columns):
                                state[i, j] = 0.0
                    T.copy(state[:dk, :valid], HS[seq, 0, head, :, col : col + valid])
                    for chunk in T.serial(length // chunk_size):
                        for within in T.serial(chunk_size):
                            token = chunk * chunk_size + within
                            T.copy(Q[b, token, head, :], q[:dk])
                            T.copy(K[b, token, head, :], k[:dk])
                            T.copy(G[b, token, head, :], g[:dk])
                            T.copy(V[b, token, head, col : col + valid], v[:valid])
                            with T.SimtVF(threads=min(1024, keys * columns)):
                                total = T.alloc_reducer((columns,), "float32", op="sum")
                                T.reducer_init(total)
                                for i, j in T.Parallel(keys, columns):
                                    if i < dk and j < valid:
                                        value = T.exp(T.cast(g[i], "float32")) * state[
                                            i, j
                                        ] + T.cast(k[i], "float32") * T.cast(v[j], "float32")
                                        state[i, j] = value
                                        T.reducer_update(total[j], T.cast(q[i], "float32") * value)
                                result = T.alloc_fragment((columns,), "float32")
                                T.finalize_reducer(total, result)
                                for j in T.Parallel(columns):
                                    out[j] = result[j] * scale
                            T.copy(out[:valid], O[b, token, head, col : col + valid])
                        T.copy(state[:dk, :valid], HS[seq, chunk + 1, head, :, col : col + valid])
                    T.copy(state[:dk, :valid], HT[seq, head, :, col : col + valid])

    return tilelang.compile(main, target="ascend", out_idx=[5, 6, 7])


class TrainingGLA:
    def __init__(self, B, S, H, K, V, dtype, scale, initial, chunk_size):
        C = 32
        self.length = S
        N = (S + C - 1) // C
        J = B * N * H
        self.mixed = K == 128
        if self.mixed:
            factories = [
                (preprocessing_mixed(B, S, H, K, V, dtype, scale, C), [4, 5, 6, 7, 8, 9, 10]),
                (summary(J, K, V, dtype, C), [2]),
                (scan_states(B, N, H, K, V, dtype, initial, chunk_size), [3, 4, 5]),
                (scores_diagonal(B, S, H, K, dtype, scale), [3]),
                (scores_cross(J, K, dtype), []),
                (output(B, S, H, K, V, dtype, C), [4]),
            ]
        else:
            factories = [
                (preprocessing(B, S, H, K, V, dtype, scale, C), [4, 5, 6, 7, 8]),
                (summary(J, K, V, dtype, C), [2]),
                (scan_states(B, N, H, K, V, dtype, initial, chunk_size), [3, 4, 5]),
                (scores(B, S, H, K, dtype, scale, C), [3]),
                (output(B, S, H, K, V, dtype, C), [4]),
            ]
        self.kernels = [
            tilelang.compile(fn, target="ascend", out_idx=indices) for fn, indices in factories
        ]

    def __call__(self, q, k, v, g, state):
        prepared = self.kernels[0](q, k, v, g)
        gc, qg, kw, vb, decay = prepared[:5]
        contribution = self.kernels[1](kw, vb)
        boundaries, final, states = self.kernels[2](contribution, decay, state)
        attention = self.kernels[3](q, k, gc)
        if self.mixed:
            self.kernels[4](prepared[5], prepared[6], attention)
            out = self.kernels[5](qg, boundaries, attention, vb)
        else:
            out = self.kernels[4](qg, boundaries, attention, vb)
        return out[:, : self.length].contiguous(), final, states

    def get_kernel_source(self):
        return "\n\n".join(
            f"// Stage {i}\n" + kernel.get_kernel_source() for i, kernel in enumerate(self.kernels)
        )


build_training = functools.lru_cache(maxsize=64)(TrainingGLA)


class AscendGLAChunkFwdKernel(Kernel, GLAChunkFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call

    def forward(self, q, k, v, g, initial_state=None):
        c = self.call
        q, k, v, g = (x.contiguous() for x in (q, k, v, g))
        seeded = initial_state is not None
        state = (
            initial_state.contiguous()
            if seeded
            else torch.empty(
                (c.batch, c.heads, c.dim_k, c.dim_v), dtype=torch.float32, device=q.device
            )
        )
        args = (
            c.batch,
            c.seq_len,
            c.heads,
            c.dim_k,
            c.dim_v,
            str(c.dtype).removeprefix("torch."),
            c.scale,
            seeded,
            c.chunk_size,
        )
        if (
            c.dtype in (torch.float16, torch.bfloat16)
            and c.dim_k in (64, 128)
            and c.dim_v in (64, 128)
            and c.chunk_size % 32 == 0
        ):
            self.compiled = build_training(*args)
        else:
            self.compiled = build_recurrent_states(*args)
        return self.compiled(q, k, v, g, state)


if T is not None:
    from tileops.backend._registry import register_kernel_type

    register_kernel_type("GLAChunkFwdOp", "ascend", AscendGLAChunkFwdKernel)
