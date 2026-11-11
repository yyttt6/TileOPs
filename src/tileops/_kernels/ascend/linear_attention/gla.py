"""GLA inference with FP32 recurrent state and independent value-column tiles."""

import functools
from typing import ClassVar

import torch

from tileops._kernels.kernel_base import Kernel
from tileops._kernels.linear_attention.gla.call_spec import GLAFwdInterface

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None

if T is not None:

    def _pick(index, values):
        result = T.int32(values[-1])
        for i in range(len(values) - 2, -1, -1):
            result = T.if_then_else(index == i, T.int32(values[i]), result)
        return result

    @functools.lru_cache(maxsize=128)
    def build(batch, length, heads, dk, dv, dtype, scale, initial, offsets):
        packed = bool(offsets)
        sequences = len(offsets) - 1 if packed else batch
        columns = 16 if sequences * heads < 16 else 32
        kp = (dk + 15) // 16 * 16
        keys = tilelang.next_power_of_2(dk)
        tiles = (dv + columns - 1) // columns
        jobs = sequences * heads * tiles
        grid = min(64, jobs)
        starts = offsets[:-1] if packed else (0,) * sequences
        ends = offsets[1:] if packed else (length,) * sequences

        @T.prim_func
        def main(
            Q: T.Tensor((batch, length, heads, dk), dtype),
            K: T.Tensor((batch, length, heads, dk), dtype),
            V: T.Tensor((batch, length, heads, dv), dtype),
            G: T.Tensor((batch, length, heads, dk), dtype),
            H0: T.Tensor((sequences, heads, dk, dv), "float32"),
            O: T.Tensor((batch, length, heads, dv), dtype),
            HT: T.Tensor((sequences, heads, dk, dv), "float32"),
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
                        first = _pick(seq, starts)
                        last = _pick(seq, ends)
                        if initial:
                            T.copy(H0[seq, head, :, col : col + valid], state[:dk, :valid])
                        else:
                            with T.SimtVF(threads=256):
                                for i, j in T.Parallel(kp, columns):
                                    state[i, j] = 0.0
                        for token in T.serial(first, last):
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
                        T.copy(state[:dk, :valid], HT[seq, head, :, col : col + valid])

        return tilelang.compile(main, target="ascend", out_idx=[5, 6])

    @functools.lru_cache(maxsize=128)
    def build_decode(batch, heads, dk, dv, dtype, scale, initial, workers=1024):
        columns = min(128, tilelang.next_power_of_2(dv)) if batch * heads >= 32 else 32
        keys = tilelang.next_power_of_2(dk)
        kp = (dk + 15) // 16 * 16
        tiles = (dv + columns - 1) // columns
        jobs = batch * heads * tiles
        grid = min(64, jobs)
        precompute = initial and batch * heads >= 64

        @T.prim_func
        def main(
            Q: T.Tensor((batch * heads, dk), dtype),
            K: T.Tensor((batch * heads, dk), dtype),
            V: T.Tensor((batch * heads, dv), dtype),
            G: T.Tensor((batch * heads, dk), dtype),
            H0: T.Tensor((batch * heads, dk, dv), "float32"),
            O: T.Tensor((batch * heads, dv), dtype),
            HT: T.Tensor((batch * heads, dk, dv), "float32"),
        ):
            with T.Kernel(grid) as bx:
                q = T.alloc_shared((kp,), dtype)
                k = T.alloc_shared((kp,), dtype)
                g = T.alloc_shared((kp,), dtype)
                v = T.alloc_shared((columns,), dtype)
                state = T.alloc_shared((kp, columns), "float32")
                out = T.alloc_shared((columns,), dtype)
                decay = T.alloc_shared((kp,), "float32")
                for step in T.Pipelined((jobs + grid - 1) // grid, num_stages=2):
                    job = step * grid + bx
                    if job < jobs:
                        head = job // tiles
                        col = job % tiles * columns
                        valid = T.min(columns, dv - col)
                        T.copy(Q[head, :], q[:dk])
                        T.copy(K[head, :], k[:dk])
                        T.copy(G[head, :], g[:dk])
                        T.copy(V[head, col : col + valid], v[:valid])
                        if initial:
                            T.copy(H0[head, :, col : col + valid], state[:dk, :valid])
                        if precompute:
                            with T.SimtVF(threads=128):
                                for i in T.Parallel(dk):
                                    decay[i] = T.exp(T.cast(g[i], "float32"))
                        with T.SimtVF(threads=min(workers, keys * columns)):
                            total = T.alloc_reducer((columns,), "float32", op="sum")
                            T.reducer_init(total)
                            for i, j in T.Parallel(keys, columns):
                                if i < dk and j < valid:
                                    previous = state[i, j] if initial else T.float32(0)
                                    gate = (
                                        decay[i] if precompute else T.exp(T.cast(g[i], "float32"))
                                    )
                                    value = gate * previous + T.cast(k[i], "float32") * T.cast(
                                        v[j], "float32"
                                    )
                                    state[i, j] = value
                                    T.reducer_update(total[j], T.cast(q[i], "float32") * value)
                            result = T.alloc_fragment((columns,), "float32")
                            T.finalize_reducer(total, result)
                            for j in T.Parallel(columns):
                                out[j] = result[j] * scale
                        T.copy(state[:dk, :valid], HT[head, :, col : col + valid])
                        T.copy(out[:valid], O[head, col : col + valid])

        return tilelang.compile(main, target="ascend", out_idx=[5, 6])


class AscendGLAKernel(Kernel, GLAFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call

    def forward(self, q, k, v, g, initial_state=None, cu_seqlens=None, cu_seqlens_cpu=None):
        c = self.call
        offsets = ()
        if cu_seqlens is not None:
            # This is sequence metadata, never arithmetic tensor contents. Re-read
            # every call so changing offsets cannot reuse an obsolete specialization.
            offsets = tuple(int(x) for x in cu_seqlens.cpu().tolist())
            if cu_seqlens_cpu is not None and tuple(cu_seqlens_cpu.tolist()) != offsets:
                raise ValueError("cu_seqlens_cpu must match cu_seqlens")
            if (
                offsets[0] != 0
                or offsets[-1] != c.seq_len
                or any(a > b for a, b in zip(offsets, offsets[1:], strict=False))
            ):
                raise ValueError(
                    "cu_seqlens must be nondecreasing prefix offsets covering the sequence"
                )
        if c.seq_len == 1 and not offsets:
            self.compiled = build_decode(
                c.batch,
                c.heads,
                c.dim_k,
                c.dim_v,
                str(c.dtype).removeprefix("torch."),
                c.scale,
                initial_state is not None,
                512 if initial_state is not None and c.batch * c.heads >= 64 else 1024,
            )
            bh = c.batch * c.heads
            state = initial_state
            if state is None:
                state = torch.empty((bh, c.dim_k, c.dim_v), dtype=torch.float32, device=q.device)
            output, final = self.compiled(
                q.view(bh, c.dim_k),
                k.view(bh, c.dim_k),
                v.view(bh, c.dim_v),
                g.view(bh, c.dim_k),
                state.view(bh, c.dim_k, c.dim_v),
            )
            return output.view(c.batch, 1, c.heads, c.dim_v), final.view(
                c.batch, c.heads, c.dim_k, c.dim_v
            )
        if (
            offsets
            and all(b > a for a, b in zip(offsets, offsets[1:], strict=False))
            and c.dim_k in (64, 128)
            and c.dim_v in (64, 128)
            and c.dtype in (torch.float16, torch.bfloat16)
        ):
            from tileops._kernels.ascend.linear_attention.gla_chunked import build_packed_chunked

            lengths = tuple(b - a for a, b in zip(offsets, offsets[1:], strict=False))
            self.compiled = build_packed_chunked(
                lengths,
                c.heads,
                c.dim_k,
                c.dim_v,
                str(c.dtype).removeprefix("torch."),
                c.scale,
                initial_state is not None,
            )
            if initial_state is None:
                initial_state = torch.empty(
                    (c.num_sequences, c.heads, c.dim_k, c.dim_v),
                    dtype=torch.float32,
                    device=q.device,
                )
            return self.compiled(q, k, v, g, initial_state)
        if (
            not offsets
            and c.seq_len > 1
            and c.dim_k in (64, 128)
            and c.dim_v in (64, 128)
            and c.dtype in (torch.float16, torch.bfloat16)
        ):
            from tileops._kernels.ascend.linear_attention.gla_chunked import build_chunked

            self.compiled = build_chunked(
                c.batch,
                c.seq_len,
                c.heads,
                c.dim_k,
                c.dim_v,
                str(c.dtype).removeprefix("torch."),
                c.scale,
                initial_state is not None,
            )
            if initial_state is None:
                initial_state = torch.empty(
                    (c.batch, c.heads, c.dim_k, c.dim_v), dtype=torch.float32, device=q.device
                )
            return self.compiled(q, k, v, g, initial_state)
        self.compiled = build(
            c.batch,
            c.seq_len,
            c.heads,
            c.dim_k,
            c.dim_v,
            str(c.dtype).removeprefix("torch."),
            c.scale,
            initial_state is not None,
            offsets,
        )
        if initial_state is None:
            initial_state = torch.empty(
                (c.num_sequences, c.heads, c.dim_k, c.dim_v), dtype=torch.float32, device=q.device
            )
        return self.compiled(q, k, v, g, initial_state)


if T is not None:
    from tileops.backend._registry import register_kernel_type

    register_kernel_type("GLAFwdOp", "ascend", AscendGLAKernel)
