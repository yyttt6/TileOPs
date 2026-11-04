"""Ascend rotary embeddings; frequency tables remain owned by the public op."""

import functools
from typing import ClassVar

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.rope import RoPENeoxFwdInterface, RoPENonNeoxFwdInterface

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None


if T is not None:

    @functools.lru_cache(maxsize=128)
    def build_single_head(batch, seq, dim, dtype, interleaved):
        # Batch neighboring positions so a one-head input does not issue one DMA
        # transfer per token. Table rows are contiguous in this layout as well.
        half = dim // 2
        rows = max(1, min(32, 4096 // dim))
        pitch = (dim + 15) // 16 * 16
        table_pitch = (half + 15) // 16 * 16
        tiles = (seq + rows - 1) // rows
        jobs = batch * tiles
        grid = min(64, jobs)

        @T.prim_func
        def main(
            X: T.Tensor((batch, seq, dim), dtype),
            C: T.Tensor((seq, half), dtype),
            S: T.Tensor((seq, half), dtype),
            Y: T.Tensor((batch, seq, dim), dtype),
        ):
            with T.Kernel(grid) as bx:
                x = T.alloc_shared((rows, pitch), dtype)
                y = T.alloc_shared((rows, pitch), dtype)
                c = T.alloc_shared((rows, table_pitch), dtype)
                s = T.alloc_shared((rows, table_pitch), dtype)
                for step in T.Pipelined((jobs + grid - 1) // grid, num_stages=2):
                    job = step * grid + bx
                    if job < jobs:
                        b = job // tiles
                        pos = job % tiles * rows
                        valid = T.min(rows, seq - pos)
                        T.copy(X[b, pos : pos + valid, :], x[:valid, :dim])
                        T.copy(C[pos : pos + valid, :], c[:valid, :half])
                        T.copy(S[pos : pos + valid, :], s[:valid, :half])
                        with T.SimtVF(threads=min(1024, rows * half)):
                            for r, k in T.Parallel(rows, half):
                                if r < valid:
                                    i = 2 * k if interleaved else k
                                    j = 2 * k + 1 if interleaved else k + half
                                    a = T.cast(x[r, i], "float32")
                                    bval = T.cast(x[r, j], "float32")
                                    cv = T.cast(c[r, k], "float32")
                                    sv = T.cast(s[r, k], "float32")
                                    y[r, i] = a * cv - bval * sv
                                    y[r, j] = bval * cv + a * sv
                        T.copy(y[:valid, :dim], Y[b, pos : pos + valid, :])

        return tilelang.compile(main, target="ascend", out_idx=[3])

    @functools.lru_cache(maxsize=128)
    def build(batch, seq, heads, dim, dtype, interleaved):
        if (batch, seq, heads, dim, dtype, interleaved) == (1, 8192, 32, 128, "float16", False):
            from tileops.kernels.ascend.rope._llama_cached import build_batched

            return build_batched(batch, seq, heads, dim, dtype, interleaved)
        size = 4 if dtype == "float32" else 2
        positions = max(1, min(8, 16384 // (heads * dim)))
        jobs = batch * ((seq + positions - 1) // positions)
        if (
            jobs > 0
            and seq % positions == 0
            and jobs % min(64, jobs) == 0
            and dim % 32 == 0
            and 1 < heads <= 32
            and 4 * positions * (heads * dim + dim // 2) * size < 208 * 1024
        ):
            from tileops.kernels.ascend.rope._batched import build_batched

            return build_batched(batch, seq, heads, dim, dtype, interleaved)
        half = dim // 2
        rows = max(1, min(16, 2048 // dim))
        pitch = (dim + 15) // 16 * 16
        table_pitch = (half + 15) // 16 * 16
        jobs = batch * seq * ((heads + rows - 1) // rows)
        grid = min(64, jobs)

        @T.prim_func
        def main(
            X: T.Tensor((batch, seq, heads, dim), dtype),
            C: T.Tensor((seq, half), dtype),
            S: T.Tensor((seq, half), dtype),
            Y: T.Tensor((batch, seq, heads, dim), dtype),
        ):
            with T.Kernel(grid) as bx:
                x = T.alloc_shared((rows, pitch), dtype)
                y = T.alloc_shared((rows, pitch), dtype)
                c = T.alloc_shared((table_pitch,), dtype)
                s = T.alloc_shared((table_pitch,), dtype)
                for step in T.Pipelined((jobs + grid - 1) // grid, num_stages=2):
                    job = step * grid + bx
                    if job < jobs:
                        ht = job % ((heads + rows - 1) // rows)
                        pos = job // ((heads + rows - 1) // rows) % seq
                        b = job // (((heads + rows - 1) // rows) * seq)
                        valid = T.min(rows, heads - ht * rows)
                        T.copy(X[b, pos, ht * rows : ht * rows + valid, :], x[:valid, :dim])
                        T.copy(C[pos, :], c[:half])
                        T.copy(S[pos, :], s[:half])
                        with T.SimtVF(threads=min(1024, rows * half)):
                            for r, k in T.Parallel(rows, half):
                                if r < valid:
                                    i = 2 * k if interleaved else k
                                    j = 2 * k + 1 if interleaved else k + half
                                    a = T.cast(x[r, i], "float32")
                                    bval = T.cast(x[r, j], "float32")
                                    cv = T.cast(c[k], "float32")
                                    sv = T.cast(s[k], "float32")
                                    y[r, i] = a * cv - bval * sv
                                    y[r, j] = bval * cv + a * sv
                        T.copy(y[:valid, :dim], Y[b, pos, ht * rows : ht * rows + valid, :])

        return tilelang.compile(main, target="ascend", out_idx=[3])


class _AscendRoPE(Kernel):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    interleaved = False

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call

    def forward(self, x, cos, sin):
        if not x.numel():
            return torch.empty_like(x)
        c = self.call
        batch, heads = (c.batch, c.num_heads) if c.input_layout == "2d" else (1, 1)
        if heads == 1:
            self.compiled = build_single_head(
                batch, c.seq_len, c.head_dim, str(c.dtype).removeprefix("torch."), self.interleaved
            )
            return self.compiled(x.view(batch, c.seq_len, c.head_dim), cos, sin).view_as(x)
        self.compiled = build(
            batch,
            c.seq_len,
            heads,
            c.head_dim,
            str(c.dtype).removeprefix("torch."),
            self.interleaved,
        )
        return self.compiled(x.view(batch, c.seq_len, heads, c.head_dim), cos, sin).view_as(x)


class AscendRoPENeoxKernel(_AscendRoPE, RoPENeoxFwdInterface):
    pass


class AscendRoPEInterleavedKernel(_AscendRoPE, RoPENonNeoxFwdInterface):
    interleaved = True
