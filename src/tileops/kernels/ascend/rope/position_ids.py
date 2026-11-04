"""Packed NeoX rotary embeddings, including partial rotation and range reporting."""

import functools
from typing import ClassVar

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.rope import RoPENeoxPositionIdsFwdInterface

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None


if T is not None:

    @functools.lru_cache(maxsize=128)
    def gather_tables(tokens, half, max_position, dtype):
        rows = 16
        jobs = (tokens + rows - 1) // rows
        grid = min(64, jobs)

        @T.prim_func
        def main(
            C: T.Tensor((max_position, half), dtype),
            S: T.Tensor((max_position, half), dtype),
            P: T.Tensor((tokens,), "int32"),
            OutC: T.Tensor((tokens, half), dtype),
            OutS: T.Tensor((tokens, half), dtype),
            Invalid: T.Tensor((tokens,), "int32"),
        ):
            with T.Kernel(grid) as bx:  # noqa: SIM117 - preserve TileLang scope nesting
                with T.SimtVF(threads=min(1024, rows * half)):
                    for step in T.serial((jobs + grid - 1) // grid):
                        for r, k in T.Parallel(rows, half):
                            token = (step * grid + bx) * rows + r
                            if token < tokens:
                                pos = P[token]
                                safe = T.max(0, T.min(max_position - 1, pos))
                                OutC[token, k] = C[safe, k]
                                OutS[token, k] = S[safe, k]
                                if k == 0:
                                    Invalid[token] = T.cast(pos < 0 or pos >= max_position, "int32")

        return tilelang.compile(main, target="ascend", out_idx=[3, 4, 5])

    @functools.lru_cache(maxsize=128)
    def build(tokens, heads, dim, rotary, max_position, dtype):
        rows = max(1, min(16, 2048 // dim))
        half = rotary // 2
        pitch = (dim + 15) // 16 * 16
        table_pitch = (half + 15) // 16 * 16
        head_tiles = (heads + rows - 1) // rows
        jobs = tokens * head_tiles
        grid = min(64, jobs)

        @T.prim_func
        def main(
            X: T.Tensor((tokens, heads, dim), dtype),
            C: T.Tensor((max_position, half), dtype),
            S: T.Tensor((max_position, half), dtype),
            P: T.Tensor((tokens,), "int32"),
            Y: T.Tensor((tokens, heads, dim), dtype),
            Invalid: T.Tensor((tokens,), "int32"),
        ):
            with T.Kernel(grid) as bx:
                x = T.alloc_shared((rows, pitch), dtype)
                y = T.alloc_shared((rows, pitch), dtype)
                c = T.alloc_shared((table_pitch,), dtype)
                s = T.alloc_shared((table_pitch,), dtype)
                for step in T.Pipelined((jobs + grid - 1) // grid, num_stages=2):
                    job = step * grid + bx
                    if job < jobs:
                        token = job // head_tiles
                        h = (job % head_tiles) * rows
                        valid = T.min(rows, heads - h)
                        T.copy(X[token, h : h + valid, :], x[:valid, :dim])
                        with T.SimtVF(threads=32):
                            for k in T.Parallel(table_pitch):
                                pos = P[token]
                                safe = T.max(0, T.min(max_position - 1, pos))
                                if k < half:
                                    c[k] = C[safe, k]
                                    s[k] = S[safe, k]
                                if k == 0 and h == 0:
                                    Invalid[token] = T.cast(pos < 0 or pos >= max_position, "int32")
                        with T.SimtVF(threads=min(1024, rows * dim)):
                            for r, k in T.Parallel(rows, dim):
                                if r < valid:
                                    if k < rotary:
                                        peer = T.if_then_else(k < half, k + half, k - half)
                                        sign = T.if_then_else(k < half, T.float32(-1), T.float32(1))
                                        y[r, k] = T.cast(x[r, k], "float32") * T.cast(
                                            c[k % half], "float32"
                                        ) + sign * T.cast(x[r, peer], "float32") * T.cast(
                                            s[k % half], "float32"
                                        )
                                    else:
                                        y[r, k] = x[r, k]
                        T.copy(y[:valid, :dim], Y[token, h : h + valid, :])

        return tilelang.compile(main, target="ascend", out_idx=[4, 5])


class AscendRoPEPositionIdsKernel(Kernel, RoPENeoxPositionIdsFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        self.invalid = None

    def forward(self, x, cos, sin, position_ids):
        c = self.call
        if not x.numel():
            self.invalid = None
            return torch.empty_like(x)
        if c.rotary_dim == c.head_dim and c.num_tokens >= 128:
            # Gather each token's table row once. Indirect GM accesses in every
            # head tile otherwise dominate the contiguous rotation's bandwidth.
            from tileops.kernels.ascend.rope.rope import build as rotate
            from tileops.kernels.ascend.rope.rope import build_single_head

            dtype = str(c.dtype).removeprefix("torch.")
            self.finish = gather_tables(c.num_tokens, c.rotary_dim // 2, c.max_position, dtype)
            gathered_cos, gathered_sin, self.invalid = self.finish(cos, sin, position_ids)
            if c.num_heads == 1:
                self.compiled = build_single_head(1, c.num_tokens, c.head_dim, dtype, False)
                return self.compiled(
                    x.view(1, c.num_tokens, c.head_dim), gathered_cos, gathered_sin
                ).view_as(x)
            self.compiled = rotate(1, c.num_tokens, c.num_heads, c.head_dim, dtype, False)
            return self.compiled(
                x.view(1, c.num_tokens, c.num_heads, c.head_dim), gathered_cos, gathered_sin
            ).view_as(x)
        self.compiled = build(
            c.num_tokens,
            c.num_heads,
            c.head_dim,
            c.rotary_dim,
            c.max_position,
            str(c.dtype).removeprefix("torch."),
        )
        output, self.invalid = self.compiled(x, cos, sin, position_ids)
        return output

    def take_out_of_range(self):
        invalid, self.invalid = self.invalid, None
        return invalid is not None and bool(invalid.cpu().any())
