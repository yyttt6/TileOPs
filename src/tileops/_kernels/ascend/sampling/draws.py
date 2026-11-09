"""Philox token draws using chunk sums and a bounded local prefix scan."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend._registry import register_kernel_type
from tileops._kernels.ascend.sampling.chain_prepare import _prepare_parallel
from tileops._kernels.ascend.sampling.draw_reduce import _draw_tile, _weights_tile
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.sampling.call_spec import (
    ChainSpeculativeSamplingFwdInterface,
    SamplingFromProbsFwdInterface,
)

__all__ = ["AscendChainSpeculativeSamplingKernel", "AscendSamplingFromProbsKernel"]


@T.macro
def _uniform(seed, offset, row, draw, result):
    c0 = T.alloc_var("uint32", init=T.cast(draw, "uint32"))
    c1 = T.alloc_var("uint32", init=T.cast(row, "uint32"))
    c2 = T.alloc_var("uint32", init=T.cast(offset, "uint32"))
    c3 = T.alloc_var("uint32", init=T.cast(T.cast(offset, "uint64") >> 32, "uint32"))
    k0 = T.alloc_var("uint32", init=T.cast(seed, "uint32"))
    k1 = T.alloc_var("uint32", init=T.cast(T.cast(seed, "uint64") >> 32, "uint32"))
    for _step in T.serial(10):
        a = T.cast(c0, "uint64") * T.uint64(0xD2511F53)
        b = T.cast(c2, "uint64") * T.uint64(0xCD9E8D57)
        t0 = T.cast(b >> 32, "uint32") ^ c1 ^ k0
        t2 = T.cast(a >> 32, "uint32") ^ c3 ^ k1
        c0 = t0
        c1 = T.cast(b, "uint32")
        c2 = t2
        c3 = T.cast(a, "uint32")
        k0 = k0 + T.uint32(0x9E3779B9)
        k1 = k1 + T.uint32(0xBB67AE85)
    result[0] = T.cast(c0 >> 8, "float32") * T.float32(2.0**-24)


@functools.lru_cache(maxsize=256)
def _prepare(batch, vocab, drafts):
    blocks = min(batch, 64)

    @T.prim_func
    def main(
        D: T.Tensor((batch, max(1, drafts), vocab), "float32"),
        I: T.Tensor((batch, max(1, drafts)), "int32"),
        Q: T.Tensor((batch, drafts + 1, vocab), "float32"),
        S: T.Tensor((1,), "int64"),
        O: T.Tensor((1,), "int64"),
        A: T.Tensor((batch,), "int32"),
        U: T.Tensor((batch,), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            seed = T.alloc_shared((1,), "int64")
            offset = T.alloc_shared((1,), "int64")
            uniform = T.alloc_shared((8,), "float32")
            accepted = T.alloc_shared((8,), "int32")
            token = T.alloc_shared((8,), "int32")
            dp = T.alloc_shared((8,), "float32")
            qp = T.alloc_shared((8,), "float32")
            T.copy(S, seed)
            T.copy(O, offset)
            for r in T.serial(T.ceildiv(batch, blocks)):
                row = r * blocks + bx
                if row < batch:
                    count = T.alloc_var("int32", init=0)
                    if drafts > 0:
                        for j in T.serial(drafts):
                            if count == j:
                                _uniform(seed[0], offset[0], row, j, uniform)
                                T.copy(I[row, j : j + 1], token[:1])
                                idx = token[0]
                                T.copy(D[row, j, idx : idx + 1], dp[:1])
                                T.copy(Q[row, j, idx : idx + 1], qp[:1])
                                if uniform[0] * dp[0] < qp[0]:
                                    count = count + 1
                    _uniform(seed[0], offset[0], row, drafts, uniform)
                    accepted[0] = count
                    T.copy(accepted[:1], A[row : row + 1])
                    T.copy(uniform[:1], U[row : row + 1])

    return tilelang.compile(main, target="ascend", out_idx=[5, 6])


@functools.lru_cache(maxsize=256)
def _parts(batch, vocab, drafts, width):
    chunks = (vocab + width - 1) // width
    jobs = batch * chunks
    blocks = min(jobs, 64)

    @T.prim_func
    def main(
        D: T.Tensor((batch, max(1, drafts), vocab), "float32"),
        Q: T.Tensor((batch, drafts + 1, vocab), "float32"),
        A: T.Tensor((batch,), "int32"),
        W: T.Tensor((batch, chunks), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            x = T.alloc_shared((width,), "float32")
            d = T.alloc_shared((width,), "float32")
            a = T.alloc_shared((8,), "int32")
            out = T.alloc_shared((8,), "float32")
            for r in T.serial(T.ceildiv(jobs, blocks)):
                job = r * blocks + bx
                if job < jobs:
                    row = job // chunks
                    part = job % chunks
                    begin = part * width
                    count = T.min(width, vocab - begin)
                    if drafts > 0:
                        T.copy(A[row : row + 1], a[:1])
                    else:
                        a[0] = 0
                    stop = a[0]
                    T.copy(Q[row, stop, begin : begin + count], x[:count])
                    if drafts > 0:
                        if stop < drafts:
                            T.copy(D[row, stop, begin : begin + count], d[:count])
                            _weights_tile(x, d, out, count, True)
                        else:
                            _weights_tile(x, d, out, count, False)
                    else:
                        _weights_tile(x, d, out, count, False)
                    T.copy(out[:1], W[row, part : part + 1])

    return tilelang.compile(main, target="ascend", out_idx=[3])


@functools.lru_cache(maxsize=256)
def _select(batch, vocab, drafts, width):
    chunks = (vocab + width - 1) // width
    padded = max(64, tilelang.next_power_of_2(chunks))
    blocks = min(batch, 64)

    @T.prim_func
    def main(
        D: T.Tensor((batch, max(1, drafts), vocab), "float32"),
        I: T.Tensor((batch, max(1, drafts)), "int32"),
        Q: T.Tensor((batch, drafts + 1, vocab), "float32"),
        A: T.Tensor((batch,), "int32"),
        U: T.Tensor((batch,), "float32"),
        W: T.Tensor((batch, chunks), "float32"),
        S: T.Tensor((1,), "int64"),
        O: T.Tensor((1,), "int64"),
        Y: T.Tensor((batch, drafts + 1), "int32"),
    ):
        with T.Kernel(blocks) as bx:
            seed = T.alloc_shared((1,), "int64")
            offset = T.alloc_shared((1,), "int64")
            if drafts == 0:
                T.copy(S, seed)
                T.copy(O, offset)
            sums = T.alloc_shared((padded,), "float32")
            x = T.alloc_shared((1, width), "float32")
            d = T.alloc_shared((width,), "float32")
            a = T.alloc_shared((8,), "int32")
            u = T.alloc_shared((8,), "float32")
            result = T.alloc_shared((max(8, drafts + 1),), "int32")
            selected = T.alloc_shared((8,), "int32")
            ids = T.alloc_shared((max(8, drafts),), "int32")
            for r in T.serial(T.ceildiv(batch, blocks)):
                row = r * blocks + bx
                if row < batch:
                    if drafts > 0:
                        T.copy(A[row : row + 1], a[:1])
                        T.copy(U[row : row + 1], u[:1])
                    else:
                        a[0] = 0
                        _uniform(seed[0], offset[0], row, 0, u)
                    T.copy(W[row, :], sums[:chunks])
                    total = T.alloc_var("float32", init=0)
                    last_nonzero = T.alloc_var("int32", init=0)
                    for j in T.serial(chunks):
                        total = total + sums[j]
                        if sums[j] > 0:
                            last_nonzero = j
                    point = T.alloc_var("float32", init=u[0] * total)
                    chosen = T.alloc_var("int32", init=0)
                    for j in T.serial(chunks):
                        if chosen == j and j < chunks - 1 and point >= sums[j]:
                            point = point - sums[j]
                            chosen = chosen + 1
                    if sums[chosen] <= 0:
                        # Roundoff near the total can pass the last nonzero chunk.
                        chosen = last_nonzero
                        point = T.infinity("float32")
                    begin = chosen * width
                    count = T.min(width, vocab - begin)
                    stop = a[0]
                    T.copy(Q[row, stop, begin : begin + count], x[0, :count])
                    if drafts > 0:
                        if stop < drafts:
                            T.copy(D[row, stop, begin : begin + count], d[:count])
                            _draw_tile(x, d, selected, count, point, begin, True)
                        else:
                            _draw_tile(x, d, selected, count, point, begin, False)
                    else:
                        _draw_tile(x, d, selected, count, point, begin, False)
                    if drafts > 0:
                        T.copy(I[row, :], ids[:drafts])
                        for j in T.serial(drafts + 1):
                            if j < stop:
                                result[j] = ids[j]
                            elif j == stop:
                                result[j] = selected[0]
                            else:
                                result[j] = -1
                    else:
                        result[0] = selected[0]
                    T.copy(result[: drafts + 1], Y[row, :])

    return tilelang.compile(main, target="ascend", out_idx=[8])


class _DrawKernel(Kernel):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def refusal(cls, call):
        if call.batch * (call.num_draft + 1) * call.vocab > 2**31 - 1:
            return "probability offsets require B * (N + 1) * V <= 2**31 - 1"
        if call.batch and (call.num_draft > 8192 or call.vocab > 2**26):
            return "unified-buffer scratch requires N <= 8192 and V <= 2**26"
        return None

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call
        if call.batch:
            width = min(8192, max(64, tilelang.next_power_of_2(call.vocab)))
            if call.num_draft:
                prepare = _prepare_parallel if call.num_draft <= 1024 else _prepare
                self.prepare = prepare(call.batch, call.vocab, call.num_draft)
            self.parts = _parts(call.batch, call.vocab, call.num_draft, width)
            self.select = _select(call.batch, call.vocab, call.num_draft, width)

    def _run(self, draft, ids, target, seed, offset):
        if self.call.num_draft:
            accepted, uniform = self.prepare(draft, ids, target, seed, offset)
        else:
            accepted = torch.empty((self.call.batch,), device=target.device, dtype=torch.int32)
            uniform = torch.empty((self.call.batch,), device=target.device, dtype=torch.float32)
        sums = self.parts(draft, target, accepted)
        result = self.select(draft, ids, target, accepted, uniform, sums, seed, offset)
        return result, accepted


class AscendSamplingFromProbsKernel(_DrawKernel, SamplingFromProbsFwdInterface):
    def forward(self, probs, seed, offset):
        if not self.call.batch:
            return torch.empty((0,), device=probs.device, dtype=torch.int32)
        dummy = torch.empty((self.call.batch, 1), device=probs.device, dtype=torch.int32)
        rows = probs.view(self.call.batch, 1, self.call.vocab)
        return self._run(rows, dummy, rows, seed, offset)[0].view(self.call.batch)


class AscendChainSpeculativeSamplingKernel(_DrawKernel, ChainSpeculativeSamplingFwdInterface):
    def forward(self, draft_probs, draft_token_ids, target_probs, seed, offset):
        if not self.call.batch:
            return torch.empty(
                (0, self.call.num_draft + 1), device=target_probs.device, dtype=torch.int32
            ), torch.empty((0,), device=target_probs.device, dtype=torch.int32)
        return self._run(draft_probs, draft_token_ids, target_probs, seed, offset)


register_kernel_type("SamplingFromProbsFwdOp", "ascend", AscendSamplingFromProbsKernel)
register_kernel_type(
    "ChainSpeculativeSamplingFwdOp", "ascend", AscendChainSpeculativeSamplingKernel
)
