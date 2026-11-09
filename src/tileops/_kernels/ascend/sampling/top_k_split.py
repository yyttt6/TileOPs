"""Byte-radix histograms split wide top-k rows across vector cores."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.ascend.reduction.cumsum import _scan
from tileops._kernels.ascend.sampling.masks import AscendTopPMaskKernel, _key
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.sampling.call_spec import TopKMaskFwdInterface, TopKTopPMaskFwdInterface
from tileops.backend._registry import register_kernel_type

__all__ = ["AscendTopKSplitKernel", "AscendTopKTopPSplitKernel"]


@T.macro
def _histogram(x, state, hist, count, dtype: str, shift: int):
    with T.SimdVF():
        full = T.simd.pset(32)
        half_mask = T.simd.pset(16)
        low = T.simd.alloc_local(1, "uint16")
        high = T.simd.alloc_local(1, "uint16")
        low[0] = T.simd.vdup(T.uint16(0), "uint16")
        high[0] = T.simd.vdup(T.uint16(0), "uint16")
        prefix = T.simd.vld(state[0], dist="BRC_B32")
        for j in T.serial((count + 63) // 64):
            valid = T.simd.update_mask(T.min(64, count - j * 64), 32)
            value = (
                T.simd.vld(x[j * 64])
                if dtype == "float32"
                else T.simd.vcvt(T.simd.vld(x[j * 64], dist="UNPK_B16"), "float32", part=0)
            )
            value = T.simd.vsel(
                T.simd.vdup(T.float32(0), "float32"),
                value,
                T.simd.vcmps(value, T.float32(0), full, op="eq"),
            )
            raw = T.reinterpret(value, "uint32x64")
            signed = T.reinterpret(value, "int32x64")
            negative = T.simd.vcmps(signed, T.int32(0), full, op="lt")
            flip = T.simd.vsel(
                T.simd.vdup(T.uint32(0xFFFFFFFF), "uint32"),
                T.simd.vdup(T.uint32(0x80000000), "uint32"),
                negative,
            )
            key = T.simd.vxor(raw, flip, full)
            absolute = T.simd.vand(raw, T.simd.vdup(T.uint32(0x7FFFFFFF), "uint32"), full)
            key = T.simd.vsel(
                T.simd.vdup(T.uint32(0xFFFFFFFF), "uint32"),
                key,
                T.simd.vcmps(absolute, T.uint32(0x7F800000), full, op="gt"),
            )
            if shift < 24:
                upper = T.simd.vand(
                    key,
                    T.simd.vdup(T.uint32((0xFFFFFFFF << (shift + 8)) & 0xFFFFFFFF), "uint32"),
                    full,
                )
                valid = T.simd.pand(valid, T.simd.vcmp(upper, prefix, full, op="eq"), full)
            byte = T.simd.vand(
                T.simd.vshrs(key, shift, full), T.simd.vdup(T.uint32(255), "uint32"), full
            )
            # A b32 predicate selects only byte zero of each reinterpreted word.
            T.simd.dhistv2(low[0], T.reinterpret(byte, "uint8x256"), valid, bin=0)
            T.simd.dhistv2(high[0], T.reinterpret(byte, "uint8x256"), valid, bin=1)
        T.simd.vsts(hist[0], low[0], half_mask, extent=128)
        T.simd.vsts(hist[128], high[0], half_mask, extent=128)


@functools.lru_cache(maxsize=128)
def _hist(batch, vocab, dtype, width, shift):
    chunks = (vocab + width - 1) // width
    jobs = batch * chunks
    blocks = min(jobs, 64)

    @T.prim_func
    def main(
        X: T.Tensor((batch, vocab), dtype),
        S: T.Tensor((batch, 2), "uint32"),
        H: T.Tensor((batch, chunks, 256), "uint16"),
    ):
        with T.Kernel(blocks) as bx:
            x = T.alloc_shared((width,), dtype)
            state = T.alloc_shared((8,), "uint32")
            hist = T.alloc_shared((256,), "uint16")
            for step in T.serial(T.ceildiv(jobs, blocks)):
                job = step * blocks + bx
                if job < jobs:
                    row = job // chunks
                    chunk = job % chunks
                    start = chunk * width
                    count = T.min(width, vocab - start)
                    if shift < 24:
                        T.copy(S[row, :], state[:2])
                    else:
                        T.fill(state, 0)
                    T.copy(X[row, start : start + count], x[:count])
                    _histogram(x, state, hist, count, dtype, shift)
                    T.copy(hist, H[row, chunk, :])

    return tilelang.compile(main, target="ascend", out_idx=[2])


@functools.lru_cache(maxsize=128)
def _choose(batch, chunks, shift):
    blocks = min(batch, 64)

    @T.prim_func
    def main(
        H: T.Tensor((batch, chunks, 256), "uint16"),
        K: T.Tensor((batch,), "int32"),
        S: T.Tensor((batch, 2), "uint32"),
        O: T.Tensor((batch, 2), "uint32"),
    ):
        with T.Kernel(blocks) as bx:
            hist = T.alloc_shared((chunks, 256), "uint16")
            counts = T.alloc_shared((1, 256), "float32")
            prefix = T.alloc_shared((1, 256), "float32")
            carry = T.alloc_shared((1, 8), "float32")
            state = T.alloc_shared((8,), "uint32")
            k = T.alloc_shared((8,), "int32")
            out = T.alloc_shared((8,), "uint32")
            for step in T.serial(T.ceildiv(batch, blocks)):
                row = step * blocks + bx
                if row < batch:
                    T.copy(H[row, :, :], hist)
                    if shift < 24:
                        T.copy(S[row, :], state[:2])
                    else:
                        T.copy(K[row : row + 1], k[:1])
                        state[0] = T.uint32(0)
                        state[1] = T.cast(k[0], "uint32")
                    with T.SimtVF(threads=256):
                        totals = T.alloc_fragment((256,), "float32")
                        for i in T.Parallel(256):
                            totals[i] = T.float32(0)
                            for part in T.serial(chunks):
                                totals[i] = totals[i] + T.cast(hist[part, 255 - i], "float32")
                            counts[0, i] = totals[i]
                    T.fill(carry, 0)
                    _scan(counts, prefix, carry, 1, 256, "float32")
                    with T.SimtVF(threads=256):
                        index = T.alloc_reducer((1,), "int32", op="min")
                        T.reducer_init(index)
                        for i in T.Parallel(256):
                            if prefix[0, i] >= T.cast(state[1], "float32"):
                                T.reducer_update(index[0], i)
                        found = T.alloc_fragment((1,), "int32")
                        T.finalize_reducer(index, found)
                        loc = T.min(found[0], 255)
                        out[0] = state[0] | (T.cast(255 - loc, "uint32") << shift)
                        out[1] = state[1] - T.cast(prefix[0, loc] - counts[0, loc], "uint32")
                    T.copy(out[:2], O[row, :])

    return tilelang.compile(main, target="ascend", out_idx=[3])


@functools.lru_cache(maxsize=128)
def _write(batch, vocab, dtype, width):
    chunks = (vocab + width - 1) // width
    jobs = batch * chunks
    blocks = min(jobs, 64)

    @T.prim_func
    def main(
        X: T.Tensor((batch, vocab), dtype),
        K: T.Tensor((batch,), "int32"),
        S: T.Tensor((batch, 2), "uint32"),
        Y: T.Tensor((batch, vocab), dtype),
    ):
        with T.Kernel(blocks) as bx:
            x = T.alloc_shared((width,), dtype)
            y = T.alloc_shared((width,), dtype)
            state = T.alloc_shared((8,), "uint32")
            k = T.alloc_shared((8,), "int32")
            for step in T.serial(T.ceildiv(jobs, blocks)):
                job = step * blocks + bx
                if job < jobs:
                    row = job // chunks
                    start = (job % chunks) * width
                    count = T.min(width, vocab - start)
                    T.copy(S[row, :], state[:2])
                    T.copy(K[row : row + 1], k[:1])
                    T.copy(X[row, start : start + count], x[:count])
                    with T.SimtVF(threads=256):
                        for i in T.Parallel(width):
                            if i < count:
                                key = _key(x[i], 0)
                                keep = k[0] >= vocab or (
                                    key >= state[0]
                                    and not T.isnan(T.cast(x[i], "float32"))
                                    and state[0] < T.uint32(0xFFC00000)
                                )
                                y[i] = T.if_then_else(
                                    keep, x[i], T.cast(-T.infinity("float32"), dtype)
                                )
                    T.copy(y[:count], Y[row, start : start + count])

    return tilelang.compile(main, target="ascend", out_idx=[3])


class AscendTopKSplitKernel(Kernel, TopKMaskFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    preferred_over: ClassVar[frozenset[str]] = frozenset({"ascend"})

    @classmethod
    def refusal(cls, call):
        if not (call.batch > 0 and 4096 <= call.vocab <= 1 << 20):
            return "split histograms require nonempty batches and 4096 <= V <= 2**20"
        if call.batch * call.vocab > 2**31 - 1:
            return "logit offsets require B * V <= 2**31 - 1"
        return None

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call
        width = 8192
        chunks = (call.vocab + width - 1) // width
        dtype = str(call.dtype).split(".")[-1]
        shifts = (24, 16) if dtype == "bfloat16" else (24, 16, 8, 0)
        self.stages = [
            (_hist(call.batch, call.vocab, dtype, width, s), _choose(call.batch, chunks, s))
            for s in shifts
        ]
        self.write = _write(call.batch, call.vocab, dtype, width)

    def forward(self, logits, k):
        state = torch.empty((self.call.batch, 2), device=logits.device, dtype=torch.uint32)
        for histogram, choose in self.stages:
            state = choose(histogram(logits, state), k, state)
        return self.write(logits, k, state)


register_kernel_type("TopKMaskFwdOp", "ascend_histogram", AscendTopKSplitKernel)


class AscendTopKTopPSplitKernel(Kernel, TopKTopPMaskFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    preferred_over: ClassVar[frozenset[str]] = frozenset({"ascend"})

    @classmethod
    def refusal(cls, call):
        if call.batch >= 16:
            return "split top-k followed by a row nucleus serves batches below 16"
        return AscendTopKSplitKernel.refusal(call)

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.top_k = AscendTopKSplitKernel(call)
        self.top_p = AscendTopPMaskKernel(call)

    def forward(self, logits, k, p):
        return self.top_p(self.top_k(logits, k), p)


register_kernel_type("TopKTopPMaskFwdOp", "ascend_histogram", AscendTopKTopPSplitKernel)
