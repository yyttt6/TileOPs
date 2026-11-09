"""Split-row maximum and parallel min-p masking for wide vocabularies."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend._registry import register_kernel_type
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.sampling.call_spec import MinPMaskFwdInterface

__all__ = ["AscendMinPMaskKernel"]


@T.macro
def _maximum_tile(x, out, count, dtype: str):
    with T.SimdVF():
        mask = T.simd.pset(32)
        one = T.simd.pset(32, "PAT_VL1")
        acc = T.simd.alloc_local(1, "int32")
        acc[0] = T.simd.vdup(T.int32(-2147483648), "int32")
        for j in T.serial((count + 63) // 64):
            valid = T.simd.update_mask(T.min(64, count - j * 64), 32)
            value = (
                T.simd.vld(x[j * 64])
                if dtype == "float32"
                else T.simd.vcvt(T.simd.vld(x[j * 64], dist="UNPK_B16"), "float32", part=0)
            )
            raw = T.reinterpret(value, "int32x64")
            negative = T.simd.vcmps(raw, T.int32(0), mask, op="lt")
            flipped = T.simd.vxor(raw, T.simd.vdup(T.int32(2147483647), "int32"), mask)
            key = T.simd.vsel(flipped, raw, negative)
            magnitude = T.simd.vand(raw, T.simd.vdup(T.int32(2147483647), "int32"), mask)
            key = T.simd.vsel(
                T.simd.vdup(T.int32(2147483647), "int32"),
                key,
                T.simd.vcmps(magnitude, T.int32(2139095040), mask, op="gt"),
            )
            key = T.simd.vsel(key, T.simd.vdup(T.int32(-2147483648), "int32"), valid)
            acc[0] = T.simd.vmax(acc[0], key, mask)
        total = T.simd.vcmax(acc[0], mask)
        T.simd.vsts(out[0], total, one, extent=1)


@T.macro
def _mask_tile(x, y, m, p, count, chunks: int, dtype: str):
    with T.SimdVF():
        mask = T.simd.pset(32)
        acc = T.simd.alloc_local(1, "int32")
        acc[0] = T.simd.vdup(T.int32(-2147483648), "int32")
        for j in T.serial((chunks + 63) // 64):
            valid = T.simd.update_mask(T.min(64, chunks - j * 64), 32)
            value = T.simd.vsel(
                T.simd.vld(m[j * 64]), T.simd.vdup(T.int32(-2147483648), "int32"), valid
            )
            acc[0] = T.simd.vmax(acc[0], value, mask)
        key = T.simd.vdupv(T.simd.vcmax(acc[0], mask), mask)
        raw = T.simd.vsel(
            T.simd.vxor(key, T.simd.vdup(T.int32(2147483647), "int32"), mask),
            key,
            T.simd.vcmps(key, T.int32(0), mask, op="lt"),
        )
        maximum = T.reinterpret(raw, "float32x64")
        probability = T.simd.vld(p[0], dist="BRC_B32")
        threshold = T.simd.vadd(maximum, T.simd.vln(probability, mask, precision="ftz_false"), mask)
        for j in T.serial((count + 63) // 64):
            raw = (
                T.simd.vld(x[j * 64])
                if dtype == "float32"
                else T.simd.vld(x[j * 64], dist="UNPK_B16")
            )
            value = raw if dtype == "float32" else T.simd.vcvt(raw, "float32", part=0)
            removed = T.simd.vcmp(value, threshold, mask, op="lt")
            if dtype == "float32":
                result = T.simd.vsel(T.simd.vdup(-T.infinity("float32"), "float32"), raw, removed)
                T.simd.vsts(y[j * 64], result, mask, extent=64)
            else:
                # Select storage bits so retained NaN payloads and signed zeros never narrow.
                negative_infinity = 0xFC00 if dtype == "float16" else 0xFF80
                result_bits = T.simd.vsel(
                    T.simd.vdup(T.int32(negative_infinity), "int32"),
                    T.reinterpret(raw, "int32x64"),
                    removed,
                )
                result = T.reinterpret(
                    result_bits, "float16x128" if dtype == "float16" else "bfloat16x128"
                )
                T.simd.vsts(y[j * 64], result, mask, dist="PK_B32", extent=64)


@functools.lru_cache(maxsize=256)
def _maximum(batch, vocab, dtype, width):
    chunks = (vocab + width - 1) // width
    jobs = batch * chunks
    blocks = min(64, jobs)

    @T.prim_func
    def main(X: T.Tensor((batch, vocab), dtype), M: T.Tensor((batch, chunks), "int32")):
        with T.Kernel(blocks) as bx:
            x = T.alloc_shared((width,), dtype)
            out = T.alloc_shared((8,), "int32")
            for step in T.serial(T.ceildiv(jobs, blocks)):
                job = step * blocks + bx
                if job < jobs:
                    row = job // chunks
                    chunk = job % chunks
                    start = chunk * width
                    count = T.min(width, vocab - start)
                    T.copy(X[row, start : start + count], x[:count])
                    _maximum_tile(x, out, count, dtype)
                    T.copy(out[:1], M[row, chunk : chunk + 1])

    return tilelang.compile(main, target="ascend", out_idx=[1])


@functools.lru_cache(maxsize=256)
def _mask(batch, vocab, dtype, width):
    chunks = (vocab + width - 1) // width
    padded = max(64, tilelang.next_power_of_2(chunks))
    jobs = batch * chunks
    blocks = min(64, jobs)

    @T.prim_func
    def main(
        X: T.Tensor((batch, vocab), dtype),
        P: T.Tensor((batch,), "float32"),
        M: T.Tensor((batch, chunks), "int32"),
        Y: T.Tensor((batch, vocab), dtype),
    ):
        with T.Kernel(blocks) as bx:
            x = T.alloc_shared((width,), dtype)
            y = T.alloc_shared((width,), dtype)
            m = T.alloc_shared((padded,), "int32")
            p = T.alloc_shared((8,), "float32")
            for step in T.serial(T.ceildiv(jobs, blocks)):
                job = step * blocks + bx
                if job < jobs:
                    row = job // chunks
                    chunk = job % chunks
                    start = chunk * width
                    count = T.min(width, vocab - start)
                    T.copy(M[row, :], m[:chunks])
                    T.copy(P[row : row + 1], p[:1])
                    T.copy(X[row, start : start + count], x[:count])
                    _mask_tile(x, y, m, p, count, chunks, dtype)
                    T.copy(y[:count], Y[row, start : start + count])

    return tilelang.compile(main, target="ascend", out_idx=[3])


class AscendMinPMaskKernel(Kernel, MinPMaskFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def refusal(cls, call):
        if call.batch * call.vocab > 2**31 - 1:
            return "logit offsets require B * V <= 2**31 - 1"
        if call.batch and call.vocab > 2**28:
            return "unified-buffer partial maxima require V <= 2**28"
        return None

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call
        if call.batch:
            # Small batches split more finely to occupy all vector cores.
            width = min(
                8192, max(1024, tilelang.next_power_of_2(call.vocab * min(call.batch, 64) // 64))
            )
            dtype = str(call.dtype).split(".")[-1]
            self.maximum = _maximum(call.batch, call.vocab, dtype, width)
            self.mask = _mask(call.batch, call.vocab, dtype, width)

    def forward(self, logits, min_p):
        if not self.call.batch:
            return torch.empty_like(logits)
        return self.mask(logits, min_p, self.maximum(logits))


register_kernel_type("MinPMaskFwdOp", "ascend", AscendMinPMaskKernel)
