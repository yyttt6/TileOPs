"""Streaming radix selection and logit masks with bounded unified-buffer storage."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend._registry import register_kernel_type
from tileops._kernels.ascend.sampling.radix_reduce import (
    _mass_tile,
    _probabilities_tile,
    _rank_tile,
)
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.sampling.call_spec import (
    TopKMaskFwdInterface,
    TopKTopPMaskFwdInterface,
    TopPMaskFwdInterface,
)

__all__ = [
    "AscendTopKMaskKernel",
    "AscendTopKTopPMaskKernel",
    "AscendTopPMaskKernel",
]


def _key(value, shift):
    value = T.cast(value, "float32")
    bits = T.reinterpret(T.if_then_else(value == 0, T.float32(0), value), "uint32")
    return (
        T.if_then_else(
            T.isnan(value),
            T.uint32(0xFFFFFFFF),
            T.if_then_else(bits >> 31 != 0, ~bits, bits ^ T.uint32(0x80000000)),
        )
        >> shift
    )


def _value_from_key(key, shift):
    full = key << shift
    full = T.if_then_else(full >> 31 == 0, full | T.uint32((1 << shift) - 1), full)
    raw = T.if_then_else(full >> 31 == 0, ~full, full ^ T.uint32(0x80000000))
    return T.reinterpret(raw, "float32")


@functools.lru_cache(maxsize=256)
def _build(batch, vocab, dtype, mode):
    # Two row tiles plus reduction scratch fit in UB for every supported dtype.
    width = min(8192, max(64, tilelang.next_power_of_2(vocab)))
    chunks = (vocab + width - 1) // width
    blocks = min(batch, 64)
    shift = 16 if dtype == "bfloat16" else 13 if dtype == "float16" else 0
    bits = 32 - shift
    has_k = mode in ("top_k", "top_k_top_p")
    has_p = mode in ("top_p", "top_k_top_p")

    @T.prim_func
    def main(
        X: T.Tensor((batch, vocab), dtype),
        K: T.Tensor((batch,), "int32"),
        P: T.Tensor((batch,), "float32"),
        Y: T.Tensor((batch, vocab), dtype),
    ):
        with T.Kernel(blocks) as bx:
            x = T.alloc_shared((width,), dtype)
            y = T.alloc_shared((width,), dtype)
            probabilities = T.alloc_shared((width,), "float32")
            boundary_probability = T.alloc_shared((8,), "float32")
            stats = T.alloc_shared((8,), "float32")
            param = T.alloc_shared((8,), "float32")
            kval = T.alloc_shared((8,), "int32")
            count_out = T.alloc_shared((8,), "int32")
            for r in T.serial(T.ceildiv(batch, blocks)):
                row = r * blocks + bx
                if row < batch:
                    if mode != "top_k":
                        T.copy(P[row : row + 1], param[:1])
                    if has_k:
                        T.copy(K[row : row + 1], kval[:1])
                    maximum = T.alloc_var("float32", init=-T.infinity("float32"))
                    nan_seen = T.alloc_var("float32", init=0)
                    if has_p:
                        for chunk in T.serial(chunks):
                            start = chunk * width
                            count = T.min(width, vocab - start)
                            T.copy(X[row, start : start + count], x[:count])
                            with T.SimtVF(threads=256):
                                mx = T.alloc_reducer((1,), "float32", op="max")
                                ns = T.alloc_reducer((1,), "float32", op="sum")
                                T.reducer_init(mx)
                                T.reducer_init(ns)
                                for i in T.Parallel(width):
                                    if i < count:
                                        T.reducer_update(
                                            mx[0],
                                            T.if_then_else(
                                                T.isnan(T.cast(x[i], "float32")),
                                                -T.infinity("float32"),
                                                T.cast(x[i], "float32"),
                                            ),
                                        )
                                        T.reducer_update(
                                            ns[0],
                                            T.if_then_else(
                                                T.isnan(T.cast(x[i], "float32")),
                                                T.float32(1),
                                                T.float32(0),
                                            ),
                                        )
                                a = T.alloc_fragment((1,), "float32")
                                b = T.alloc_fragment((1,), "float32")
                                T.finalize_reducer(mx, a)
                                T.finalize_reducer(ns, b)
                                stats[0] = a[0]
                                stats[1] = b[0]
                            maximum = T.max(maximum, stats[0])
                            nan_seen = nan_seen + stats[1]
                    cutoff = T.alloc_var("uint32", init=0)
                    if has_k:  # noqa: SIM102 -- compile-time mode removes unused parameter reads
                        if kval[0] < vocab:
                            for step in T.serial(bits):
                                pivot = cutoff | (T.uint32(1) << (bits - 1 - step))
                                rank = T.alloc_var("int32", init=0)
                                for chunk in T.serial(chunks):
                                    start = chunk * width
                                    count = T.min(width, vocab - start)
                                    T.copy(X[row, start : start + count], x[:count])
                                    _rank_tile(x, count_out, count, pivot, shift, dtype)
                                    rank = rank + count_out[0]
                                if rank >= kval[0]:
                                    cutoff = pivot
                    if has_k:  # noqa: SIM102 -- compile-time mode removes unused parameter reads
                        if kval[0] < vocab:
                            nan_seen = T.float32(0)
                    nucleus = T.alloc_var("uint32", init=0)
                    mass = T.alloc_var("float32", init=0)
                    if has_p:  # noqa: SIM102 -- compile-time mode removes unused parameter reads
                        if (
                            param[0] > 0
                            and param[0] <= 1
                            and nan_seen == 0
                            and (
                                maximum != T.infinity("float32")
                                and maximum != -T.infinity("float32")
                            )
                        ):
                            for chunk in T.serial(chunks):
                                start = chunk * width
                                count = T.min(width, vocab - start)
                                T.copy(X[row, start : start + count], x[:count])
                                _mass_tile(x, stats, count, cutoff, maximum, shift, dtype)
                                mass = mass + stats[0]
                            target = T.max(mass * param[0], T.float32(2.0**-126))
                            for step in T.serial(T.if_then_else(param[0] < 1, bits, 0)):
                                pivot = nucleus | (T.uint32(1) << (bits - 1 - step))
                                above = T.alloc_var("float32", init=0)
                                for chunk in T.serial(chunks):
                                    start = chunk * width
                                    count = T.min(width, vocab - start)
                                    T.copy(X[row, start : start + count], x[:count])
                                    _mass_tile(
                                        x, stats, count, T.max(pivot, cutoff), maximum, shift, dtype
                                    )
                                    above = above + stats[0]
                                if above >= target:
                                    nucleus = pivot
                    for chunk in T.serial(chunks):
                        start = chunk * width
                        count = T.min(width, vocab - start)
                        T.copy(X[row, start : start + count], x[:count])
                        if has_p:
                            _probabilities_tile(
                                x,
                                probabilities,
                                boundary_probability,
                                count,
                                maximum,
                                mass,
                                _value_from_key(nucleus, shift),
                                dtype,
                            )
                        with T.SimtVF(threads=256):
                            for i in T.Parallel(width):
                                if i < count:
                                    v = T.cast(x[i], "float32")
                                    keep = _key(v, shift) >= cutoff
                                    if has_k:
                                        keep = keep and (
                                            kval[0] >= vocab
                                            or (
                                                not T.isnan(v)
                                                and cutoff != (T.uint32(0xFFFFFFFF) >> shift)
                                            )
                                        )
                                    if has_p:
                                        # Different logits may round to the same probability.
                                        # Expand the selected boundary to the entire probability tie.
                                        probability = probabilities[i]
                                        boundary = boundary_probability[0]
                                        keep = keep and (
                                            param[0] <= 0
                                            or param[0] >= 1
                                            or nan_seen > 0
                                            or maximum == T.infinity("float32")
                                            or maximum == -T.infinity("float32")
                                            or probability >= boundary
                                        )
                                        keep = keep and (
                                            param[0] != 1
                                            or nan_seen > 0
                                            or maximum == T.infinity("float32")
                                            or maximum == -T.infinity("float32")
                                            or probability > 0
                                        )
                                        keep = keep and (
                                            param[0] > 0
                                            or nan_seen > 0
                                            or (
                                                maximum == T.infinity("float32")
                                                or maximum == -T.infinity("float32")
                                            )
                                        )
                                    y[i] = T.if_then_else(
                                        keep, x[i], T.cast(-T.infinity("float32"), dtype)
                                    )
                        T.copy(y[:count], Y[row, start : start + count])

    return tilelang.compile(main, target="ascend", out_idx=[3])


class _MaskKernel(Kernel):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    mode: ClassVar[str]

    @classmethod
    def refusal(cls, call):
        if call.batch * call.vocab > 2**31 - 1:
            return "logit offsets require B * V <= 2**31 - 1"
        return None

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call
        self.kernel = (
            _build(call.batch, call.vocab, str(call.dtype).split(".")[-1], self.mode)
            if call.batch and call.vocab
            else None
        )

    def _run(self, logits, k=None, p=None):
        if self.kernel is None:
            return torch.empty_like(logits)
        # Unused arguments remain device buffers and never influence the selected mode.
        if k is None:
            k = torch.empty((self.call.batch,), dtype=torch.int32, device=logits.device)
        if p is None:
            p = torch.empty((self.call.batch,), dtype=torch.float32, device=logits.device)
        return self.kernel(logits, k, p)


class AscendTopKMaskKernel(_MaskKernel, TopKMaskFwdInterface):
    mode = "top_k"

    def forward(self, logits, k):
        return self._run(logits, k=k)


class AscendTopPMaskKernel(_MaskKernel, TopPMaskFwdInterface):
    mode = "top_p"

    def forward(self, logits, p):
        return self._run(logits, p=p)


class AscendTopKTopPMaskKernel(_MaskKernel, TopKTopPMaskFwdInterface):
    mode = "top_k_top_p"

    def forward(self, logits, k, p):
        return self._run(logits, k=k, p=p)


register_kernel_type("TopKMaskFwdOp", "ascend", AscendTopKMaskKernel)
register_kernel_type("TopPMaskFwdOp", "ascend", AscendTopPMaskKernel)
register_kernel_type("TopKTopPMaskFwdOp", "ascend", AscendTopKTopPMaskKernel)
