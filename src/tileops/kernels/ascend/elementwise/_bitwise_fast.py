"""Candidate channel broadcast directly from compact operand storage."""

import functools
import math
import torch
import tilelang
import tilelang.ascend.language as T
from tileops.kernels.ascend.elementwise import _bitwise_fused as base


@T.macro
def body(a, b, c, width: int, channel, period: int, itemsize: int, boolean: bool, kind: str):
    with T.SimdVF():
        mask = T.simd.pset(32)
        if itemsize == 8:
            lo = T.simd.vld(b[(channel % period) * 2], dist="BRC_B32")
            hi = T.simd.vld(b[(channel % period) * 2 + 1], dist="BRC_B32")
            bv, unused = T.simd.vintlv(lo, hi)
        elif itemsize == 4:
            bv = T.simd.vld(b[channel % period], dist="BRC_B32")
        else:
            packed = T.simd.vld(b[(channel % period) // 4], dist="BRC_B32")
            shifted = T.simd.vshrs(packed, T.cast(((channel % period) % 4) * 8, "int32"), mask)
            byte = T.simd.vand(shifted, T.simd.vdup(T.int32(255), "int32", mask), mask)
            pred = T.simd.vcmps(byte, T.int32(0), mask, "ne")
            bv = T.simd.vsel(
                T.simd.vdup(T.int32(0x01010101), "int32", mask),
                T.simd.vdup(T.int32(0), "int32", mask),
                pred,
            )
        for i in range(width // 64):
            av = T.simd.vld(a[i * 64])
            if boolean:
                low = T.simd.vdup(T.int32(0x7F7F7F7F), "int32", mask)
                one = T.simd.vdup(T.int32(0x01010101), "int32", mask)
                aa = T.simd.vor(
                    T.simd.vor(T.simd.vadd(T.simd.vand(av, low, mask), low, mask), av, mask),
                    low,
                    mask,
                )
                av = T.simd.vand(T.simd.vshrs(aa, T.int32(7), mask), one, mask)
            if kind == "and":
                cv = T.simd.vand(av, bv, mask)
            elif kind == "or":
                cv = T.simd.vor(av, bv, mask)
            else:
                cv = T.simd.vxor(av, bv, mask)
            T.simd.vsts(c[i * 64], cv, mask)


@functools.lru_cache(maxsize=128)
def build_channel(n, period, repeat, itemsize, boolean, kind):
    words = n * itemsize // 4
    channel_words = repeat * itemsize // 4
    width = (channel_words + 63) // 64 * 64
    bwords = period * itemsize // 4
    channels = n // repeat
    grid = min(64, channels)

    @T.prim_func
    def main(
        A: T.Tensor((words,), "int32"),
        B: T.Tensor((bwords,), "int32"),
        C: T.Tensor((words,), "int32"),
    ):
        with T.Kernel(grid) as bx:
            a = T.alloc_shared((width,), "int32")
            b = T.alloc_shared((bwords,), "int32")
            c = T.alloc_shared((width,), "int32")
            T.copy(B, b)
            T.annotate_buffer_versions({a: 2, c: 2})
            for step in T.Pipelined((channels + grid - 1) // grid, num_stages=2):
                channel = step * grid + bx
                if channel < channels:
                    begin = channel * channel_words
                    T.copy(A[begin : begin + channel_words], a[:channel_words])
                    body(a, b, c, width, channel, period, itemsize, boolean, kind)
                    T.copy(
                        c[:channel_words],
                        C[begin : begin + channel_words],
                        l2_cache_ctrl=0,
                    )

    return tilelang.compile(main, target="ascend", out_idx=-1)


@T.macro
def period_body(a, b, c, width: int, kind: str, boolean: bool, begin, period: int):
    if width % 64 == 0:
        with T.SimdVF():
            mask = T.simd.pset(32)
            for i in range(width // 64):
                av = T.simd.vld(a[i * 64])
                bv = T.simd.vld(b[(begin + i * 64) % period])
                if boolean:
                    low = T.simd.vdup(T.int32(0x7F7F7F7F), "int32", mask)
                    one = T.simd.vdup(T.int32(0x01010101), "int32", mask)
                    aa = T.simd.vor(
                        T.simd.vor(T.simd.vadd(T.simd.vand(av, low, mask), low, mask), av, mask),
                        low,
                        mask,
                    )
                    bb = T.simd.vor(
                        T.simd.vor(T.simd.vadd(T.simd.vand(bv, low, mask), low, mask), bv, mask),
                        low,
                        mask,
                    )
                    av = T.simd.vand(T.simd.vshrs(aa, T.int32(7), mask), one, mask)
                    bv = T.simd.vand(T.simd.vshrs(bb, T.int32(7), mask), one, mask)
                if kind == "and":
                    cv = T.simd.vand(av, bv, mask)
                elif kind == "or":
                    cv = T.simd.vor(av, bv, mask)
                else:
                    cv = T.simd.vxor(av, bv, mask)
                T.simd.vsts(c[i * 64], cv, mask)
    else:
        with T.SimtVF(threads=1024):
            for i in T.Parallel(width):
                av = a[i]
                bv = b[(begin + i) % period]
                if boolean:
                    au = T.reinterpret(av, "uint32")
                    bu = T.reinterpret(bv, "uint32")
                    aa = T.bitwise_or(
                        T.bitwise_or(
                            T.bitwise_and(au, T.uint32(0x7F7F7F7F)) + T.uint32(0x7F7F7F7F),
                            au,
                        ),
                        T.uint32(0x7F7F7F7F),
                    )
                    bb = T.bitwise_or(
                        T.bitwise_or(
                            T.bitwise_and(bu, T.uint32(0x7F7F7F7F)) + T.uint32(0x7F7F7F7F),
                            bu,
                        ),
                        T.uint32(0x7F7F7F7F),
                    )
                    av = T.reinterpret(
                        T.bitwise_and(T.shift_right(aa, T.uint32(7)), T.uint32(0x01010101)),
                        "int32",
                    )
                    bv = T.reinterpret(
                        T.bitwise_and(T.shift_right(bb, T.uint32(7)), T.uint32(0x01010101)),
                        "int32",
                    )
                if kind == "and":
                    c[i] = T.bitwise_and(av, bv)
                elif kind == "or":
                    c[i] = T.bitwise_or(av, bv)
                else:
                    c[i] = T.bitwise_xor(av, bv)


@functools.lru_cache(maxsize=128)
def build_period(words, bwords, boolean, kind):
    tile = min(4096, words)
    assert words % tile == 0 and bwords % 64 == 0
    full = words // tile
    grid = min(64, full)

    @T.prim_func
    def main(
        A: T.Tensor((words,), "int32"),
        B: T.Tensor((bwords,), "int32"),
        C: T.Tensor((words,), "int32"),
    ):
        with T.Kernel(grid) as bx:
            a = T.alloc_shared((tile,), "int32")
            b = T.alloc_shared((bwords,), "int32")
            c = T.alloc_shared((tile,), "int32")
            T.copy(B, b)
            T.annotate_buffer_versions({a: 2, c: 2})
            for step in T.Pipelined((full + grid - 1) // grid, num_stages=2):
                block = step * grid + bx
                if block < full:
                    begin = block * tile
                    T.copy(A[begin : begin + tile], a)
                    period_body(a, b, c, tile, kind, boolean, begin, bwords)
                    T.copy(c, C[begin : begin + tile], l2_cache_ctrl=0)

    return tilelang.compile(main, target="ascend", out_idx=-1)


class FastBitwiseBinaryBase(base.BitwiseBinaryBase):
    kind = "and"

    def __init__(self, a_shape, b_shape, dtype):
        shape = tuple(torch.broadcast_shapes(a_shape, b_shape))
        padded = (1,) * (len(shape) - len(b_shape)) + tuple(b_shape)
        axes = [i for i, s in enumerate(padded) if s > 1]
        self.fast = False
        if (
            axes
            and math.prod(a_shape) == math.prod(shape)
            and dtype in (torch.bool, torch.int32, torch.int64)
        ):
            lo, hi = axes[0], axes[-1]
            period = math.prod(padded[lo : hi + 1])
            repeat = math.prod(shape[hi + 1 :])
            if (
                all(padded[i] == shape[i] for i in range(lo, hi + 1))
                and repeat > 1
                and repeat * dtype.itemsize % 4 == 0
                and period * dtype.itemsize % 4 == 0
                and ((repeat * dtype.itemsize + 255) // 256 * 256 * 4 + period * dtype.itemsize)
                < 248 * 1024
            ):
                super().__init__((0,), (0,), dtype)
                self.shape, self.n, self.dtype = shape, math.prod(shape), dtype
                self.compiled = build_channel(
                    self.n,
                    period,
                    repeat,
                    dtype.itemsize,
                    dtype == torch.bool,
                    self.kind,
                )
                self.fast = True
            if (
                not self.fast
                and all(padded[i] == shape[i] for i in range(lo, hi + 1))
                and repeat == 1
                and period * dtype.itemsize % 256 == 0
                and period * dtype.itemsize <= 32768
                and math.prod(shape) * dtype.itemsize // 4 % 4096 == 0
            ):
                super().__init__((0,), (0,), dtype)
                self.shape, self.n, self.dtype = shape, math.prod(shape), dtype
                self.compiled = build_period(
                    self.n * dtype.itemsize // 4,
                    period * dtype.itemsize // 4,
                    dtype == torch.bool,
                    self.kind,
                )
                self.fast = True
        if not self.fast:
            super().__init__(a_shape, b_shape, dtype)

    def forward(self, input, other):
        if not self.fast:
            return super().forward(input, other)
        a = input.contiguous().reshape(-1).view(torch.int32)
        b = other.contiguous().reshape(-1).view(torch.int32)
        return self.compiled(a, b).view(self.dtype).reshape(self.shape)
