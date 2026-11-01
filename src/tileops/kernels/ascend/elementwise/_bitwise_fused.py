"""Ascend integer bitwise kernels, using int32 words without integer narrowing."""

import functools
import math
from typing import ClassVar

import torch
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.elementwise.call_spec import BinaryElementwiseFwdInterface

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None


if T is not None:

    @T.macro
    def _body(a, b, c, width: int, kind: str, boolean: bool):
        if width % 64 == 0:
            with T.SimdVF():
                mask = T.simd.pset(32)
                for i in range(width // 64):
                    av = T.simd.vld(a[i * 64])
                    bv = T.simd.vld(b[i * 64])
                    if boolean:
                        low = T.simd.vdup(T.int32(0x7F7F7F7F), "int32", mask)
                        one = T.simd.vdup(T.int32(0x01010101), "int32", mask)
                        aa = T.simd.vor(
                            T.simd.vor(
                                T.simd.vadd(T.simd.vand(av, low, mask), low, mask),
                                av,
                                mask,
                            ),
                            low,
                            mask,
                        )
                        bb = T.simd.vor(
                            T.simd.vor(
                                T.simd.vadd(T.simd.vand(bv, low, mask), low, mask),
                                bv,
                                mask,
                            ),
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
                    bv = b[i]
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
    def build(words, kind, tile=4096, blocks=64, boolean=False):
        tile = min(tile, words)
        full, tail = divmod(words, tile)
        grid = min(blocks, full)

        @T.prim_func
        def main(
            A: T.Tensor((words,), "int32"),
            B: T.Tensor((words,), "int32"),
            C: T.Tensor((words,), "int32"),
        ):
            with T.Kernel(grid) as bx:
                a = T.alloc_shared((tile,), "int32")
                b = T.alloc_shared((tile,), "int32")
                c = T.alloc_shared((tile,), "int32")
                T.annotate_buffer_versions({a: 2, b: 2, c: 2})
                for step in T.Pipelined((full + grid - 1) // grid, num_stages=2):
                    block = step * grid + bx
                    if block < full:
                        begin = block * tile
                        T.copy(A[begin : begin + tile], a)
                        T.copy(B[begin : begin + tile], b)
                        _body(a, b, c, tile, kind, boolean)
                        T.copy(c, C[begin : begin + tile], l2_cache_ctrl=0)
                if tail:
                    if bx == 0:
                        at = T.alloc_shared((tail,), "int32")
                        bt = T.alloc_shared((tail,), "int32")
                        ct = T.alloc_shared((tail,), "int32")
                        T.copy(A[full * tile : words], at)
                        T.copy(B[full * tile : words], bt)
                        _body(at, bt, ct, tail, kind, boolean)
                        T.copy(ct, C[full * tile : words], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)


class BitwiseBinaryBase(Kernel, BinaryElementwiseFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    kind = "and"

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.a_shape, call.b_shape, call.dtype)

    def __init__(self, a_shape, b_shape, dtype):
        super().__init__()
        self.shape = tuple(torch.broadcast_shapes(a_shape, b_shape))
        self.dtype = dtype
        self.n = math.prod(self.shape)
        self.bytes = self.n * dtype.itemsize
        self.words = (self.bytes + 3) // 4
        self.compiled = (
            build(self.words, self.kind, boolean=dtype == torch.bool) if self.words else None
        )

    def _pack(self, x):
        x = x.expand(self.shape).contiguous().reshape(-1)
        raw = x.view(torch.uint8)
        if self.bytes % 4:
            raw = torch.nn.functional.pad(raw, (0, self.words * 4 - self.bytes))
        return raw.view(torch.int32)

    def forward(self, input, other):
        if not self.n:
            return torch.empty(self.shape, dtype=self.dtype, device=input.device)
        result = self.compiled(self._pack(input), self._pack(other))
        return result.view(torch.uint8)[: self.bytes].view(self.dtype).reshape(self.shape)
