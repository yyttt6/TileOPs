"""Ascend bitwise NOT with exact integer words, bool semantics and arbitrary tails."""

import functools
from typing import ClassVar

import torch

from tileops.kernels.elementwise.call_spec import UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Kernel

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None


if T is not None:

    @T.macro
    def _body(a, c, width: int, boolean: bool):
        if width % 64 == 0:
            with T.SimdVF():
                mask = T.simd.pset(32)
                if boolean:
                    one = T.simd.vdup(T.int32(0x01010101), "int32", mask)
                for i in range(width // 64):
                    av = T.simd.vld(a[i * 64])
                    if boolean:
                        low = T.simd.vdup(T.int32(0x7F7F7F7F), "int32", mask)
                        aa = T.simd.vor(
                            T.simd.vor(
                                T.simd.vadd(T.simd.vand(av, low, mask), low, mask), av, mask
                            ),
                            low,
                            mask,
                        )
                        av = T.simd.vand(T.simd.vshrs(aa, T.int32(7), mask), one, mask)
                        cv = T.simd.vxor(av, one, mask)
                    else:
                        cv = T.simd.vnot(av, mask)
                    T.simd.vsts(c[i * 64], cv, mask)
        else:
            with T.SimtVF(threads=1024):
                for i in T.Parallel(width):
                    if boolean:
                        au = T.reinterpret(a[i], "uint32")
                        aa = T.bitwise_or(
                            T.bitwise_or(
                                T.bitwise_and(au, T.uint32(0x7F7F7F7F)) + T.uint32(0x7F7F7F7F), au
                            ),
                            T.uint32(0x7F7F7F7F),
                        )
                        av = T.reinterpret(
                            T.bitwise_and(T.shift_right(aa, T.uint32(7)), T.uint32(0x01010101)),
                            "int32",
                        )
                        c[i] = T.bitwise_xor(av, T.int32(0x01010101))
                    else:
                        c[i] = T.bitwise_xor(a[i], T.int32(-1))

    @functools.lru_cache(maxsize=128)
    def build(words, boolean, tile=16384, blocks=64):
        tile = min(tile, words)
        full, tail = divmod(words, tile)
        grid = min(blocks, full)

        @T.prim_func
        def main(A: T.Tensor((words,), "int32"), C: T.Tensor((words,), "int32")):
            with T.Kernel(grid) as bx:
                a = T.alloc_shared((tile,), "int32")
                c = a
                T.annotate_buffer_versions({a: 2})
                for step in T.Pipelined((full + grid - 1) // grid, num_stages=2):
                    block = step * grid + bx
                    if block < full:
                        begin = block * tile
                        T.copy(A[begin : begin + tile], a)
                        _body(a, c, tile, boolean)
                        T.copy(c, C[begin : begin + tile], l2_cache_ctrl=0)
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        at = T.alloc_shared((tail,), "int32")
                        ct = at
                        T.copy(A[full * tile : words], at)
                        _body(at, ct, tail, boolean)
                        T.copy(ct, C[full * tile : words], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)


class AscendBitwiseNotKernel(Kernel, UnaryElementwiseFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total, dtype, config=None):
        super().__init__()
        self.n_total, self.dtype = n_total, dtype
        self.bytes = n_total * dtype.itemsize
        self.words = (self.bytes + 3) // 4
        self.compiled = build(self.words, dtype == torch.bool) if self.words else None

    def forward(self, input):
        if input.dtype != self.dtype or input.numel() != self.n_total or input.device.type != "npu":
            raise ValueError("Input must match the NPU kernel dtype and element count")
        if not self.n_total:
            return input.clone()
        raw = input.contiguous().reshape(-1).view(torch.uint8)
        if self.bytes % 4:
            raw = torch.nn.functional.pad(raw, (0, self.words * 4 - self.bytes))
        result = self.compiled(raw.view(torch.int32))
        return result.view(torch.uint8)[: self.bytes].view(self.dtype).reshape(input.shape)


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("BitwiseNotFwdOp", "ascend", AscendBitwiseNotKernel)
