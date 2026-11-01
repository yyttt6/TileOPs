"""Ascend logical NOT: SIMD zero predicates, packed output, exact integer tails."""

import functools
from typing import ClassVar
import torch
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.elementwise.call_spec import UnaryPredicateFwdInterface

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None

if T is not None:

    @T.macro
    def _body(a, c, width: int, dtype: str, native: bool):
        if native:
            with T.SimdVF():
                if dtype != "float32":
                    mask = T.simd.pset(16)
                    one = T.simd.vdup(T.int16(1), "int16", mask)
                    zero = T.simd.vdup(T.int16(0), "int16", mask)
                    for i in range(width // 128):
                        av = T.simd.vld(a[i * 128])
                        pred = T.simd.vcmps(av, 0, mask, "eq")
                        val = T.simd.vsel(one, zero, pred)
                        T.simd.vsts(c[i * 128], val, mask, dist="PK_B16")
                    if width % 128:
                        last = width // 128 * 128
                        mask32 = T.simd.pset(32)
                        av32 = T.simd.vcvt(T.simd.vld(a[last], dist="UNPK_B16"), "float32", part=0)
                        pred32 = T.simd.vcmps(av32, 0, mask32, "eq")
                        one32 = T.simd.vdup(T.int32(1), "int32", mask32)
                        zero32 = T.simd.vdup(T.int32(0), "int32", mask32)
                        val32 = T.simd.vsel(one32, zero32, pred32)
                        T.simd.vsts(c[last], val32, mask32, dist="PK4_B32")
                else:
                    mask = T.simd.pset(32)
                    one = T.simd.vdup(T.int32(1), "int32", mask)
                    zero = T.simd.vdup(T.int32(0), "int32", mask)
                    for i in range(width // 64):
                        av = T.simd.vld(a[i * 64])
                        pred = T.simd.vcmps(av, 0, mask, "eq")
                        val = T.simd.vsel(one, zero, pred)
                        T.simd.vsts(c[i * 64], val, mask, dist="PK4_B32")
        else:
            with T.SimtVF(threads=1024):
                for i in T.Parallel(width // 4):
                    p0 = T.cast(a[i * 4] == T.cast(0, dtype), "int32")
                    p1 = T.cast(a[i * 4 + 1] == T.cast(0, dtype), "int32")
                    p2 = T.cast(a[i * 4 + 2] == T.cast(0, dtype), "int32")
                    p3 = T.cast(a[i * 4 + 3] == T.cast(0, dtype), "int32")
                    c[i] = T.bitwise_or(
                        T.bitwise_or(p0, T.shift_left(p1, 8)),
                        T.bitwise_or(T.shift_left(p2, 16), T.shift_left(p3, 24)),
                    )

    @functools.lru_cache(maxsize=256)
    def build(n, dtype, native, tile, blocks=64, stages=2):
        tile = min(n, tile)
        full, tail = divmod(n, tile)
        grid = min(blocks, full)
        pack = 1 if native else 4
        out_dtype = "uint8" if native else "int32"

        @T.prim_func
        def main(A: T.Tensor((n,), dtype), C: T.Tensor((n // pack,), out_dtype)):
            with T.Kernel(grid) as bx:
                a = T.alloc_shared((tile,), dtype)
                c = T.alloc_shared((tile // pack,), out_dtype)
                T.annotate_buffer_versions({a: stages, c: stages})
                for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                    block = step * grid + bx
                    if block < full:
                        start = block * tile
                        T.copy(A[start : start + tile], a)
                        _body(a, c, tile, dtype, native)
                        T.copy(c, C[start // pack : (start + tile) // pack], l2_cache_ctrl=0)
                if tail:
                    if bx == 0:
                        at = T.alloc_shared((tail,), dtype)
                        ct = T.alloc_shared((tail // pack,), out_dtype)
                        T.copy(A[full * tile : n], at)
                        _body(at, ct, tail, dtype, native)
                        T.copy(ct, C[full * tile // pack : n // pack], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)


class AscendLogicalNotKernel(Kernel, UnaryPredicateFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total, dtype):
        super().__init__()
        self.n, self.dtype = n_total, dtype
        self.bool_impl = None
        self.padded = (n_total + 3) // 4 * 4
        if dtype == torch.bool:
            from tileops.kernels.ascend.elementwise.bitwise_not import AscendBitwiseNotKernel

            self.bool_impl = AscendBitwiseNotKernel(n_total, dtype)
            self.compiled = self.bool_impl.compiled
        elif n_total:
            native = dtype in (torch.float16, torch.bfloat16, torch.float32) and n_total % 64 == 0
            tile = 16384 if native and dtype != torch.float32 else 8192
            if dtype == torch.int64:
                tile = 4096
            self.compiled = build(self.padded, str(dtype).removeprefix("torch."), native, tile)
        else:
            self.compiled = None

    def forward(self, input):
        if input.device.type != "npu" or input.dtype != self.dtype or input.numel() != self.n:
            raise ValueError("Input must match the NPU kernel dtype and element count")
        if self.bool_impl is not None:
            return self.bool_impl.forward(input)
        if not self.n:
            return torch.empty(input.shape, dtype=torch.bool, device=input.device)
        a = input.contiguous().reshape(-1)
        if self.padded != self.n:
            a = torch.nn.functional.pad(a, (0, self.padded - self.n))
        out = self.compiled(a)
        return out.view(torch.uint8).view(torch.bool)[: self.n].reshape(input.shape)


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("LogicalNotFwdOp", "ascend", AscendLogicalNotKernel)
