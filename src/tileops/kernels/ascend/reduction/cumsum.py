"""Bounded SIMD prefix-scan candidate; FP32 prefixes, narrowing only at output."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.call_spec import CumsumFwdInterface


@T.macro
def _scan(x, y, state, rows, count, dtype: str):
    with T.SimdVF():
        mask = T.simd.pset(32)
        one = T.simd.pset(32, "PAT_VL1")
        lanes = T.simd.vci(T.int32(0), "int32")
        last = T.simd.vdup(T.uint32(63), "uint32")
        zero = T.simd.vdup(T.float32(0), "float32")
        indices = T.simd.alloc_local(6, "int32")
        for k in T.unroll(6, explicit=True):
            indices[k] = T.simd.vmaxs(
                T.simd.vadds(lanes, T.int32(-(1 << k)), mask), T.int32(0), mask
            )
        for r in T.serial(rows):
            carry = T.simd.alloc_local(1, "float32")
            carry[0] = T.simd.vld(state[r, 0], dist="BRC_B32")
            for i in T.serial((count + 63) // 64):
                if dtype == "float32":
                    value = T.simd.vld(x[r, i * 64])
                else:
                    value = T.simd.vcvt(
                        T.simd.vld(x[r, i * 64], dist="UNPK_B16"), "float32", part=0
                    )
                valid = T.simd.vcmps(lanes, count - i * 64, mask, op="lt")
                value = T.simd.vsel(value, zero, valid)
                prefix = T.simd.alloc_local(1, "float32")
                prefix[0] = value
                for k in T.unroll(6, explicit=True):
                    shifted = T.simd.vselr(prefix[0], indices[k])
                    active = T.simd.vcmps(lanes, T.int32(1 << k), mask, op="ge")
                    prefix[0] = T.simd.vadd(prefix[0], T.simd.vsel(shifted, zero, active), mask)
                value = T.simd.vadd(prefix[0], carry[0], mask)
                carry[0] = T.simd.vselr(value, last)
                if dtype == "float32":
                    T.simd.vsts(y[r, i * 64], value, mask, extent=64)
                else:
                    T.simd.vsts(
                        y[r, i * 64],
                        T.simd.vcvt(value, dtype, sat=False),
                        mask,
                        dist="PK_B32",
                        extent=64,
                    )
            T.simd.vsts(state[r, 0], carry[0], one, extent=1)
        T.simd.mem_bar("VST_VLD")


@functools.lru_cache(maxsize=256)
def build(m, n, dtype):
    width = min(8192, max(64, tilelang.next_power_of_2(n)))
    rows = 1
    groups = (m + rows - 1) // rows
    blocks = min(64, groups)
    parts = (n + width - 1) // width

    @T.prim_func
    def main(X: T.Tensor((m, n), dtype), Y: T.Tensor((m, n), dtype)):
        with T.Kernel(blocks) as bx:
            x = T.alloc_shared((rows, width), dtype)
            state = T.alloc_shared((rows, 8), "float32")
            for group in T.serial((groups + blocks - 1) // blocks):
                row = (group * blocks + bx) * rows
                if row < m:
                    T.fill(state, 0)
                    for part in T.serial(parts):
                        begin = part * width
                        count = T.min(width, n - begin)
                        T.copy(X[row : row + 1, begin : begin + count], x[:, :count])
                        _scan(x, x, state, rows, count, dtype)
                        T.copy(x[:, :count], Y[row : row + 1, begin : begin + count])

    return tilelang.compile(main, target="ascend", out_idx=[1])


class AscendCumsumKernel(Kernel, CumsumFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        self.M = call.m
        self.N = call.n
        self.dtype = call.dtype
        self.scan_axis = call.axis
        if call.op_kind != "sum":
            raise ValueError("Expected sum scan")
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("Unsupported cumsum dtype")
        self.empty = self.M == 0 or self.N == 0
        if not self.empty:
            self.kernel = build(self.M, self.N, str(call.dtype).split(".")[-1])

    def forward(self, x):
        if tuple(x.shape) != self.call.shape or x.dtype != self.dtype:
            raise ValueError("Input signature changed")
        if self.empty:
            return torch.empty_like(x)
        if x.is_neg():
            x = x.resolve_neg()
        arg = x.movedim(self.scan_axis, -1).contiguous().reshape(self.M, self.N)
        y = self.kernel(arg)
        kept = self.call.shape[: self.scan_axis] + self.call.shape[self.scan_axis + 1 :]
        return y.reshape(*kept, self.N).movedim(-1, self.scan_axis)


register_kernel_type("CumsumFwdOp", "ascend", AscendCumsumKernel)
