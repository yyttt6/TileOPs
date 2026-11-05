"""Ascend cumprod with a bounded multiply path and range-recovering log/sign scan."""

import functools
import math
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.call_spec import CumprodFwdInterface


@T.macro
def _scan(x, y, state, rows, count, dtype: str):
    with T.SimdVF():
        mask = T.simd.pset(32)
        one = T.simd.pset(32, "PAT_VL1")
        lanes = T.simd.vci(T.int32(0), "int32")
        last = T.simd.vdup(T.uint32(63), "uint32")
        zero = T.simd.vdup(T.float32(1), "float32")
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
                    prefix[0] = T.simd.vmul(prefix[0], T.simd.vsel(shifted, zero, active), mask)
                value = T.simd.vmul(prefix[0], carry[0], mask)
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


@T.macro
def _check(x, flag, count, dtype: str):
    with T.SimdVF():
        mask = T.simd.pset(32)
        one = T.simd.pset(32, "PAT_VL1")
        lanes = T.simd.vci(T.int32(0), "int32")
        zero = T.simd.vdup(T.int32(0), "int32")
        ones = T.simd.vdup(T.int32(1), "int32")
        bad = T.simd.alloc_local(1, "int32")
        bad[0] = T.simd.vld(flag[0], dist="BRC_B32")
        for i in T.serial((count + 63) // 64):
            if dtype == "float32":
                value = T.simd.vld(x[0, i * 64])
            else:
                value = T.simd.vcvt(T.simd.vld(x[0, i * 64], dist="UNPK_B16"), "float32", part=0)
            invalid = T.simd.vcmps(T.simd.vabs(value, mask), T.float32(1), mask, op="gt")
            lane_bad = T.simd.vsel(ones, zero, invalid)
            lane_bad = T.simd.vsel(ones, lane_bad, T.simd.vcmp(value, value, mask, op="ne"))
            lane_bad = T.simd.vsel(
                lane_bad, zero, T.simd.vcmps(lanes, count - i * 64, mask, op="lt")
            )
            bad[0] = T.simd.vmax(bad[0], lane_bad, mask)
        total = T.simd.vcmax(bad[0], mask)
        T.simd.vsts(flag[0], total, one, extent=1)
        T.simd.mem_bar("VST_LD")


@T.macro
def _logscan(x, y, state, count, dtype: str):
    with T.SimdVF():
        mask = T.simd.pset(32)
        one = T.simd.pset(32, "PAT_VL1")
        lanes = T.simd.vci(T.int32(0), "int32")
        last = T.simd.vdup(T.uint32(63), "uint32")
        zero = T.simd.vdup(T.float32(0), "float32")
        izero = T.simd.vdup(T.int32(0), "int32")
        indices = T.simd.alloc_local(6, "int32")
        for k in T.unroll(6, explicit=True):
            indices[k] = T.simd.vmaxs(
                T.simd.vadds(lanes, T.int32(-(1 << k)), mask), T.int32(0), mask
            )
        carry = T.simd.alloc_local(1, "float32")
        carry[0] = T.simd.vld(state[0, 0], dist="BRC_B32")
        carrysign = T.simd.alloc_local(1, "int32")
        carrysign[0] = T.reinterpret(T.simd.vld(state[0, 8], dist="BRC_B32"), "int32x64")
        for i in T.serial((count + 63) // 64):
            if dtype == "float32":
                value = T.simd.vld(x[0, i * 64])
            else:
                value = T.simd.vcvt(T.simd.vld(x[0, i * 64], dist="UNPK_B16"), "float32", part=0)
            raw = T.reinterpret(value, "int32x64")
            abits = T.simd.vand(raw, T.simd.vdup(T.int32(2147483647), "int32"), mask)
            sign = T.simd.alloc_local(1, "int32")
            sign[0] = T.simd.vand(raw, T.simd.vdup(T.int32(-2147483648), "int32"), mask)
            normal = T.simd.vln(T.reinterpret(abits, "float32x64"), mask, precision="ftz_false")
            # Decode subnormal input magnitude from its integer significand; arithmetic
            # on the original subnormal may flush before logarithm evaluation.
            fraction = T.simd.vcvt(abits, "float32", mask=mask)
            tiny = T.simd.vadds(
                T.simd.vln(fraction, mask, precision="ftz_false"),
                T.float32(-149 * math.log(2)),
                mask,
            )
            prefix = T.simd.alloc_local(1, "float32")
            prefix[0] = T.simd.vsel(
                tiny, normal, T.simd.vcmps(abits, T.int32(8388608), mask, op="lt")
            )
            valid = T.simd.vcmps(lanes, count - i * 64, mask, op="lt")
            prefix[0] = T.simd.vsel(prefix[0], zero, valid)
            sign[0] = T.simd.vsel(sign[0], izero, valid)
            for k in T.unroll(6, explicit=True):
                active = T.simd.vcmps(lanes, T.int32(1 << k), mask, op="ge")
                shifted = T.simd.vselr(prefix[0], indices[k])
                prefix[0] = T.simd.vadd(prefix[0], T.simd.vsel(shifted, zero, active), mask)
                shifted_sign = T.simd.vselr(sign[0], indices[k])
                sign[0] = T.simd.vxor(sign[0], T.simd.vsel(shifted_sign, izero, active), mask)
            total = T.simd.vadd(prefix[0], carry[0], mask)
            sign[0] = T.simd.vxor(sign[0], carrysign[0], mask)
            carry[0] = T.simd.vselr(total, last)
            carrysign[0] = T.simd.vselr(sign[0], last)
            magnitude = T.simd.vexp(total, mask, precision="ftz_false")
            if dtype == "float32":
                out = T.reinterpret(magnitude, "int32x64")
                outsign = sign[0]
            else:
                out = T.reinterpret(T.simd.vcvt(magnitude, dtype, sat=False), "int32x64")
                outsign = T.simd.vand(
                    T.simd.vshrs(sign[0], 16, mask), T.simd.vdup(T.int32(32768), "int32"), mask
                )
            if dtype != "float16":
                small = T.simd.vcmps(total, T.float32(-126 * math.log(2)), mask, op="lt")
                scale = (149 if dtype == "float32" else 133) * math.log(2)
                scaled = T.simd.vexp(
                    T.simd.vadds(total, T.float32(scale), mask), small, precision="ftz_false"
                )
                bits = T.simd.vcvt(scaled, "int32", mask=small, round="ROUND_R")
                out = T.simd.vsel(bits, out, small)
            out = T.simd.vor(out, outsign, mask)
            if dtype == "float32":
                T.simd.vsts(y[0, i * 64], T.reinterpret(out, "float32x64"), mask, extent=64)
            elif dtype == "float16":
                T.simd.vsts(
                    y[0, i * 64], T.reinterpret(out, "float16x128"), mask, dist="PK_B32", extent=64
                )
            else:
                T.simd.vsts(
                    y[0, i * 64], T.reinterpret(out, "bfloat16x128"), mask, dist="PK_B32", extent=64
                )
        T.simd.vsts(state[0, 0], carry[0], one, extent=1)
        T.simd.vsts(state[0, 8], T.reinterpret(carrysign[0], "float32x64"), one, extent=1)
        T.simd.mem_bar("VST_VLD")


@functools.lru_cache(maxsize=256)
def build(m, n, dtype):
    width = min(8192, max(64, tilelang.next_power_of_2(n)))
    blocks = min(64, m)
    parts = (n + width - 1) // width

    @T.prim_func
    def main(X: T.Tensor((m, n), dtype), Y: T.Tensor((m, n), dtype)):
        with T.Kernel(blocks) as bx:
            x = T.alloc_shared((1, width), dtype)
            state = T.alloc_shared((1, 16), "float32")
            flag = T.alloc_shared((8,), "int32")
            for group in T.serial((m + blocks - 1) // blocks):
                row = group * blocks + bx
                if row < m:
                    T.fill(flag, 0)
                    for part in T.serial(parts):
                        begin = part * width
                        count = T.min(width, n - begin)
                        T.copy(X[row : row + 1, begin : begin + count], x[:, :count])
                        _check(x, flag, count, dtype)
                    if flag[0] == 0:
                        T.fill(state, 1)
                        for part in T.serial(parts):
                            begin = part * width
                            count = T.min(width, n - begin)
                            T.copy(X[row : row + 1, begin : begin + count], x[:, :count])
                            _scan(x, x, state, 1, count, dtype)
                            T.copy(x[:, :count], Y[row : row + 1, begin : begin + count])
                    else:
                        T.fill(state, 0)
                        for part in T.serial(parts):
                            begin = part * width
                            count = T.min(width, n - begin)
                            T.copy(X[row : row + 1, begin : begin + count], x[:, :count])
                            _logscan(x, x, state, count, dtype)
                            T.copy(x[:, :count], Y[row : row + 1, begin : begin + count])

    return tilelang.compile(main, target="ascend", out_idx=[1])


class AscendCumprodKernel(Kernel, CumprodFwdInterface):
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
        if call.op_kind != "prod":
            raise ValueError("Expected product scan")
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("Unsupported cumprod dtype")
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


register_kernel_type("CumprodFwdOp", "ascend", AscendCumprodKernel)
