"""Grouped RMSNorm with optional bias and a gate before or after normalization."""

from functools import lru_cache

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.kernel_base import Kernel
from tileops._kernels.norm.call_spec import RMSNormGatedFwdInterface
from tileops.backend._registry import register_kernel_type


@lru_cache(maxsize=64)
def _normalize(rows, n, group, dtype, weight_type, bias_type, gate_type, eps, before, silu):
    groups = n // group
    tasks = rows * groups
    blocks = min(tasks, 64)
    width = min(1024, tilelang.next_power_of_2(group))
    chunks = (group + width - 1) // width

    bias_storage = bias_type or weight_type
    gate_storage = gate_type or dtype
    has_bias = bias_type is not None
    has_gate = gate_type is not None

    @T.macro
    def gate(raw):
        # exp(-abs(z)) is finite on either tail. Preserve gates for large negative z.
        value = T.cast(raw, "float32")
        ex = T.exp(-T.abs(value))
        sigmoid = T.if_then_else(value < 0, ex / (1 + ex), 1 / (1 + ex))
        return T.if_then_else(silu, value * sigmoid, sigmoid)

    @T.prim_func
    def main(
        x: T.Tensor((rows, n), dtype),
        w: T.Tensor((n,), weight_type),
        bias: T.Tensor((n,), bias_storage),
        z: T.Tensor((rows, n), gate_storage),
        out: T.Tensor((rows, n), dtype),
    ):
        with T.Kernel(blocks) as bx:
            xv = T.alloc_shared((width,), dtype)
            zv = T.alloc_shared((width,), gate_storage)
            wv = T.alloc_shared((width,), weight_type)
            bv = T.alloc_shared((width,), bias_storage)
            y = T.alloc_shared((width,), dtype)
            square = T.alloc_shared((16,), "float32")
            for step in T.serial(T.ceildiv(tasks, blocks)):
                task = step * blocks + bx
                if task < tasks:
                    row = task // groups
                    first = task % groups * group
                    T.clear(square)
                    for part in T.serial(chunks):
                        start = first + part * width
                        count = T.min(width, group - part * width)
                        T.copy(x[row, start : start + count], xv[:count])
                        if has_gate and not before:
                            T.copy(z[row, start : start + count], zv[:count])
                        with T.SimtVF(threads=min(128, width)):
                            sums = T.alloc_reducer((1,), "float32", op="sum")
                            reduced = T.alloc_fragment((1,), "float32")
                            T.reducer_init(sums)
                            for i in T.Parallel(width):
                                if i < count:
                                    value = T.cast(xv[i], "float32")
                                    if has_gate and not before:
                                        value = value * gate(zv[i])
                                    T.reducer_update(sums[0], value * value)
                            T.finalize_reducer(sums, reduced)
                            square[0] = square[0] + reduced[0]
                    with T.SimtVF(threads=32):
                        square[0] = T.rsqrt(square[0] / group + eps)
                    for part in T.serial(chunks):
                        start = first + part * width
                        count = T.min(width, group - part * width)
                        T.copy(x[row, start : start + count], xv[:count])
                        T.copy(w[start : start + count], wv[:count])
                        if has_gate:
                            T.copy(z[row, start : start + count], zv[:count])
                        if has_bias:
                            T.copy(bias[start : start + count], bv[:count])
                        with T.SimtVF(threads=min(128, width)):
                            for i in T.Parallel(width):
                                if i < count:
                                    value = T.cast(xv[i], "float32")
                                    if has_gate and not before:
                                        value = value * gate(zv[i])
                                    value = value * square[0] * T.cast(wv[i], "float32")
                                    if has_bias:
                                        value = value + T.cast(bv[i], "float32")
                                    if has_gate and before:
                                        value = value * gate(zv[i])
                                    y[i] = T.cast(value, dtype)
                        T.copy(y[:count], out[row, start : start + count])

    return tilelang.compile(main, target="ascend", out_idx=[4])


@lru_cache(maxsize=32)
def _short_rows(rows, dtype, weight_dtype, eps, silu):
    width = 128
    batch = 32
    blocks = min(64, rows // batch)

    @T.prim_func
    def main(
        x: T.Tensor((rows, width), dtype),
        w: T.Tensor((width,), weight_dtype),
        z: T.Tensor((rows, width), dtype),
        out: T.Tensor((rows, width), dtype),
    ):
        with T.Kernel(blocks) as bx:
            xv = T.alloc_shared((batch, width), dtype)
            zv = T.alloc_shared((batch, width), dtype)
            y = T.alloc_shared((batch, width), dtype)
            wv = T.alloc_shared((width,), weight_dtype)
            T.annotate_buffer_versions({xv: 2, zv: 2, y: 2})
            T.copy(w, wv)
            for step in T.Pipelined(T.ceildiv(rows, blocks * batch), num_stages=2):
                start = (step * blocks + bx) * batch
                if start < rows:
                    T.copy(x[start : start + batch, :], xv)
                    T.copy(z[start : start + batch, :], zv)
                    with T.SimtVF(threads=128):
                        sums = T.alloc_reducer((batch,), "float32", op="sum")
                        reduced = T.alloc_fragment((batch,), "float32")
                        T.reducer_init(sums)
                        for r, c in T.Parallel(batch, width):
                            v = T.cast(xv[r, c], "float32")
                            T.reducer_update(sums[r], v * v)
                        T.finalize_reducer(sums, reduced)
                        for r, c in T.Parallel(batch, width):
                            gate = T.cast(zv[r, c], "float32")
                            ex = T.exp(-T.abs(gate))
                            sigmoid = T.if_then_else(gate < 0, ex / (1 + ex), 1 / (1 + ex))
                            act = T.if_then_else(silu, gate * sigmoid, sigmoid)
                            result = (
                                T.cast(xv[r, c], "float32")
                                * T.rsqrt(reduced[r] / width + eps)
                                * T.cast(wv[c], "float32")
                                * act
                            )
                            y[r, c] = T.cast(result, dtype)
                    T.copy(y, out[start : start + batch, :], l2_cache_ctrl=0)

    return tilelang.compile(main, target="ascend", out_idx=[3])


@lru_cache(maxsize=16)
def _mixed_weight_rows(rows, dtype, eps, silu):
    """Keep the 128 float32 weights resident and reduce each row in two vector loads."""
    batch = 32
    blocks = min(64, rows // batch)

    @T.prim_func
    def main(
        x: T.Tensor((rows, 128), dtype),
        w: T.Tensor((128,), "float32"),
        z: T.Tensor((rows, 128), dtype),
        out: T.Tensor((rows, 128), dtype),
    ):
        with T.Kernel(blocks) as bx:
            xv = T.alloc_shared((batch, 128), dtype)
            zv = T.alloc_shared((batch, 128), dtype)
            y = T.alloc_shared((batch, 128), dtype)
            wv = T.alloc_shared((128,), "float32")
            T.annotate_buffer_versions({xv: 2, zv: 2, y: 2})
            T.copy(w, wv)
            for step in T.Pipelined(T.ceildiv(rows, blocks * batch), num_stages=2):
                first = (step * blocks + bx) * batch
                if first < rows:
                    T.copy(x[first : first + batch, :], xv)
                    T.copy(z[first : first + batch, :], zv)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        one = T.simd.vdup(T.float32(1), "float32")
                        for row in range(batch):
                            lo = T.simd.vcvt(
                                T.simd.vld(xv[row, 0], dist="UNPK_B16"), "float32", part=0
                            )
                            hi = T.simd.vcvt(
                                T.simd.vld(xv[row, 64], dist="UNPK_B16"), "float32", part=0
                            )
                            squares = T.simd.vadd(
                                T.simd.vmul(lo, lo, mask), T.simd.vmul(hi, hi, mask), mask
                            )
                            norm = T.simd.vmuls(
                                T.simd.vdupv(T.simd.vcadd(squares, mask), mask),
                                T.float32(1 / 128),
                                mask,
                            )
                            inv = T.simd.vdiv(
                                one,
                                T.simd.vsqrt(T.simd.vadds(norm, T.float32(eps), mask), mask),
                                mask,
                            )
                            for part in range(2):
                                col = part * 64
                                v = T.simd.vcvt(
                                    T.simd.vld(xv[row, col], dist="UNPK_B16"), "float32", part=0
                                )
                                gate = T.simd.vcvt(
                                    T.simd.vld(zv[row, col], dist="UNPK_B16"), "float32", part=0
                                )
                                ex = T.simd.vexp(T.simd.vneg(T.simd.vabs(gate, mask), mask), mask)
                                numerator = T.simd.vsel(
                                    ex, one, T.simd.vcmps(gate, T.float32(0), op="lt")
                                )
                                act = T.simd.vdiv(
                                    numerator, T.simd.vadds(ex, T.float32(1), mask), mask
                                )
                                if silu:
                                    act = T.simd.vmul(act, gate, mask)
                                value = T.simd.vmul(
                                    T.simd.vmul(
                                        T.simd.vmul(v, inv, mask), T.simd.vld(wv[col]), mask
                                    ),
                                    act,
                                    mask,
                                )
                                packed = T.simd.vcvt(
                                    value, dtype, part=0, round="ROUND_R", sat=False
                                )
                                T.simd.vsts(y[row, col], packed, mask, dist="PK_B32")
                    T.copy(y, out[first : first + batch, :], l2_cache_ctrl=0)

    return tilelang.compile(main, target="ascend", out_idx=[3])


@lru_cache(maxsize=16)
def _resident_rows(rows, n, dtype, eps, has_gate, has_bias, silu):
    """Keep the gated values in the UB from the square reduction through scaling."""
    pitch = (n + 63) // 64 * 64
    blocks = min(rows, 64)

    @T.prim_func
    def main(
        x: T.Tensor((rows, n), dtype),
        w: T.Tensor((n,), "float32"),
        bias: T.Tensor((n,), "float32"),
        z: T.Tensor((rows, n), dtype),
        out: T.Tensor((rows, n), dtype),
    ):
        with T.Kernel(blocks) as bx:
            xs = T.alloc_shared((pitch,), dtype)
            zs = T.alloc_shared((pitch,), dtype)
            ws = T.alloc_shared((pitch,), "float32")
            bs = T.alloc_shared((pitch,), "float32")
            scaled = T.alloc_shared((pitch,), "float32")
            ys = T.alloc_shared((pitch,), dtype)
            T.copy(w, ws[:n])
            if has_bias:
                T.copy(bias, bs[:n])
            if n != pitch:
                T.clear(xs)
                if has_gate:
                    T.clear(zs)
            for step in T.serial(T.ceildiv(rows, blocks)):
                row = step * blocks + bx
                if row < rows:
                    T.copy(x[row, :], xs[:n])
                    if has_gate:
                        T.copy(z[row, :], zs[:n])
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.vdup(T.float32(1), "float32")
                        sq = T.simd.alloc_local(1, "float32")
                        sq[0] = T.simd.vdup(T.float32(0), "float32")
                        for j in range(pitch // 64):
                            c = j * 64
                            if dtype == "float32":
                                v = T.simd.vld(xs[c])
                            else:
                                v = T.simd.vcvt(
                                    T.simd.vld(xs[c], dist="UNPK_B16"), "float32", part=0
                                )
                            if has_gate:
                                if dtype == "float32":
                                    gate = T.simd.vld(zs[c])
                                else:
                                    gate = T.simd.vcvt(
                                        T.simd.vld(zs[c], dist="UNPK_B16"), "float32", part=0
                                    )
                                exp = T.simd.vexp(T.simd.vneg(T.simd.vabs(gate, full), full), full)
                                num = T.simd.vsel(
                                    exp, one, T.simd.vcmps(gate, T.float32(0), full, "lt")
                                )
                                act = T.simd.vdiv(num, T.simd.vadds(exp, T.float32(1), full), full)
                                if silu:
                                    act = T.simd.vmul(act, gate, full)
                                v = T.simd.vmul(v, act, full)
                            sq[0] = T.simd.vadd(sq[0], T.simd.vmul(v, v, full), full)
                            T.simd.vsts(scaled[c], v, full)
                        mean = T.simd.vmuls(
                            T.simd.vdupv(T.simd.vcadd(sq[0], full), full), T.float32(1 / n), full
                        )
                        inv = T.simd.vdiv(
                            one, T.simd.vsqrt(T.simd.vadds(mean, T.float32(eps), full), full), full
                        )
                        for j in range(pitch // 64):
                            c = j * 64
                            y = T.simd.vmul(
                                T.simd.vmul(T.simd.vld(scaled[c]), inv, full),
                                T.simd.vld(ws[c]),
                                full,
                            )
                            if has_bias:
                                y = T.simd.vadd(y, T.simd.vld(bs[c]), full)
                            if dtype == "float32":
                                T.simd.vsts(ys[c], y, full)
                            else:
                                packed = T.simd.vcvt(y, dtype, part=0, round="ROUND_R", sat=False)
                                T.simd.vsts(ys[c], packed, full, dist="PK_B32")
                    T.copy(ys[:n], out[row, :], l2_cache_ctrl=0)

    return tilelang.compile(main, target="ascend", out_idx=[4])


class AscendRMSNormGatedKernel(Kernel, RMSNormGatedFwdInterface):
    devices = frozenset({"npu"})
    supported_archs = None

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call

    def forward(self, x, weight, bias, z):
        if not x.numel():
            return torch.empty_like(x)
        call = self.call
        rows = x.numel() // call.n

        def dtype(t):
            return self.dtype_to_str(t.dtype) if t is not None else None

        if (
            call.n == call.group == 128
            and rows % 32 == 0
            and bias is None
            and z is not None
            and z.dtype == x.dtype
            and x.dtype in (torch.bfloat16, torch.float16)
            and weight.dtype == torch.float32
            and call.norm_before_gate
        ):
            kernel = _mixed_weight_rows(rows, dtype(x), call.eps, call.activation == "silu")
            return kernel(x.view(rows, 128), weight, z.view(rows, 128)).view_as(x)
        if (
            call.n == call.group == 128
            and rows % 32 == 0
            and bias is None
            and z is not None
            and z.dtype == x.dtype
            and weight.dtype == x.dtype
            and call.norm_before_gate
        ):
            program = _short_rows(
                rows, dtype(x), dtype(weight), call.eps, call.activation == "silu"
            )
            return program(x.view(rows, 128), weight, z.view(rows, 128)).view_as(x)
        if (
            call.n == call.group
            and 1024 <= call.n <= 8192
            and weight.dtype == torch.float32
            and (bias is None or bias.dtype == torch.float32)
            and (z is None or (not call.norm_before_gate and z.dtype == x.dtype))
            and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
        ):
            flat = x.view(rows, call.n)
            kernel = _resident_rows(
                rows,
                call.n,
                dtype(x),
                call.eps,
                z is not None,
                bias is not None,
                call.activation == "silu",
            )
            return kernel(
                flat,
                weight,
                weight if bias is None else bias,
                flat if z is None else z.view_as(flat),
            ).view_as(x)
        program = _normalize(
            rows,
            call.n,
            call.group,
            dtype(x),
            dtype(weight),
            dtype(bias),
            dtype(z),
            call.eps,
            call.norm_before_gate,
            call.activation == "silu",
        )
        flat = x.view(rows, call.n)
        return program(
            flat, weight, weight if bias is None else bias, flat if z is None else z.view_as(flat)
        ).view_as(x)


register_kernel_type("RMSNormGatedFwdOp", "ascend", AscendRMSNormGatedKernel)
