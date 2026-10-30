"""Ascend maximum pooling: SIMD window fold and contiguous global reduction.

Only allocation/reshape is performed by Torch. Every value and index is computed
by TileLang. Finite ties keep the first tap; each NaN replaces the prior winner,
so the last visited NaN wins. Device confirmation remains part of acceptance.
"""

import functools
import math

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.ascend.language import simd as S

from tileops.backend import register_kernel_type
from tileops.kernels.ascend.pool import (
    max_alexnet_pairs,
    max_c3d_native4,
    max_c3d_pairs,
    max_dilated,
    max_global_native_guard,
    max_medical_indices_ring,
    max_medical_separable,
    max_medicalnet,
    max_resnet_indices_wh,
    max_resnet_separable,
    max_spatial,
)
from tileops.kernels.ascend.pool.max_helpers import normalize_geometry, staged_1d_plan
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.call_spec import (
    MaxPool1dFwdInterface,
    MaxPool1dIndicesFwdInterface,
    MaxPool2dFwdInterface,
    MaxPool2dIndicesFwdInterface,
    MaxPool3dFwdInterface,
    MaxPool3dIndicesFwdInterface,
)


def _staged_1d_program(
    rows,
    width,
    taps,
    stride,
    dilation,
    outputs,
    dtype,
    indices,
    row_tile,
    padded_outputs,
    blocks=64,
):
    """DMA whole rows, SIMD gather taps, DMA values; no SIMT GM input loads.

    Dispatch guarantees no padding/partial real windows. Padded output lanes
    gather offset zero, so even the final vector cannot read past a staged row.
    """
    tiles = rows // row_tile
    cores = min(blocks, tiles)
    index_size = rows * outputs if indices else 1
    fp32_elements = row_tile * width if dtype != "float32" else 64
    pick_width = padded_outputs if indices else 1

    @T.prim_func
    def main(
        X: T.Tensor((rows * width,), dtype),
        Y: T.Tensor((rows * outputs,), dtype),
        I: T.Tensor((index_size,), "int64"),
    ):
        with T.Kernel(cores) as bx:
            x_ub = T.alloc_shared((row_tile * width,), dtype)
            x_fp32 = T.alloc_shared((fp32_elements,), "float32")
            y_ub = T.alloc_shared((row_tile, padded_outputs), dtype)
            picks = T.alloc_shared((row_tile, pick_width), "uint32")
            for step in T.serial((tiles + cores - 1) // cores):
                tile_id = step * cores + bx
                if tile_id < tiles:
                    begin = tile_id * row_tile * width
                    T.copy(X[begin : begin + row_tile * width], x_ub)
                    with T.SimdVF():
                        full = S.pset(32, "PAT_ALL")
                        if dtype != "float32":
                            for c in range(row_tile * width // 64):
                                converted = S.vcvt(
                                    S.vld(x_ub[c * 64], "UNPK_B16"), "float32", full, part=0
                                )
                                S.vsts(x_fp32[c * 64], converted, full)
                            S.mem_bar("VST_VLD")
                        value = S.alloc_local((1,), "float32")
                        pick = S.alloc_local((1,), "uint32")
                        zero = S.vdup(T.uint32(0), "uint32", full)
                        for r in range(row_tile):
                            for v in range(padded_outputs // 64):
                                # asc_arange requires a signed ramp; gather uses unsigned offsets.
                                signed_positions = S.vci(
                                    T.cast(v * 64, "int32"), "int32", "INC_ORDER"
                                )
                                output_positions = T.reinterpret(signed_positions, "uint32x64")
                                active = S.vcmps(output_positions, T.uint32(outputs), full, "lt")
                                starts = S.vmuls(output_positions, T.uint32(stride), full)
                                value[0] = S.vdup(T.float32(float("-inf")), "float32", full)
                                if indices:
                                    pick[0] = starts
                                for k in T.unroll(taps):
                                    positions = S.vadds(starts, T.uint32(k * dilation), full)
                                    offsets = S.vsel(positions, zero, active)
                                    if dtype == "float32":
                                        incoming = S.vgather2(x_ub[r * width], offsets, full)
                                    else:
                                        incoming = S.vgather2(x_fp32[r * width], offsets, full)
                                    take = S.por(
                                        S.vcmp(incoming, value[0], full, "gt"),
                                        S.vcmp(incoming, incoming, full, "ne"),
                                        full,
                                    )
                                    value[0] = S.vsel(incoming, value[0], take)
                                    if indices:
                                        pick[0] = S.vsel(positions, pick[0], take)
                                if dtype == "float32":
                                    S.vsts(y_ub[r, v * 64], value[0], full)
                                else:
                                    S.vsts(
                                        y_ub[r, v * 64],
                                        S.vcvt(value[0], dtype, full),
                                        full,
                                        "PK_B32",
                                    )
                                if indices:
                                    S.vsts(picks[r, v * 64], pick[0], full)
                    for r in T.serial(row_tile):
                        output_begin = (tile_id * row_tile + r) * outputs
                        T.copy(y_ub[r, :outputs], Y[output_begin : output_begin + outputs])
                    if indices:
                        with T.SimtVF(threads=256):
                            for r, col in T.Parallel(row_tile, outputs):
                                I[(tile_id * row_tile + r) * outputs + col] = T.cast(
                                    picks[r, col], "int64"
                                )

    return main


def _window_program(
    rows, size, window, stride, pad, dilation, out, dtype, indices, tile=256, tap_tile=16, blocks=64
):
    di, hi, wi = size
    kd, kh, kw = window
    sd, sh, sw = stride
    pd, ph, pw = pad
    dd, dh, dw = dilation
    od, oh, ow = out
    spatial = di * hi * wi
    out_spatial = od * oh * ow
    total = rows * out_spatial
    taps = kd * kh * kw
    kt = min(tap_tile, taps)
    chunks = (taps + kt - 1) // kt
    tiles = (total + tile - 1) // tile
    cores = min(blocks, tiles)
    index_size = total if indices else 1

    @T.prim_func
    def main(
        X: T.Tensor((rows * spatial,), dtype),
        Y: T.Tensor((total,), dtype),
        I: T.Tensor((index_size,), "int64"),
    ):
        with T.Kernel(cores) as bx:
            gathered = T.alloc_shared((kt, tile), "float32")
            best_value = T.alloc_shared((tile,), "float32")
            best_tap = T.alloc_shared((tile,), "int32")
            for step in T.serial((tiles + cores - 1) // cores):
                tile_id = step * cores + bx
                if tile_id < tiles:
                    with T.SimdVF():
                        full = S.pset(32, "PAT_ALL")
                        for v in range(tile // 64):
                            S.vsts(
                                best_value[v * 64],
                                S.vdup(T.float32(float("-inf")), "float32", full),
                                full,
                            )
                            if indices:
                                S.vsts(best_tap[v * 64], S.vdup(T.int32(-1), "int32", full), full)
                    for chunk in T.serial(chunks):
                        # Parallel gather, not a per-output scalar window scan.
                        # Adjacent SIMD lanes own adjacent outputs; taps occupy UB rows.
                        with T.SimtVF(threads=256):
                            for k, lane in T.Parallel(kt, tile):
                                q = tile_id * tile + lane
                                tap = chunk * kt + k
                                z = q // (oh * ow) % od * sd - pd + tap // (kh * kw) * dd
                                y = q // ow % oh * sh - ph + tap // kw % kh * dh
                                x = q % ow * sw - pw + tap % kw * dw
                                if (
                                    q < total
                                    and tap < taps
                                    and z >= 0
                                    and z < di
                                    and y >= 0
                                    and y < hi
                                    and x >= 0
                                    and x < wi
                                ):
                                    offset = (
                                        T.cast(q // out_spatial, "int64") * spatial
                                        + (T.cast(z, "int64") * hi + y) * wi
                                        + x
                                    )
                                    gathered[k, lane] = T.cast(X[offset], "float32")
                                else:
                                    gathered[k, lane] = T.float32(float("-inf"))
                        with T.SimdVF():
                            full = S.pset(32, "PAT_ALL")
                            value = S.alloc_local((1,), "float32")
                            pick = S.alloc_local((1,), "int32")
                            for v in range(tile // 64):
                                value[0] = S.vld(best_value[v * 64])
                                if indices:
                                    pick[0] = S.vld(best_tap[v * 64])
                                for k in range(kt):
                                    incoming = S.vld(gathered[k, v * 64])
                                    take = S.por(
                                        S.vcmp(incoming, value[0], full, "gt"),
                                        S.vcmp(incoming, incoming, full, "ne"),
                                        full,
                                    )
                                    value[0] = S.vsel(incoming, value[0], take)
                                    if indices:
                                        pick[0] = S.vsel(
                                            S.vdup(T.cast(chunk * kt + k, "int32"), "int32", full),
                                            pick[0],
                                            take,
                                        )
                                S.vsts(best_value[v * 64], value[0], full)
                                if indices:
                                    S.vsts(best_tap[v * 64], pick[0], full)
                    with T.SimtVF(threads=256):
                        for lane in T.Parallel(tile):
                            q = tile_id * tile + lane
                            if q < total:
                                Y[q] = T.cast(best_value[lane], dtype)
                                if indices:
                                    front = q // (oh * ow) % od * sd - pd
                                    top = q // ow % oh * sh - ph
                                    left = q % ow * sw - pw
                                    tap = best_tap[lane]
                                    if tap >= 0:
                                        z = front + tap // (kh * kw) * dd
                                        y = top + tap // kw % kh * dh
                                        x = left + tap % kw * dw
                                        I[q] = (T.cast(z, "int64") * hi + y) * wi + x
                                    else:
                                        z0 = front + T.ceildiv(T.max(-front, 0), dd) * dd
                                        y0 = top + T.ceildiv(T.max(-top, 0), dh) * dh
                                        x0 = left + T.ceildiv(T.max(-left, 0), dw) * dw
                                        I[q] = (T.cast(z0, "int64") * hi + y0) * wi + x0

    return main


def _global_program(rows, width, dtype, indices, blocks=64):
    """Contiguous full-row reduction: DMA input, SIMD max/argmin/last-NaN."""
    row_tile = 16
    while rows % row_tile:
        row_tile //= 2
    tiles = rows // row_tile
    cores = min(blocks, tiles)
    index_size = rows if indices else 1

    @T.prim_func
    def main(
        X: T.Tensor((rows * width,), dtype),
        Y: T.Tensor((rows,), dtype),
        I: T.Tensor((index_size,), "int64"),
    ):
        with T.Kernel(cores) as bx:
            x_ub = T.alloc_shared((row_tile * width,), dtype)
            picks = T.alloc_shared((64,), "int32")
            for step in T.serial((tiles + cores - 1) // cores):
                tile_id = step * cores + bx
                if tile_id < tiles:
                    begin = tile_id * row_tile * width
                    T.copy(X[begin : begin + row_tile * width], x_ub)
                    with T.SimdVF():
                        full = S.pset(32, "PAT_ALL")
                        one = S.pset(32, "PAT_VL1")
                        value = S.alloc_local((1,), "float32")
                        pick = S.alloc_local((1,), "int32")
                        for r in range(row_tile):
                            value[0] = S.vdup(T.float32(float("-inf")), "float32", full)
                            pick[0] = S.vci(0, "int32", "INC_ORDER")
                            for c in range(width // 64):
                                if dtype == "float32":
                                    incoming = S.vld(x_ub[r * width + c * 64])
                                else:
                                    incoming = S.vcvt(
                                        S.vld(x_ub[r * width + c * 64], "UNPK_B16"),
                                        "float32",
                                        full,
                                        part=0,
                                    )
                                take = S.por(
                                    S.vcmp(incoming, value[0], full, "gt"),
                                    S.vcmp(incoming, incoming, full, "ne"),
                                    full,
                                )
                                value[0] = S.vsel(incoming, value[0], take)
                                pick[0] = S.vsel(S.vci(c * 64, "int32", "INC_ORDER"), pick[0], take)
                            nan_mask = S.vcmp(value[0], value[0], full, "ne")
                            clean = S.vsel(
                                S.vdup(T.float32(float("-inf")), "float32", full),
                                value[0],
                                nan_mask,
                            )
                            maximum = S.vdupv(S.vcmax(clean, full), full)
                            nan_index = S.vdupv(
                                S.vcmax(
                                    S.vsel(pick[0], S.vdup(T.int32(-1), "int32", full), nan_mask),
                                    full,
                                ),
                                full,
                            )
                            has_nan = S.vcmps(nan_index, -1, full, "gt")
                            equals = S.vcmp(value[0], maximum, full, "eq")
                            minimum_index = S.vcmin(
                                S.vsel(pick[0], S.vdup(T.int32(2147483647), "int32", full), equals),
                                full,
                            )
                            final_index = S.vsel(nan_index, minimum_index, has_nan)
                            S.vsts(picks[r], final_index, one, "ONEPT_B32")
                    with T.SimtVF(threads=32):
                        for r in T.Parallel(row_tile):
                            # Read the selected element to preserve signed-zero ties
                            # and NaN payloads rather than a reduction's zero sign.
                            Y[tile_id * row_tile + r] = x_ub[r * width + picks[r]]
                            if indices:
                                I[tile_id * row_tile + r] = T.cast(picks[r], "int64")

    return main


@functools.lru_cache(maxsize=256)
def _build(rows, size, window, stride, pad, dilation, out, dtype, indices):
    global_row = (
        math.prod(out) == 1
        and window == size
        and pad == (0, 0, 0)
        and dilation == (1, 1, 1)
        and math.prod(size) % 64 == 0
        and math.prod(size) <= 2048
    )
    staged = staged_1d_plan(rows, size, window, stride, pad, dilation, out, dtype, indices)
    if global_row:
        program = _global_program(rows, math.prod(size), dtype, indices)
    elif staged is not None:
        program = _staged_1d_program(
            rows,
            size[2],
            window[2],
            stride[2],
            dilation[2],
            out[2],
            dtype,
            indices,
            staged["row_tile"],
            staged["padded_outputs"],
        )
    elif max_spatial.supports(rows, size, window, stride, pad, dilation, out, dtype, indices):
        program = max_spatial.program(
            rows, size, window, stride, pad, dilation, out, dtype, indices
        )
    else:
        program = _window_program(rows, size, window, stride, pad, dilation, out, dtype, indices)
    # Caller allocation gives both outputs proper TorchNPU storage, including
    # the indices tensor consumed by the independent CPU edge oracle.
    program = program.with_attr("tilelang_callee_allocated_outputs", 0)
    return tilelang.compile(program, target="ascend", out_idx=[1, 2])


class _AscendMaxPoolBase(Kernel):
    devices = frozenset({"npu"})
    supported_archs = None
    ndim = 0
    with_indices = False

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__(device_index=call.device.index if call.device is not None else None)
        if len(call.size) != self.ndim or call.n < 0 or call.c_in <= 0:
            raise ValueError("MaxPool rank/batch/channels differ from the kernel contract")
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Ascend MaxPool requires float16, bfloat16 or float32")
        size, window, stride, pad, dilation, out, original_out = normalize_geometry(
            call.size, call.window, call.stride, call.pad, call.dilation, call.ceil_mode
        )
        self.call = call
        self.dtype = call.dtype
        self.output_shape = (call.n, call.c_in, *original_out)
        if call.n == 0:
            self.kernel = None
            self.config = {"protocol": "maximum-spatial-v1", "path": "empty", "launch": False}
            return
        selected_dilated = self.ndim == 1 and max_dilated.supports_call(call)
        selected_alexnet = self.ndim == 2 and max_alexnet_pairs.supports_call(call)
        selected_c3d = self.ndim == 3 and max_c3d_pairs.supports_call(call)
        selected_medicalnet = self.ndim == 3 and max_medicalnet.supports_call(call)
        selected_resnet = self.ndim == 2 and max_resnet_separable.supports_call(
            call, self.with_indices
        )
        selected_resnet_indices = self.ndim == 2 and max_resnet_indices_wh.supports_call(
            call, self.with_indices
        )
        selected_global128 = self.ndim == 1 and max_global_native_guard.supports_call(
            call, self.with_indices
        )
        if selected_dilated:
            self.kernel = max_dilated.build(self.with_indices)
        elif selected_alexnet:
            self.kernel = max_alexnet_pairs.build(str(call.dtype).split(".")[-1], self.with_indices)
        elif selected_c3d:
            self.kernel = (
                max_c3d_native4.build(True) if self.with_indices else max_c3d_pairs.build(False)
            )
        elif selected_medicalnet:
            if self.with_indices:
                assert max_medical_indices_ring.supports_call(call, self.with_indices)
            self.kernel = (
                max_medical_indices_ring.build()
                if self.with_indices
                else max_medical_separable.build()
            )
        elif selected_resnet:
            self.kernel = max_resnet_separable.build(str(call.dtype).split(".")[-1])
        elif selected_resnet_indices:
            self.kernel = max_resnet_indices_wh.build(str(call.dtype).split(".")[-1])
        elif selected_global128:
            self.kernel = max_global_native_guard.build("native_guard", False)
        else:
            self.kernel = _build(
                call.n * call.c_in,
                size,
                window,
                stride,
                pad,
                dilation,
                out,
                str(call.dtype).split(".")[-1],
                self.with_indices,
            )
        staged = staged_1d_plan(
            call.n * call.c_in,
            size,
            window,
            stride,
            pad,
            dilation,
            out,
            str(call.dtype).split(".")[-1],
            self.with_indices,
        )
        spatial = max_spatial.plan(
            call.n * call.c_in,
            size,
            window,
            stride,
            pad,
            dilation,
            out,
            str(call.dtype).split(".")[-1],
            self.with_indices,
        )
        self.config = {
            "protocol": "maximum-spatial-v1",
            "tile_outputs": 256,
            "tap_tile": 16,
            "staged_1d": staged,
            "spatial_candidate_plan": spatial,
        }
        if selected_dilated:
            self.config.update(
                path="dilated1d_dma_simd",
                row_tile=4,
                outputs_per_tile=4092,
                padded_outputs=1024,
                stages=1,
            )
        elif selected_alexnet:
            self.config.update(
                path="alexnet_pairs",
                stages=2,
                active_lanes=54,
                final_pair_lanes=27,
                output_pair_dma_copies=14,
            )
        elif selected_c3d:
            self.config.update(
                path="c3d_pool2_native4_indices" if self.with_indices else "c3d_pool2_pairs_values",
                stages=2,
                active_lanes=112 if self.with_indices else 56,
                final_pair_lanes=112 if self.with_indices else 56,
                input_dma_copies=1,
                output_dma_copies=1,
            )
        elif selected_resnet:
            self.config.update(
                path="resnet_separable_wh_values",
                stages=2,
                native_lanes=128,
                output_pair_lanes=112,
                explicit_ub_bytes=65024,
            )
        elif selected_medicalnet and not self.with_indices:
            self.config.update(
                path="medicalnet_separable_ring_values",
                stages=2,
                input_planes_per_channel=32,
                packed_row_pair=True,
                explicit_ub_bytes=137984,
            )
        elif selected_medicalnet:
            self.config.update(
                path="medicalnet_separable_ring_indices",
                stages=2,
                input_planes_per_channel=32,
                packed_row_pair=True,
                explicit_ub_bytes=189952,
            )
        elif selected_resnet_indices:
            self.config.update(
                path="resnet_separable_wh_indices",
                stages=2,
                native_lanes=128,
                output_pair_lanes=112,
            )
        elif selected_global128:
            self.config.update(
                path="global128_native_guard256_l2keep_values",
                l2_cache_ctrl=4,
                **max_global_native_guard.CONFIGS["native_guard"],
            )

    def forward(self, x):
        if (
            x.device.type != "npu"
            or x.dtype != self.dtype
            or tuple(x.shape) != (self.call.n, self.call.c_in, *self.call.size)
        ):
            raise ValueError("Ascend MaxPool input differs from its specialization")
        if not x.is_contiguous():
            raise ValueError("MaxPool kernel requires the public op's contiguous input")
        if self.kernel is None:
            values = x.new_empty(self.output_shape)
            if self.with_indices:
                return values, x.new_empty(self.output_shape, dtype=torch.int64)
            return values
        values, indices = self.kernel(x.reshape(-1))
        values = values.reshape(self.output_shape)
        if self.with_indices:
            return values, indices.reshape(self.output_shape)
        return values


class AscendMaxPool1dKernel(_AscendMaxPoolBase, MaxPool1dFwdInterface):
    ndim = 1


class AscendMaxPool1dIndicesKernel(_AscendMaxPoolBase, MaxPool1dIndicesFwdInterface):
    ndim = 1
    with_indices = True


class AscendMaxPool2dKernel(_AscendMaxPoolBase, MaxPool2dFwdInterface):
    ndim = 2


class AscendMaxPool2dIndicesKernel(_AscendMaxPoolBase, MaxPool2dIndicesFwdInterface):
    ndim = 2
    with_indices = True


class AscendMaxPool3dKernel(_AscendMaxPoolBase, MaxPool3dFwdInterface):
    ndim = 3


class AscendMaxPool3dIndicesKernel(_AscendMaxPoolBase, MaxPool3dIndicesFwdInterface):
    ndim = 3
    with_indices = True


for _name, _kernel in (
    ("MaxPool1dFwdOp", AscendMaxPool1dKernel),
    ("MaxPool1dIndicesFwdOp", AscendMaxPool1dIndicesKernel),
    ("MaxPool2dFwdOp", AscendMaxPool2dKernel),
    ("MaxPool2dIndicesFwdOp", AscendMaxPool2dIndicesKernel),
    ("MaxPool3dFwdOp", AscendMaxPool3dKernel),
    ("MaxPool3dIndicesFwdOp", AscendMaxPool3dIndicesKernel),
):
    register_kernel_type(_name, "ascend", _kernel)
