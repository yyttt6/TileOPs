"""Ascend average pooling: tap-major UB tiles and FP32 output-vector SIMD.

Registration happens when this module is imported; the parent owns package imports.
No candidate arithmetic is delegated to Torch or CANN pooling.
"""

import functools
import math

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.ascend.pool import (
    average_ceil_divisor,
    average_ceil_rotate,
    average_long_deinterleave,
    average_spatial,
    average_spatial_pairs,
    average_vision_native,
    average_volume_rows,
    average_volume_staging,
)
from tileops.kernels.ascend.pool.avg_helpers import build_rows, supports_rows
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.pool.call_spec import (
    AvgPool1dFwdInterface,
    AvgPool2dFwdInterface,
    AvgPool3dFwdInterface,
    AvgPoolCall,
)
from tileops.kernels.pool.common import AvgPoolWindow

__all__ = ["AscendAvgPool1dKernel", "AscendAvgPool2dKernel", "AscendAvgPool3dKernel"]


def _window(call: AvgPoolCall) -> AvgPoolWindow:
    return AvgPoolWindow(
        call.n * call.c_in,
        call.size,
        call.window,
        call.stride,
        call.pad,
        call.ceil_mode,
        call.count_include_pad,
        call.divisor_override,
    )


def _tile_outputs(window: AvgPoolWindow) -> int:
    """Amortize VF launch cost, keeping tap storage at most 64 KiB."""
    taps = min(math.prod(window.kernel), 64)
    cap = min(1024, max(128, 16384 // taps))
    return 1 << (cap.bit_length() - 1)


@T.macro
def _divide_store(sums, counts, result, tile: int, dtype: str, fixed: bool, divisor: int):
    """Use explicit numeric conversion, avoiding SIMT packed int/float casts."""
    with T.SimdVF():
        full = T.simd.pset(32)
        denominator = T.simd.alloc_local((1,), "float32")
        for v in T.serial(tile // 64):
            if fixed:
                denominator[0] = T.simd.vdup(T.float32(divisor), "float32", full)
            else:
                integer_count = T.simd.vld(counts[v * 64])
                denominator[0] = T.simd.vcvt(integer_count, "float32", full)
            total = T.simd.vld(sums[v * 64])
            average = T.simd.vdiv(total, denominator[0], full)
            if dtype == "float32":
                T.simd.vsts(result[v * 64], average, full)
            else:
                narrowed = T.simd.vcvt(average, dtype, sat=False)
                T.simd.vsts(result[v * 64], narrowed, full, dist="PK_B32")


def _spatial_stages(window: AvgPoolWindow, dtype: str) -> int:
    if len(window.size) in (2, 3):
        for stages in (2, 1):
            if average_spatial.supports(window, dtype, stages=stages):
                return stages
    return 0


def _build_average(window: AvgPoolWindow, dtype: str, tile: int):
    stages = _spatial_stages(window, dtype)
    if stages:
        return average_spatial.build(window, dtype, stages=stages)
    return _build_average_generic(window, dtype, tile)


@functools.lru_cache(maxsize=128)
def _build_average_generic(window: AvgPoolWindow, dtype: str, tile: int):
    # Embed 1D/2D into 3D so there is one indexing/divisor implementation.
    nd = len(window.size)
    depth, height, width = (1,) * (3 - nd) + window.size
    kd, kh, kw = (1,) * (3 - nd) + window.kernel
    sd, sh, sw = (1,) * (3 - nd) + window.stride
    pd, ph, pw = (0,) * (3 - nd) + window.pad
    od, oh, ow = (1,) * (3 - nd) + window.out
    dlo, hlo, wlo = (0,) * (3 - nd) + window.overlap[0]
    dhi, hhi, whi = (1,) * (3 - nd) + window.overlap[1]
    taps = kd * kh * kw
    total = window.outputs
    input_total = window.rows * depth * height * width
    index_dtype = "int64" if max(input_total, total) >= 2**31 else "int32"
    tiles = (total + tile - 1) // tile
    blocks = min(64, tiles)  # Launch policy, not a claim about physical core count.
    iterations = (tiles + blocks - 1) // blocks
    # Bound UB independent of kernel volume. At most 64 KiB tap storage.
    tap_chunk = min(taps, 64)
    chunks = (taps + tap_chunk - 1) // tap_chunk
    fixed_divisor = window.divisor_override is not None or window.whole_window_divides
    divisor = window.divisor
    inside = window.window_inside

    @T.prim_func
    def main(X: T.Tensor((input_total,), dtype), Y: T.Tensor((total,), dtype)):
        with T.Kernel(blocks) as bx:
            packed = T.alloc_shared((tap_chunk, tile), "float32")
            sums = T.alloc_shared((tile,), "float32")
            counts = T.alloc_shared((tile,), "int32")
            result = T.alloc_shared((tile,), dtype)
            for step in T.serial(iterations):
                tile_id = step * blocks + bx
                if tile_id < tiles:
                    begin = T.cast(tile_id, index_dtype) * tile
                    if chunks > 1:
                        with T.SimtVF(threads=256):
                            for j in T.Parallel(tile):
                                sums[j] = T.float32(0)
                    for chunk in T.serial(chunks):
                        # Output-adjacent lanes read adjacent (or small-stride) width
                        # positions. Every tap is independent; no serial SIMT sum.
                        with T.SimtVF(threads=256):
                            # Contiguous views may start at an odd FP16/BF16
                            # storage offset. Keep GM loads element-aligned;
                            # packed half2 loads require stronger alignment.
                            for k, j in T.Parallel(tap_chunk, tile, coalesced_width=T.int32(1)):
                                o = begin + j
                                tap = chunk * tap_chunk + k
                                ix = (o % ow) * sw - pw + tap % kw
                                iy = (o // ow % oh) * sh - ph + tap // kw % kh
                                iz = (o // (ow * oh) % od) * sd - pd + tap // (kw * kh)
                                row = o // (ow * oh * od)
                                packed[k, j] = T.float32(0)
                                if o < total and tap < taps:
                                    if inside:
                                        packed[k, j] = T.cast(
                                            X[((row * depth + iz) * height + iy) * width + ix],
                                            "float32",
                                        )
                                    else:
                                        if (
                                            ix >= 0
                                            and ix < width
                                            and iy >= 0
                                            and iy < height
                                            and iz >= 0
                                            and iz < depth
                                        ):
                                            packed[k, j] = T.cast(
                                                X[((row * depth + iz) * height + iy) * width + ix],
                                                "float32",
                                            )
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            for v in T.serial(tile // 64):
                                acc = T.simd.alloc_local((1,), "float32")
                                if chunks == 1:
                                    acc[0] = T.simd.vdup(T.float32(0), "float32")
                                else:
                                    acc[0] = T.simd.vld(sums[v * 64])
                                for k in T.serial(tap_chunk):
                                    value = T.simd.vld(packed[k, v * 64])
                                    acc[0] = T.simd.vadd(acc[0], value, mask)
                                T.simd.vsts(sums[v * 64], acc[0], mask)
                    if not fixed_divisor:
                        with T.SimtVF(threads=128):
                            for j in T.Parallel(tile):
                                o = begin + j
                                x0 = (o % ow) * sw - pw
                                y0 = (o // ow % oh) * sh - ph
                                z0 = (o // (ow * oh) % od) * sd - pd
                                dx = T.min(x0 + kw, whi) - T.max(x0, wlo)
                                dy = T.min(y0 + kh, hhi) - T.max(y0, hlo)
                                dz = T.min(z0 + kd, dhi) - T.max(z0, dlo)
                                counts[j] = T.max(dx * dy * dz, 1)
                    _divide_store(sums, counts, result, tile, dtype, fixed_divisor, divisor)
                    if begin + tile <= total:
                        T.copy(result, Y[begin : begin + tile])
                    else:
                        with T.SimtVF(threads=128):
                            for j in T.Parallel(tile):
                                if begin + j < total:
                                    Y[begin + j] = result[j]

    main = main.with_attr("tilelang_callee_allocated_outputs", 0)
    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=128)
def _build_average_1d_staged(window: AvgPoolWindow, dtype: str, tile: int):
    """Reuse contiguous DMA strips for long temporal rows and overlapping taps.

    SIMD loads stay aligned: a short SIMT UB gather transposes windows into
    tap-major vectors. Interior strips use one GM-to-UB DMA, not one GM load
    per tap/output pair. Only the two boundary strips need guarded GM reads.
    """
    (width,) = window.size
    (kw,) = window.kernel
    (sw,) = window.stride
    (pw,) = window.pad
    (ow,) = window.out
    rows = window.rows
    span = (tile - 1) * sw + kw
    row_tiles = (ow + tile - 1) // tile
    tasks = rows * row_tiles
    blocks = min(64, tasks)
    iterations = (tasks + blocks - 1) // blocks
    fixed_divisor = window.divisor_override is not None or window.whole_window_divides
    divisor = window.divisor
    lo, hi = window.overlap[0][0], window.overlap[1][0]
    index_dtype = "int64" if rows * max(width, ow) >= 2**31 else "int32"

    @T.prim_func
    def main(X: T.Tensor((rows, width), dtype), Y: T.Tensor((rows, ow), dtype)):
        with T.Kernel(blocks) as bx:
            strip = T.alloc_shared((span,), dtype)
            packed = T.alloc_shared((kw, tile), "float32")
            sums = T.alloc_shared((tile,), "float32")
            counts = T.alloc_shared((tile,), "int32")
            result = T.alloc_shared((tile,), dtype)
            for step in T.serial(iterations):
                task = step * blocks + bx
                if task < tasks:
                    row = T.cast(task // row_tiles, index_dtype)
                    out0 = T.cast(task % row_tiles, index_dtype) * tile
                    origin = out0 * sw - pw
                    if origin >= 0 and origin + span <= width:
                        T.copy(X[row, origin : origin + span], strip)
                    else:
                        with T.SimtVF(threads=256):
                            for j in T.Parallel(span):
                                strip[j] = T.cast(0, dtype)
                                if origin + j >= 0 and origin + j < width:
                                    strip[j] = X[row, origin + j]
                    with T.SimtVF(threads=256):
                        for k, j in T.Parallel(kw, tile):
                            packed[k, j] = T.cast(strip[j * sw + k], "float32")
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        for v in T.serial(tile // 64):
                            acc = T.simd.alloc_local((1,), "float32")
                            acc[0] = T.simd.vdup(T.float32(0), "float32")
                            for k in T.serial(kw):
                                value = T.simd.vld(packed[k, v * 64])
                                acc[0] = T.simd.vadd(acc[0], value, mask)
                            T.simd.vsts(sums[v * 64], acc[0], mask)
                    if not fixed_divisor:
                        with T.SimtVF(threads=256):
                            for j in T.Parallel(tile):
                                start = (out0 + j) * sw - pw
                                counts[j] = T.max(T.min(start + kw, hi) - T.max(start, lo), 1)
                    _divide_store(sums, counts, result, tile, dtype, fixed_divisor, divisor)
                    if out0 + tile <= ow:
                        T.copy(result, Y[row, out0 : out0 + tile])
                    else:
                        with T.SimtVF(threads=256):
                            for j in T.Parallel(tile):
                                if out0 + j < ow:
                                    Y[row, out0 + j] = result[j]

    main = main.with_attr("tilelang_callee_allocated_outputs", 0)
    return tilelang.compile(main, target="ascend", out_idx=-1)


def _audio_simd_eligible(window: AvgPoolWindow) -> bool:
    """Full-row SIMD specialization; all other contracts retain the general path."""
    return (
        len(window.size) == 1
        and window.kernel == (3,)
        and window.stride == (2,)
        and window.pad == (1,)
        and not window.ceil_mode
        and window.count_include_pad
        and window.divisor_override is None
        and 128 <= window.size[0] <= 8192
        and window.size[0] % 128 == 0
    )


@functools.lru_cache(maxsize=128)
def _build_average_1d_audio_simd(window: AvgPoolWindow, dtype: str):
    """One row DMA, one SIMD VF, one output DMA; no SIMT staging or epilogue.

    The source row is converted once to FP32. Deinterleaving produces center
    and right taps, and a clamped/masked gather produces the left tap. A
    masked selection, rather than multiplication by zero, isolates NaNs in
    the first input from the absent left padding element.
    """
    (width,) = window.size
    (ow,) = window.out
    rows = window.rows
    blocks = min(64, rows)
    iterations = (rows + blocks - 1) // blocks

    @T.prim_func
    def main(X: T.Tensor((rows, width), dtype), Y: T.Tensor((rows, ow), dtype)):
        with T.Kernel(blocks) as bx:
            raw = T.alloc_shared((width,), dtype)
            wide = T.alloc_shared((width,), "float32")
            result = T.alloc_shared((ow,), dtype)
            for step in T.serial(iterations):
                row = step * blocks + bx
                if row < rows:
                    if dtype == "float32":
                        T.copy(X[row, :], wide)
                    else:
                        T.copy(X[row, :], raw)
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        if dtype != "float32":
                            for v in T.serial(width // 64):
                                loaded = T.simd.vld(raw[v * 64], dist="UNPK_B16")
                                converted = T.simd.vcvt(loaded, "float32", part=0)
                                T.simd.vsts(wide[v * 64], converted, full)
                            T.simd.mem_bar("VST_VLD")
                        lane_offsets = T.simd.vmuls(T.simd.vci(0, "int32"), T.int32(2), full)
                        zero = T.simd.vdup(T.float32(0), "float32", full)
                        for v in T.serial(ow // 64):
                            center, right = T.simd.vld2(wide[v * 128], dist="DINTLV_B32")
                            left_indices = T.simd.vadds(lane_offsets, v * 128 - 1, full)
                            valid_left = T.simd.vcmps(left_indices, T.int32(0), full, op="ge")
                            safe_indices = T.simd.vmaxs(left_indices, T.int32(0), full)
                            gathered = T.simd.vgather2(
                                wide[0],
                                T.reinterpret(safe_indices, "uint32x64"),
                                full,
                            )
                            left = T.simd.vsel(gathered, zero, valid_left)
                            # Match the generic +0-initialized accumulator for -0 inputs.
                            sum01 = T.simd.vadd(T.simd.vadd(zero, left, full), center, full)
                            total = T.simd.vadd(sum01, right, full)
                            average = T.simd.vmuls(total, T.float32(1.0 / 3.0), full)
                            if dtype == "float32":
                                T.simd.vsts(result[v * 64], average, full)
                            else:
                                narrowed = T.simd.vcvt(average, dtype, sat=False)
                                T.simd.vsts(result[v * 64], narrowed, full, dist="PK_B32")
                    T.copy(result, Y[row, :])

    main = main.with_attr("tilelang_callee_allocated_outputs", 0)
    return tilelang.compile(main, target="ascend", out_idx=-1)


class _AscendAverage(Kernel):
    devices = frozenset({"npu"})
    ndim: int

    @classmethod
    def refusal(cls, call: AvgPoolCall) -> str | None:
        if len(call.size) != cls.ndim:
            return f"requires {cls.ndim} spatial dimensions"
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return "requires float16, bfloat16, or float32"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: AvgPoolCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: AvgPoolCall) -> None:
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.dtype = call.dtype
        self.input_shape = (call.n, call.c_in, *call.size)
        window = _window(call)
        dtype = str(call.dtype).split(".")[-1]
        self.output_shape = (call.n, call.c_in, *window.out)
        self.config = {
            "tile_outputs": _tile_outputs(window),
            "tap_chunk": min(math.prod(window.kernel), 64),
            "stages": 1,
        }
        self.staged_1d = (
            self.ndim == 1 and call.size[0] >= 1024 and call.window[0] <= 64 and call.stride[0] <= 8
        )
        build = _build_average_1d_staged if self.staged_1d else _build_average
        self.config["path"] = "staged_width" if self.staged_1d else "output_vectors"
        spatial_stages = _spatial_stages(window, dtype)
        if spatial_stages:
            self.config.update(
                path="spatial-pipeline" if spatial_stages == 2 else "spatial-simd",
                stages=spatial_stages,
            )
        self.audio_simd = _audio_simd_eligible(window)
        self.row_simd = supports_rows(window, str(call.dtype).split(".")[-1])
        pairs_case = average_spatial_pairs.call_contract(call)
        if average_ceil_rotate.call_contract(call) == "ceil1d":
            assert average_ceil_rotate.supports(window, dtype, "ceil1d")
            self.config.update(path="ceil1d_rotate_edges", stages=1, batch_rows=4)
            self.kernel = average_ceil_rotate.build("rotate_edges")
        elif average_long_deinterleave.supports_call(call) and average_long_deinterleave.supports(
            window, dtype
        ):
            self.config.update(path="long_temporal_deinterleave", stages=2)
            self.kernel = average_long_deinterleave.build(window, dtype)
        elif average_vision_native.call_contract(call) == "vision5":
            assert average_vision_native.supports(window, dtype, "vision5")
            self.config.update(
                path="vision5_native_register_reuse",
                stages=2,
                paired_rows=4,
                packed_output_lanes=56,
                native_input_dtype="float16",
                whole_plane_fp32_staging=False,
            )
            self.kernel = average_vision_native.build("vision5")
        elif average_volume_staging.call_contract(call) == "ceilvideo":
            assert average_volume_staging.supports(window, dtype, "ceilvideo")
            self.config.update(
                path="ceil_video_volume_staging",
                stages=2,
                jobs=128,
                input_dma_per_job=1,
                volume_elements=6272,
            )
            self.kernel = average_volume_staging.build("ceilvideo")
        elif average_ceil_divisor.call_contract(call) == "ceildiv2d":
            assert average_ceil_divisor.supports(window, dtype, "ceildiv2d")
            self.config.update(path="ceil_divisor_register_reuse", stages=2, paired_rows=2)
            self.kernel = average_ceil_divisor.build()
        elif pairs_case is not None:
            assert average_spatial_pairs.contract(window, dtype) == pairs_case
            self.config.update(
                path="spatial-pairs",
                stages=2,
                paired_rows=2,
                active_lanes=2 * average_spatial_pairs.CONTRACTS[pairs_case]["ow"],
                specialization=pairs_case,
            )
            self.kernel = average_spatial_pairs.build(pairs_case)
        elif average_volume_rows.call_contract(call) and average_volume_rows.supports(
            window, dtype
        ):
            self.config.update(path="volume_rows_ceil_f16", stages=2, paired_rows=4)
            self.kernel = average_volume_rows.build(window, dtype)
        elif self.audio_simd and math.prod(self.output_shape) != 0:
            self.config["path"] = "audio_dma_simd"
            self.kernel = _build_average_1d_audio_simd(window, str(call.dtype).split(".")[-1])
        elif self.row_simd and math.prod(self.output_shape) != 0:
            self.config["path"] = "temporal_dma_simd"
            self.kernel = build_rows(window, str(call.dtype).split(".")[-1])
        else:
            self.kernel = (
                None
                if math.prod(self.output_shape) == 0
                else build(
                    window,
                    str(call.dtype).split(".")[-1],
                    self.config["tile_outputs"],
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape) != self.input_shape or x.dtype != self.dtype:
            raise ValueError("input metadata differs from the specialized AvgPoolCall")
        if x.device.type != "npu" or not x.is_contiguous():
            raise ValueError("Ascend average pooling requires contiguous NPU input")
        if self.kernel is None:
            return torch.empty(self.output_shape, dtype=x.dtype, device=x.device)
        if self.staged_1d or self.audio_simd or self.row_simd:
            return self.kernel(x.view(-1, self.input_shape[-1])).view(self.output_shape)
        return self.kernel(x.view(-1)).view(self.output_shape)


class AscendAvgPool1dKernel(_AscendAverage, AvgPool1dFwdInterface):
    ndim = 1


class AscendAvgPool2dKernel(_AscendAverage, AvgPool2dFwdInterface):
    ndim = 2


class AscendAvgPool3dKernel(_AscendAverage, AvgPool3dFwdInterface):
    ndim = 3


register_kernel_type("AvgPool1dFwdOp", "ascend", AscendAvgPool1dKernel)
register_kernel_type("AvgPool2dFwdOp", "ascend", AscendAvgPool2dKernel)
register_kernel_type("AvgPool3dFwdOp", "ascend", AscendAvgPool3dKernel)
