"""Ascend adaptive pooling: batched SIMD reductions with exact argmax semantics.

The 7x7 global workloads use contiguous GM->UB plane copies. Other bins are
gathered in parallel, then reduced in SIMD; bins larger than 512 elements stream
through bounded UB tiles. No Torch arithmetic or native pooling is dispatched.
"""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.ascend.pool.adaptive_helpers import adaptive_tiling
from tileops.kernels.ascend.pool.adaptive_spatial import configuration as spatial_configuration
from tileops.kernels.ascend.pool.adaptive_spatial import spatial_program
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.pool.call_spec import (
    AdaptiveAvgPool2dFwdInterface,
    AdaptiveMaxPool2dFwdInterface,
    AdaptiveMaxPool2dIndicesFwdInterface,
    AdaptivePool2dCall,
)

__all__ = [
    "AscendAdaptiveAvgPool2dKernel",
    "AscendAdaptiveMaxPool2dKernel",
    "AscendAdaptiveMaxPool2dWithIndicesKernel",
]


@T.macro
def _extreme_rows(source, output, rows: int, cols: int, maximum: bool):
    """Explicit row extrema: bypass the backend's reduce_min -> vcmax lowering.

    Start from actual data instead of finite +/-FLT_MAX identities: an all--inf
    row must remain -inf. Only lane zero of each reduction is stored to UB.
    """
    with T.SimdVF():
        full = T.simd.pset(32)
        one = T.simd.pset(32, "PAT_VL1")
        for r in T.serial(rows):
            acc = T.simd.alloc_local((1,), "float32")
            if maximum:
                acc[0] = T.simd.vcmax(T.simd.vld(source[r, 0]), full)
            else:
                acc[0] = T.simd.vcmin(T.simd.vld(source[r, 0]), full)
            for v in T.serial(1, cols // 64):
                if maximum:
                    acc[0] = T.simd.vmax(
                        acc[0], T.simd.vcmax(T.simd.vld(source[r, v * 64]), full), full
                    )
                else:
                    acc[0] = T.simd.vmin(
                        acc[0], T.simd.vcmin(T.simd.vld(source[r, v * 64]), full), full
                    )
            T.simd.vsts(output[r], acc[0], one, "ONEPT_B32")


def _global_avg_simd_program(planes, spatial, dtype):
    """Small global bins: aligned DMA rows, one SIMD reduction per plane.

    Pad UB rows (not GM reads) to 64 elements, so the 49-element canonical
    planes never require an unaligned vector load or a SIMT packing pass.
    More planes per VF amortize launch costs; keep 64 blocks for the CHW case.
    """
    rows = 64 if planes % 64 == 0 and planes >= 4096 else 32 if planes % 32 == 0 else 16
    tiles = planes // rows
    blocks = min(64, tiles)

    @T.prim_func
    def main(
        X: T.Tensor((planes, spatial), dtype),
        Y: T.Tensor((planes,), dtype),
        Indices: T.Tensor((1,), "int64"),
    ):
        with T.Kernel(blocks) as core:
            staged = T.alloc_shared((rows, 64), dtype)
            sums = T.alloc_shared((64,), "float32")
            result = T.alloc_shared((64,), dtype)
            for step in T.serial((tiles + blocks - 1) // blocks):
                tile = step * blocks + core
                if tile < tiles:
                    begin = tile * rows
                    T.copy(X[begin : begin + rows, :], staged[:, :spatial])
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.pset(32, "PAT_VL1")
                        valid = T.simd.update_mask(spatial, 32)
                        T.simd.vsts(sums[0], T.simd.vdup(0.0, "float32", full), full)
                        for r in T.serial(rows):
                            value = T.simd.vcvt(
                                T.simd.vld(staged[r, 0], dist="UNPK_B16"),
                                "float32",
                                valid,
                                part=0,
                            )
                            total = T.simd.vcadd(value, valid)
                            T.simd.vsts(sums[r], total, one, "ONEPT_B32")
                        T.simd.mem_bar("VST_VLD")
                        averages = T.simd.vmuls(T.simd.vld(sums[0]), 1.0 / spatial, full)
                        narrowed = T.simd.vcvt(averages, dtype, full, sat=False)
                        T.simd.vsts(result[0], narrowed, full, "PK_B32")
                    T.copy(result[:rows], Y[begin : begin + rows])

    return main


def _global_max_simd_program(planes, dtype, with_indices):
    """7x7 global max: all selection stays in SIMD registers.

    Numeric ranks select the first finite tie and the last NaN. The selected
    original value is shuffled from the input register, retaining signed zero.
    Only the optional int32->int64 index store uses a short SIMT VF.
    """
    rows = 64 if planes % 64 == 0 and planes >= 4096 else 32 if planes % 32 == 0 else 16
    tiles = planes // rows
    blocks = min(64, tiles)
    neg_inf = float("-inf")

    @T.prim_func
    def main(
        X: T.Tensor((planes, 49), dtype),
        Y: T.Tensor((planes,), dtype),
        Indices: T.Tensor((planes if with_indices else 1,), "int64"),
    ):
        with T.Kernel(blocks) as core:
            staged = T.alloc_shared((rows, 64), dtype)
            chosen_values = T.alloc_shared((64,), "float32")
            result = T.alloc_shared((64,), dtype)
            if with_indices:
                chosen_indices = T.alloc_shared((rows,), "int32")
            for step in T.serial((tiles + blocks - 1) // blocks):
                tile = step * blocks + core
                if tile < tiles:
                    begin = tile * rows
                    T.copy(X[begin : begin + rows, :], staged[:, :49])
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.pset(32, "PAT_VL1")
                        valid = T.simd.update_mask(49, 32)
                        lane = T.simd.vci(0, "int32", "INC_ORDER")
                        ordinal = T.simd.vcvt(lane, "float32", full)
                        nan_rank = T.simd.vadds(T.simd.vneg(ordinal, full), -1.0, full)
                        missing = T.simd.vdup(65.0, "float32", full)
                        negative_infinity = T.simd.vdup(neg_inf, "float32", full)
                        T.simd.vsts(chosen_values[0], T.simd.vdup(0.0, "float32", full), full)
                        for r in T.serial(rows):
                            value = T.simd.vcvt(
                                T.simd.vld(staged[r, 0], dist="UNPK_B16"),
                                "float32",
                                valid,
                                part=0,
                            )
                            is_nan = T.simd.vcmp(value, value, full, "ne")
                            finite = T.simd.vsel(negative_infinity, value, is_nan)
                            maximum = T.simd.vcmax(finite, valid)
                            broadcast_maximum = T.simd.vdupv(maximum, full)
                            tied = T.simd.vcmp(value, broadcast_maximum, full, "eq")
                            finite_rank = T.simd.vsel(ordinal, missing, tied)
                            ranked = T.simd.vsel(nan_rank, finite_rank, is_nan)
                            selected_rank = T.simd.vcmin(ranked, valid)
                            selected_nan = T.simd.vcmps(selected_rank, 0.0, one, "lt")
                            decoded_nan = T.simd.vneg(T.simd.vadds(selected_rank, 1.0, one), one)
                            location_float = T.simd.vsel(decoded_nan, selected_rank, selected_nan)
                            location = T.simd.vcvt(location_float, "int32", one)
                            original = T.simd.vselr(value, location)
                            T.simd.vsts(chosen_values[r], original, one, "ONEPT_B32")
                            if with_indices:
                                T.simd.vsts(chosen_indices[r], location, one, "ONEPT_B32")
                        T.simd.mem_bar("VST_VLD")
                        packed = T.simd.vcvt(T.simd.vld(chosen_values[0]), dtype, full, sat=False)
                        T.simd.vsts(result[0], packed, full, "PK_B32")
                    T.copy(result[:rows], Y[begin : begin + rows])
                    if with_indices:
                        with T.SimtVF(threads=128):
                            for r in T.Parallel(rows):
                                Indices[begin + r] = T.cast(chosen_indices[r], "int64")

    return main


def _identity_program(planes, spatial, dtype, with_indices):
    total = planes * spatial
    tile = 4096
    tiles = (total + tile - 1) // tile
    blocks = min(64, max(1, tiles))

    @T.prim_func
    def main(
        X: T.Tensor((planes, spatial), dtype),
        Y: T.Tensor((total,), dtype),
        Indices: T.Tensor((total if with_indices else 1,), "int64"),
    ):
        with T.Kernel(blocks) as core:
            staged = T.alloc_shared((tile,), dtype)
            flat = T.reshape(X, (total,))
            for step in T.serial((tiles + blocks - 1) // blocks):
                begin = (step * blocks + core) * tile
                if begin < total:
                    if begin + tile <= total:
                        T.copy(flat[begin : begin + tile], staged)
                        T.copy(staged, Y[begin : begin + tile])
                    else:
                        with T.SimtVF(threads=256):
                            # Offset views need element-aligned tail loads.
                            for j in T.Parallel(tile, coalesced_width=T.int32(1)):
                                if begin + j < total:
                                    Y[begin + j] = flat[begin + j]
                    if with_indices:
                        with T.SimtVF(threads=256):
                            for j in T.Parallel(tile):
                                if begin + j < total:
                                    Indices[begin + j] = T.cast((begin + j) % spatial, "int64")

    return main


def adaptive_program(planes, h, w, oh, ow, dtype, maximum, with_indices):
    """Return a PrimFunc, also usable by the parent's offline lowering checks."""
    if spatial_configuration(planes, h, w, oh, ow, dtype)["path"] == "plane-simd-bins":
        return spatial_program(planes, h, w, oh, ow, dtype, maximum, with_indices)
    if oh == h and ow == w:
        return _identity_program(planes, h * w, dtype, with_indices)
    if not maximum and oh == ow == 1 and h * w <= 64 and planes % 16 == 0:
        return _global_avg_simd_program(planes, h * w, dtype)
    if maximum and oh == ow == 1 and h == w == 7 and planes % 16 == 0:
        return _global_max_simd_program(planes, dtype, with_indices)
    rows, cols, kh, kw = adaptive_tiling(h, w, oh, ow)
    total = planes * oh * ow
    tiles = (total + rows - 1) // rows
    blocks = min(64, max(1, tiles))  # Scheduling cap, not a hardware core count.
    segments = (kh * kw + cols - 1) // cols
    global_dma = oh == ow == 1 and h * w <= cols and planes % rows == 0
    neg_inf = float("-inf")

    @T.prim_func
    def main(
        X: T.Tensor((planes, h * w), dtype),
        Y: T.Tensor((total,), dtype),
        Indices: T.Tensor((total if with_indices else 1,), "int64"),
    ):
        with T.Kernel(blocks) as core:
            raw = T.alloc_shared((rows, cols), "float32")
            work = T.alloc_shared((rows, cols), "float32")
            reduced = T.alloc_shared((rows,), "float32")
            if maximum:
                rank_work = T.alloc_shared((rows, cols), "float32")
                ranks = T.alloc_shared((rows,), "float32")
            else:
                avg_values = T.alloc_shared((64,), "float32")
                avg_counts = T.alloc_shared((64,), "int32")
                avg_output = T.alloc_shared((64,), dtype)
            state = T.alloc_shared((rows,), "float32")
            best_index = T.alloc_shared((rows,), "int64")
            if global_dma:
                compact = T.alloc_shared((rows, h * w), dtype)
            for step in T.serial((tiles + blocks - 1) // blocks):
                tile = step * blocks + core
                if tile < tiles:
                    if global_dma:
                        T.copy(X[tile * rows : (tile + 1) * rows, :], compact)
                    with T.SimtVF(threads=128):
                        for r in T.Parallel(rows):
                            state[r] = neg_inf if maximum else 0.0
                            best_index[r] = -1
                    for segment in T.serial(segments):
                        with T.SimtVF(threads=256):
                            for r, k in T.Parallel(rows, cols):
                                out = tile * rows + r
                                ph = (out // ow) % oh
                                pw = out % ow
                                hs = ph * h // oh
                                he = ((ph + 1) * h + oh - 1) // oh
                                ws = pw * w // ow
                                we = ((pw + 1) * w + ow - 1) // ow
                                ordinal = segment * cols + k
                                ih = hs + ordinal // kw
                                iw = ws + ordinal % kw
                                raw[r, k] = neg_inf if maximum else 0.0
                                if out < total and ordinal < kh * kw and ih < he and iw < we:
                                    if global_dma:
                                        raw[r, k] = T.cast(compact[r, ordinal], "float32")
                                    else:
                                        raw[r, k] = T.cast(
                                            X[out // (oh * ow), ih * w + iw], "float32"
                                        )
                                if maximum:
                                    # Never rely on the target's reduction NaN policy.
                                    work[r, k] = T.if_then_else(
                                        T.isnan(raw[r, k]), neg_inf, raw[r, k]
                                    )
                                else:
                                    work[r, k] = raw[r, k]
                        if maximum:
                            _extreme_rows(work, reduced, rows, cols, True)
                        else:
                            with T.SimdVF():
                                T.reduce_sum(work, reduced, dim=1)
                        if maximum:
                            with T.SimtVF(threads=256):
                                for r, k in T.Parallel(rows, cols):
                                    out = tile * rows + r
                                    ph = (out // ow) % oh
                                    pw = out % ow
                                    ordinal = segment * cols + k
                                    ih = ph * h // oh + ordinal // kw
                                    iw = pw * w // ow + ordinal % kw
                                    valid = (out < total) & (ordinal < kh * kw)
                                    valid = valid & (ih < ((ph + 1) * h + oh - 1) // oh)
                                    valid = valid & (iw < ((pw + 1) * w + ow - 1) // ow)
                                    # Negative ranks select the last NaN; nonnegative
                                    # ranks select the first finite tie (including -inf).
                                    rank_work[r, k] = T.if_then_else(
                                        valid,
                                        T.if_then_else(
                                            T.isnan(raw[r, k]),
                                            T.cast(-k - 1, "float32"),
                                            T.if_then_else(
                                                raw[r, k] == reduced[r],
                                                T.cast(k, "float32"),
                                                cols + 1.0,
                                            ),
                                        ),
                                        cols + 1.0,
                                    )
                            _extreme_rows(rank_work, ranks, rows, cols, False)
                            with T.SimtVF(threads=128):
                                for r in T.Parallel(rows):
                                    rank = T.cast(ranks[r], "int32")
                                    pick = T.if_then_else(rank < 0, -rank - 1, rank)
                                    if pick < cols:
                                        value = raw[r, pick]
                                        if best_index[r] < 0 or T.isnan(value) or value > state[r]:
                                            out = tile * rows + r
                                            ordinal = segment * cols + pick
                                            ih = ((out // ow) % oh) * h // oh + ordinal // kw
                                            iw = (out % ow) * w // ow + ordinal % kw
                                            state[r] = value
                                            best_index[r] = T.cast(ih, "int64") * w + iw
                        else:
                            with T.SimtVF(threads=128):
                                for r in T.Parallel(rows):
                                    state[r] = state[r] + reduced[r]
                    if maximum:
                        with T.SimtVF(threads=128):
                            for r in T.Parallel(rows):
                                out = tile * rows + r
                                if out < total:
                                    Y[out] = T.cast(state[r], dtype)
                                    if with_indices:
                                        Indices[out] = best_index[r]
                    else:
                        # Keep counts integer in SIMT; perform their numeric
                        # conversion explicitly in SIMD, avoiding vector bitcasts.
                        # Initialize a complete vector, including unused rows.
                        with T.SimtVF(threads=128):
                            for r in T.Parallel(64):
                                avg_values[r] = T.float32(0)
                                avg_counts[r] = 1
                                out = tile * rows + r
                                if r < rows and out < total:
                                    ph = (out // ow) % oh
                                    pw = out % ow
                                    nh = ((ph + 1) * h + oh - 1) // oh - ph * h // oh
                                    nw = ((pw + 1) * w + ow - 1) // ow - pw * w // ow
                                    avg_values[r] = state[r]
                                    avg_counts[r] = nh * nw
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            denominator = T.simd.vcvt(T.simd.vld(avg_counts[0]), "float32", full)
                            quotient = T.simd.vdiv(T.simd.vld(avg_values[0]), denominator, full)
                            converted = T.simd.vcvt(quotient, dtype, full, sat=False)
                            T.simd.vsts(avg_output[0], converted, full, "PK_B32")
                        if (tile + 1) * rows <= total:
                            T.copy(avg_output[:rows], Y[tile * rows : (tile + 1) * rows])
                        else:
                            with T.SimtVF(threads=128):
                                for r in T.Parallel(rows):
                                    if tile * rows + r < total:
                                        Y[tile * rows + r] = avg_output[r]

    return main


@functools.lru_cache(maxsize=64)
def _build(planes, h, w, oh, ow, dtype, maximum, with_indices):
    program = adaptive_program(planes, h, w, oh, ow, dtype, maximum, with_indices)
    return tilelang.compile(
        program.with_attr("tilelang_callee_allocated_outputs", 0),
        target="ascend",
        out_idx=[],
    )


class _AscendAdaptiveBase(Kernel):
    devices = frozenset({"npu"})
    maximum = False
    with_indices = False

    @classmethod
    def entry_for(cls, call: AdaptivePool2dCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: AdaptivePool2dCall):
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        self.kernel = None
        if call.n * call.c_in:
            self.kernel = _build(
                call.n * call.c_in,
                call.h_in,
                call.w_in,
                call.out_h,
                call.out_w,
                str(call.dtype).split(".")[-1],
                self.maximum,
                self.with_indices,
            )

    def forward(self, x: torch.Tensor):
        c = self.call
        shape = (*x.shape[:-2], c.out_h, c.out_w)
        y = torch.empty(shape, dtype=x.dtype, device=x.device)
        indices = torch.empty(
            shape if self.with_indices else (1,), dtype=torch.int64, device=x.device
        )
        if self.kernel is not None:
            self.kernel(x.view(c.n * c.c_in, c.h_in * c.w_in), y.view(-1), indices.view(-1))
        return (y, indices) if self.with_indices else y


class AscendAdaptiveAvgPool2dKernel(_AscendAdaptiveBase, AdaptiveAvgPool2dFwdInterface):
    pass


class AscendAdaptiveMaxPool2dKernel(_AscendAdaptiveBase, AdaptiveMaxPool2dFwdInterface):
    maximum = True


class AscendAdaptiveMaxPool2dWithIndicesKernel(
    _AscendAdaptiveBase, AdaptiveMaxPool2dIndicesFwdInterface
):
    maximum = True
    with_indices = True


register_kernel_type("AdaptiveAvgPool2dFwdOp", "ascend", AscendAdaptiveAvgPool2dKernel)
register_kernel_type("AdaptiveMaxPool2dFwdOp", "ascend", AscendAdaptiveMaxPool2dKernel)
register_kernel_type(
    "AdaptiveMaxPool2dIndicesFwdOp", "ascend", AscendAdaptiveMaxPool2dWithIndicesKernel
)
