"""DMA-staged spatial pooling with SIMD gathers and FP32 accumulation.

One job stages the depth slices used by one output plane. The width reduction
uses UB gathers; no SIMT launch or GM scalar gather occurs in this specialization.
"""

import functools
import math

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S


def supports(window, dtype, stages=1):
    if stages not in (1, 2) or len(window.size) not in (2, 3) or not window.rows:
        return False
    h, w = window.size[-2:]
    kd = window.kernel[0] if len(window.size) == 3 else 1
    oh, ow = window.out[-2:]
    padded = ((h * w + 63) // 64) * 64 + 128
    item = 4 if dtype == "float32" else 2
    ub = kd * padded * (stages * item + 4) + stages * oh * 64 * item
    return (
        0 < ow <= 64
        and oh > 0
        and kd <= 8
        and window.kernel[-1] <= 8
        and math.prod(window.kernel) <= 125
        and ub <= (192 if stages == 1 else 240) * 1024
        and window.rows * math.prod(window.size) < 2**31
        and window.outputs < 2**31
    )


@functools.lru_cache(maxsize=64)
def build(window, dtype, stages=1):
    assert supports(window, dtype, stages)
    assert stages in (1, 2)
    nd = len(window.size)
    d, h, w = (1,) * (3 - nd) + window.size
    kd, kh, kw = (1,) * (3 - nd) + window.kernel
    sd, sh, sw = (1,) * (3 - nd) + window.stride
    pd, ph, pw = (0,) * (3 - nd) + window.pad
    od, oh, ow = (1,) * (3 - nd) + window.out
    dlo, hlo, wlo = (0,) * (3 - nd) + window.overlap[0]
    dhi, hhi, whi = (1,) * (3 - nd) + window.overlap[1]
    plane = h * w
    deinterleave = kw == sw == 2 and pw == 0 and w % 8 == 0
    width_three = kw == 3 and sw == 2 and pw == 1 and w % 8 == 0 and ow == w // 2
    reuse_rows = (
        nd == 2 and width_three and kh == 3 and sh == 2 and ph == 1 and h % 2 == 0 and oh == h // 2
    )
    stride_plane = ((plane + 63) // 64) * 64 + (128 if deinterleave or width_three else 0)
    jobs = window.rows * od
    blocks = min(64, jobs)
    total = window.rows * d * plane
    fixed = window.divisor_override is not None or window.whole_window_divides
    divisor = window.divisor
    item = 4 if dtype == "float32" else 2
    assert kd * stride_plane * (stages * item + 4) + stages * oh * 64 * item <= 240 * 1024

    @T.prim_func
    def main(X: T.Tensor((total,), dtype), Y: T.Tensor((jobs, oh, ow), dtype)):
        with T.Kernel(blocks) as core:
            raw = T.alloc_shared((64 if dtype == "float32" else kd * stride_plane,), dtype)
            data = T.alloc_shared((kd * stride_plane,), "float32")
            output = T.alloc_shared((oh, 64), dtype)
            # Raw inputs live from MTE2 completion to SIMD completion; outputs
            # live from SIMD completion to MTE3 completion. FP32 scratch is
            # consumed within one SIMD stage; for FP32 it also receives DMA and
            # therefore needs the same version count as the raw input.
            if dtype == "float32":
                T.annotate_buffer_versions({data: stages, output: stages})
            else:
                T.annotate_buffer_versions({raw: stages, data: 1, output: stages})
            for step in T.Pipelined((jobs + blocks - 1) // blocks, num_stages=stages):
                job = step * blocks + core
                if job < jobs:
                    row = job // od
                    zstart = job % od * sd - pd
                    for dz in T.serial(kd):
                        iz = zstart + dz
                        if iz >= 0 and iz < d:
                            begin = (row * d + iz) * plane
                            if dtype == "float32":
                                T.copy(
                                    X[begin : begin + plane],
                                    data[dz * stride_plane : dz * stride_plane + plane],
                                )
                            else:
                                T.copy(
                                    X[begin : begin + plane],
                                    raw[dz * stride_plane : dz * stride_plane + plane],
                                )
                    with T.SimdVF():
                        full = S.pset(32)
                        zero = S.vdup(T.float32(0), "float32", full)
                        izero = S.vdup(T.int32(0), "int32", full)
                        lane = S.vci(0, "int32")
                        out_mask = S.vcmps(lane, T.int32(ow), full, "lt")
                        offsets = S.alloc_local((kw,), "uint32")
                        masks = S.alloc_local((kw,), "bool")
                        if not deinterleave:
                            for kx in T.Unroll(kw, explicit=True):
                                xpos = S.vadds(
                                    S.vmuls(lane, T.int32(sw), full), T.int32(kx - pw), full
                                )
                                masks[kx] = S.pand(
                                    S.pand(
                                        S.vcmps(xpos, T.int32(0), full, "ge"),
                                        S.vcmps(xpos, T.int32(w), full, "lt"),
                                        full,
                                    ),
                                    out_mask,
                                    full,
                                )
                                offsets[kx] = T.reinterpret(
                                    S.vsel(xpos, izero, masks[kx]), "uint32x64"
                                )
                        for dz in T.serial(kd):
                            for part in T.serial(stride_plane // 64):
                                if zstart + dz >= 0 and zstart + dz < d:
                                    if dtype != "float32":
                                        converted = S.vcvt(
                                            S.vld(
                                                raw[dz * stride_plane + part * 64], dist="UNPK_B16"
                                            ),
                                            "float32",
                                            full,
                                            part=0,
                                        )
                                        S.vsts(data[dz * stride_plane + part * 64], converted, full)
                                else:
                                    S.vsts(data[dz * stride_plane + part * 64], zero, full)
                        S.mem_bar("VST_VLD")
                        acc = S.alloc_local((1,), "float32")
                        last_row = S.alloc_local((3,), "float32")
                        for cached in T.Unroll(3, explicit=True):
                            last_row[cached] = zero
                        for oy in T.serial(oh):
                            acc[0] = zero
                            if reuse_rows:
                                # Adjacent 3x3/stride2 windows share one input
                                # row. Retain its three taps, preserving FP32
                                # tap-order accumulation under cancellation.
                                for cached in T.Unroll(3, explicit=True):
                                    acc[0] = S.vadd(acc[0], last_row[cached], full)
                            for dz in T.serial(kd):
                                for ky in T.serial(1 if reuse_rows else 0, kh):
                                    iy = oy * sh - ph + ky
                                    if iy >= 0 and iy < h:
                                        if deinterleave:
                                            # The allocation includes a full extra vector pair,
                                            # so the final row's unmasked load stays in UB.
                                            even, odd = S.vld2(
                                                data[dz * stride_plane + iy * w],
                                                dist="DINTLV_B32",
                                            )
                                            acc[0] = S.vadd(S.vadd(acc[0], even, full), odd, full)
                                        elif width_three:
                                            center, right = S.vld2(
                                                data[dz * stride_plane + iy * w],
                                                dist="DINTLV_B32",
                                            )
                                            left_raw = S.vgather2(
                                                data[dz * stride_plane + iy * w], offsets[0], full
                                            )
                                            left = S.vsel(left_raw, zero, masks[0])
                                            acc[0] = S.vadd(
                                                S.vadd(S.vadd(acc[0], left, full), center, full),
                                                right,
                                                full,
                                            )
                                            if reuse_rows:
                                                last_row[0] = left
                                                last_row[1] = center
                                                last_row[2] = right
                                        else:
                                            for kx in T.Unroll(kw, explicit=True):
                                                if w % 8 == 0:
                                                    values = S.vgather2(
                                                        data[dz * stride_plane + iy * w],
                                                        offsets[kx],
                                                        full,
                                                    )
                                                else:
                                                    values = S.vgather2(
                                                        data[dz * stride_plane],
                                                        S.vadds(
                                                            offsets[kx],
                                                            T.cast(iy * w, "uint32"),
                                                            full,
                                                        ),
                                                        full,
                                                    )
                                                acc[0] = S.vadd(
                                                    acc[0], S.vsel(values, zero, masks[kx]), full
                                                )
                            if fixed:
                                average = S.vmuls(acc[0], T.float32(1.0 / divisor), full)
                            else:
                                xstart = S.vadds(
                                    S.vmuls(lane, T.int32(sw), full), T.int32(-pw), full
                                )
                                count_x = S.vsub(
                                    S.vmins(S.vadds(xstart, T.int32(kw), full), T.int32(whi), full),
                                    S.vmaxs(xstart, T.int32(wlo), full),
                                    full,
                                )
                                count_y = T.min(oy * sh - ph + kh, hhi) - T.max(oy * sh - ph, hlo)
                                count_z = T.min(zstart + kd, dhi) - T.max(zstart, dlo)
                                counts = S.vmaxs(
                                    S.vmuls(count_x, T.cast(count_y * count_z, "int32"), full),
                                    T.int32(1),
                                    full,
                                )
                                average = S.vdiv(acc[0], S.vcvt(counts, "float32", full), full)
                            if dtype == "float32":
                                S.vsts(output[oy, 0], average, out_mask)
                            else:
                                narrowed = S.vcvt(average, dtype, full, sat=False)
                                S.vsts(output[oy, 0], narrowed, out_mask, "PK_B32")
                    T.copy(output[:, :ow], Y[job, :, :])

    return tilelang.compile(
        main.with_attr("tilelang_callee_allocated_outputs", 0), target="ascend", out_idx=-1
    )
