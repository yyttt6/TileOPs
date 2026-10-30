"""Experimental DMA/SIMD spatial max pool; opt-in only, no registration.

build/supports take the same normalized 3D geometry as maximum._build. Unit
dilation only; unsupported geometries must use the existing generic builder.
No GM input loads in SIMT. SIMT is used only to widen selected indices to int64.
"""

import functools
import math


def plan(rows, size, window, stride, pad, dilation, out, dtype, indices):
    if (
        dtype not in ("float16", "bfloat16", "float32")
        or rows <= 0
        or any(len(x) != 3 for x in (size, window, stride, pad, dilation, out))
    ):
        return None
    d, h, w = size
    kd, kh, kw = window
    od, oh, ow = out
    if (
        min(*size, *window, *stride, *out) <= 0
        or min(pad) < 0
        or dilation != (1, 1, 1)
        or h <= 1
        or ow > 64
        or kd > 8
        or kw > 8
        or math.prod(window) > 125
        or rows * math.prod(size) >= 2**31
        or rows * math.prod(out) >= 2**31
    ):
        return None
    # Every real window must contain at least one real tap on every axis.
    # This makes max(origin, 0) exactly the all-minus-infinity fallback index.
    if any(
        -p + k <= 0 or (o - 1) * s - p >= n
        for n, k, s, p, o in zip(size, window, stride, pad, out, strict=True)
    ):
        return None
    item = 4 if dtype == "float32" else 2

    def memory(height, outputs):
        plane_stride = (height * w + 127) // 128 * 128
        ub = kd * plane_stride * (item + 4)
        ub += outputs * 64 * (item + (4 if indices else 0)) + 1024
        return plane_stride, ub

    plane_stride, ub = memory(h, oh)
    if ub <= 160 * 1024:
        return dict(
            full_plane=True,
            stripe_outputs=oh,
            stage_height=h,
            plane_stride=plane_stride,
            estimated_ub_bytes=ub,
            stripes=1,
        )
    stripe = min(8, oh)
    while stripe:
        height = (stripe - 1) * stride[1] + kh
        plane_stride, ub = memory(height, stripe)
        if ub <= 160 * 1024:
            return dict(
                full_plane=False,
                stripe_outputs=stripe,
                stage_height=height,
                plane_stride=plane_stride,
                estimated_ub_bytes=ub,
                stripes=(oh + stripe - 1) // stripe,
            )
        stripe //= 2
    return None


def supports(rows, size, window, stride, pad, dilation, out, dtype, indices):
    return plan(rows, size, window, stride, pad, dilation, out, dtype, indices) is not None


def program(rows, size, window, stride, pad, dilation, out, dtype, indices):
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    cfg = plan(rows, size, window, stride, pad, dilation, out, dtype, indices)
    if cfg is None:
        raise ValueError("Unsupported experimental spatial maximum geometry")
    d, h, w = size
    kd, kh, kw = window
    sd, sh, sw = stride
    pd, ph, pw = pad
    od, oh, ow = out
    sy = cfg["stripe_outputs"]
    stage_h = cfg["stage_height"]
    plane_stride = cfg["plane_stride"]
    stripes = cfg["stripes"]
    whole = cfg["full_plane"]
    jobs = rows * od * stripes
    cores = min(64, jobs)
    elements = kd * plane_stride
    native_lanes = 64 if dtype == "float32" else 128
    pick_rows = sy if indices else 1
    index_size = rows * od * oh * ow if indices else 1

    @T.prim_func
    def main(
        X: T.Tensor((rows * d * h * w,), dtype),
        Y: T.Tensor((rows * od, oh, ow), dtype),
        I: T.Tensor((index_size,), "int64"),
    ):
        with T.Kernel(cores) as core:
            raw = T.alloc_shared((elements,), dtype)
            data = T.alloc_shared((elements,), "float32")
            output = T.alloc_shared((sy, 64), dtype)
            picks = T.alloc_shared((pick_rows, 64), "int32")
            # Initialize once: conversion never reads uninitialized UB padding.
            # Later stale invalid rows are never consumed by a valid tap.
            with T.SimdVF():
                native_mask = S.pset(32 if dtype == "float32" else 16)
                native_zero = S.vdup(T.cast(0, dtype), dtype, native_mask)
                for part in T.serial(elements // native_lanes):
                    if dtype == "float32":
                        S.vsts(raw[part * native_lanes], native_zero, native_mask)
                    else:
                        S.vsts(raw[part * native_lanes], native_zero, native_mask, "NORM_B16")
            for step in T.serial((jobs + cores - 1) // cores):
                job = step * cores + core
                if job < jobs:
                    plane_job = job // stripes
                    ybase = job % stripes * sy
                    zstart = plane_job % od * sd - pd
                    row = plane_job // od
                    ystart = 0 if whole else ybase * sh - ph
                    for dz in T.serial(kd):
                        iz = zstart + dz
                        if iz >= 0 and iz < d:
                            if whole:
                                begin = (row * d + iz) * h * w
                                T.copy(
                                    X[begin : begin + h * w],
                                    raw[dz * plane_stride : dz * plane_stride + h * w],
                                )
                            else:
                                if ystart >= 0 and ystart + stage_h <= h:
                                    begin = ((row * d + iz) * h + ystart) * w
                                    T.copy(
                                        X[begin : begin + stage_h * w],
                                        raw[dz * plane_stride : dz * plane_stride + stage_h * w],
                                    )
                                else:
                                    for iy_local in T.serial(stage_h):
                                        iy = ystart + iy_local
                                        if iy >= 0 and iy < h:
                                            begin = ((row * d + iz) * h + iy) * w
                                            dest = dz * plane_stride + iy_local * w
                                            T.copy(X[begin : begin + w], raw[dest : dest + w])
                    with T.SimdVF():
                        full = S.pset(32)
                        for part in T.serial(elements // 64):
                            if dtype == "float32":
                                converted = S.vld(raw[part * 64])
                            else:
                                converted = S.vcvt(
                                    S.vld(raw[part * 64], "UNPK_B16"), "float32", full, part=0
                                )
                            S.vsts(data[part * 64], converted, full)
                        S.mem_bar("VST_VLD")
                        lane = S.vci(0, "int32")
                        izero = S.vdup(T.int32(0), "int32", full)
                        left = S.vadds(S.vmuls(lane, T.int32(sw), full), T.int32(-pw), full)
                        out_mask = S.vcmps(lane, T.int32(ow), full, "lt")
                        offsets = S.alloc_local((kw,), "uint32")
                        xpos = S.alloc_local((kw,), "int32")
                        valid = S.alloc_local((kw,), "bool")
                        for kx in T.Unroll(kw, explicit=True):
                            xpos[kx] = S.vadds(left, T.int32(kx), full)
                            valid[kx] = S.pand(
                                out_mask,
                                S.pand(
                                    S.vcmps(xpos[kx], T.int32(0), full, "ge"),
                                    S.vcmps(xpos[kx], T.int32(w), full, "lt"),
                                    full,
                                ),
                                full,
                            )
                            offsets[kx] = T.reinterpret(
                                S.vsel(xpos[kx], izero, valid[kx]), "uint32x64"
                            )
                        best = S.alloc_local((1,), "float32")
                        pick = S.alloc_local((1,), "int32")
                        for oy_local in T.serial(sy):
                            oy = ybase + oy_local
                            if oy < oh:
                                top = oy * sh - ph
                                best[0] = S.vdup(T.float32(float("-inf")), "float32", full)
                                if indices:
                                    first_base = (T.max(zstart, 0) * h + T.max(top, 0)) * w
                                    pick[0] = S.vadds(
                                        S.vmaxs(left, T.int32(0), full),
                                        T.cast(first_base, "int32"),
                                        full,
                                    )
                                for dz in T.serial(kd):
                                    iz = zstart + dz
                                    if iz >= 0 and iz < d:
                                        for ky in T.serial(kh):
                                            iy = top + ky
                                            if iy >= 0 and iy < h:
                                                ub_row = dz * plane_stride + (iy - ystart) * w
                                                for kx in T.Unroll(kw, explicit=True):
                                                    incoming = S.vgather2(
                                                        data[ub_row], offsets[kx], full
                                                    )
                                                    take = S.pand(
                                                        valid[kx],
                                                        S.por(
                                                            S.vcmp(incoming, best[0], full, "gt"),
                                                            S.vcmp(incoming, incoming, full, "ne"),
                                                            full,
                                                        ),
                                                        full,
                                                    )
                                                    best[0] = S.vsel(incoming, best[0], take)
                                                    if indices:
                                                        spatial_index = S.vadds(
                                                            xpos[kx],
                                                            T.cast((iz * h + iy) * w, "int32"),
                                                            full,
                                                        )
                                                        pick[0] = S.vsel(
                                                            spatial_index, pick[0], take
                                                        )
                                if dtype == "float32":
                                    S.vsts(output[oy_local, 0], best[0], out_mask)
                                else:
                                    narrowed = S.vcvt(best[0], dtype, full, sat=False)
                                    S.vsts(output[oy_local, 0], narrowed, out_mask, "PK_B32")
                                if indices:
                                    S.vsts(picks[oy_local, 0], pick[0], out_mask)
                    if whole:
                        T.copy(output[:, :ow], Y[plane_job, :, :])
                    else:
                        for oy_local in T.serial(sy):
                            if ybase + oy_local < oh:
                                T.copy(output[oy_local, :ow], Y[plane_job, ybase + oy_local, :])
                    if indices:
                        with T.SimtVF(threads=128):
                            for oy_local, ox in T.Parallel(sy, ow):
                                if ybase + oy_local < oh:
                                    I[(plane_job * oh + ybase + oy_local) * ow + ox] = T.cast(
                                        picks[oy_local, ox], "int64"
                                    )

    return main


@functools.lru_cache(maxsize=128)
def build(rows, size, window, stride, pad, dilation, out, dtype, indices):
    import tilelang

    prim = program(rows, size, window, stride, pad, dilation, out, dtype, indices)
    return tilelang.compile(
        prim.with_attr("tilelang_callee_allocated_outputs", 0), target="ascend", out_idx=[1, 2]
    )
