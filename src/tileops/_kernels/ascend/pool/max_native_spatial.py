"""Unregistered experimental native FP16/BF16 spatial maximum pooling.

Full-plane DMA plus 128-lane native gathers/comparisons: no FP32 staging or
value conversion. Index vectors carry int16 tap ordinals, expanded to flat
spatial int64 only in the final SIMT store. Unit dilation, nonempty windows.
Unsupported geometries/dtypes must use the frozen spatial or generic builder.
"""

import functools
import math


def plan(rows, size, window, stride, pad, dilation, out, dtype, indices):
    if (
        dtype not in ("float16", "bfloat16")
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
    if any(
        -p + k <= 0 or (o - 1) * s - p >= n
        for n, k, s, p, o in zip(size, window, stride, pad, out, strict=True)
    ):
        return None
    # All 128 lanes execute signed int16 arithmetic BEFORE the output mask.
    # Bound intermediates as well as valid offsets to prevent tail-lane wrap.
    if (
        w > 32767
        or stride[2] > 32767
        or pad[2] > 32767
        or 127 * stride[2] > 32767
        or 127 * stride[2] - pad[2] + kw - 1 > 32767
    ):
        return None
    plane_stride = (h * w + 127) // 128 * 128 + 128
    ub = kd * plane_stride * 2 + oh * 128 * (2 + (2 if indices else 0)) + 1024
    if ub > 160 * 1024:
        return None
    return dict(
        full_plane=True,
        native_lanes=128,
        valid_columns=ow,
        plane_stride=plane_stride,
        estimated_ub_bytes=ub,
        fp32_staging_bytes=0,
        tap_dtype="int16",
        jobs=rows * od,
    )


def supports(rows, size, window, stride, pad, dilation, out, dtype, indices):
    return plan(rows, size, window, stride, pad, dilation, out, dtype, indices) is not None


def program(rows, size, window, stride, pad, dilation, out, dtype, indices):
    cfg = plan(rows, size, window, stride, pad, dilation, out, dtype, indices)
    if cfg is None:
        raise ValueError("Unsupported native-16 spatial maximum geometry")
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    d, h, w = size
    kd, kh, kw = window
    sd, sh, sw = stride
    pd, ph, pw = pad
    od, oh, ow = out
    plane_stride = cfg["plane_stride"]
    jobs = rows * od
    cores = min(64, jobs)
    pick_rows = oh if indices else 1
    index_size = jobs * oh * ow if indices else 1

    @T.prim_func
    def main(
        X: T.Tensor((rows * d * h * w,), dtype),
        Y: T.Tensor((jobs, oh, ow), dtype),
        I: T.Tensor((index_size,), "int64"),
    ):
        with T.Kernel(cores) as core:
            raw = T.alloc_shared((kd * plane_stride,), dtype)
            output = T.alloc_shared((oh, 128), dtype)
            picks = T.alloc_shared((pick_rows, 128), "int16")
            for step in T.serial((jobs + cores - 1) // cores):
                job = step * cores + core
                if job < jobs:
                    row = job // od
                    zstart = job % od * sd - pd
                    for dz in T.serial(kd):
                        iz = zstart + dz
                        if iz >= 0 and iz < d:
                            begin = (row * d + iz) * h * w
                            T.copy(
                                X[begin : begin + h * w],
                                raw[dz * plane_stride : dz * plane_stride + h * w],
                            )
                    with T.SimdVF():
                        full = S.pset(16)
                        lane = S.vci(0, "int16")
                        izero = S.vdup(T.int16(0), "int16", full)
                        left = S.vadds(S.vmuls(lane, T.int16(sw), full), T.int16(-pw), full)
                        out_mask = S.vcmps(lane, T.int16(ow), full, "lt")
                        offsets = S.alloc_local((kw,), "uint16")
                        valid = S.alloc_local((kw,), "bool")
                        for kx in T.Unroll(kw, explicit=True):
                            xpos = S.vadds(left, T.int16(kx), full)
                            valid[kx] = S.pand(
                                out_mask,
                                S.pand(
                                    S.vcmps(xpos, T.int16(0), full, "ge"),
                                    S.vcmps(xpos, T.int16(w), full, "lt"),
                                    full,
                                ),
                                full,
                            )
                            offsets[kx] = T.reinterpret(
                                S.vsel(xpos, izero, valid[kx]), "uint16x128"
                            )
                        best = S.alloc_local((1,), dtype)
                        pick = S.alloc_local((1,), "int16")
                        for oy in T.serial(oh):
                            top = oy * sh - ph
                            best[0] = S.vdup(T.cast(float("-inf"), dtype), dtype, full)
                            if indices:
                                pick[0] = S.vdup(T.int16(-1), "int16", full)
                            for dz in T.serial(kd):
                                iz = zstart + dz
                                if iz >= 0 and iz < d:
                                    for ky in T.serial(kh):
                                        iy = top + ky
                                        if iy >= 0 and iy < h:
                                            ub_row = dz * plane_stride + iy * w
                                            for kx in T.Unroll(kw, explicit=True):
                                                # Safe offset zero still names a real element
                                                # of this valid z/y row for every masked lane.
                                                incoming = S.vgather2(
                                                    raw[ub_row], offsets[kx], full
                                                )
                                                better = S.por(
                                                    S.vcmp(incoming, best[0], full, "gt"),
                                                    S.vcmp(incoming, incoming, full, "ne"),
                                                    full,
                                                )
                                                if indices:
                                                    take = S.pand(
                                                        valid[kx],
                                                        S.por(
                                                            better,
                                                            S.vcmps(
                                                                pick[0], T.int16(0), full, "lt"
                                                            ),
                                                            full,
                                                        ),
                                                        full,
                                                    )
                                                else:
                                                    take = S.pand(valid[kx], better, full)
                                                best[0] = S.vsel(incoming, best[0], take)
                                                if indices:
                                                    ordinal = S.vdup(
                                                        T.cast((dz * kh + ky) * kw + kx, "int16"),
                                                        "int16",
                                                        full,
                                                    )
                                                    pick[0] = S.vsel(ordinal, pick[0], take)
                            S.vsts(output[oy, 0], best[0], out_mask, "NORM_B16")
                            if indices:
                                S.vsts(picks[oy, 0], pick[0], out_mask, "NORM_B16")
                    T.copy(output[:, :ow], Y[job, :, :])
                    if indices:
                        with T.SimtVF(threads=128):
                            for oy, ox in T.Parallel(oh, ow):
                                tap = T.cast(picks[oy, ox], "int32")
                                iz = zstart + tap // (kh * kw)
                                iy = oy * sh - ph + tap // kw % kh
                                ix = ox * sw - pw + tap % kw
                                I[(job * oh + oy) * ow + ox] = (
                                    T.cast(iz, "int64") * h + iy
                                ) * w + ix

    return main


@functools.lru_cache(maxsize=128)
def build(rows, size, window, stride, pad, dilation, out, dtype, indices):
    import tilelang

    prim = program(rows, size, window, stride, pad, dilation, out, dtype, indices)
    return tilelang.compile(
        prim.with_attr("tilelang_callee_allocated_outputs", 0), target="ascend", out_idx=[1, 2]
    )
