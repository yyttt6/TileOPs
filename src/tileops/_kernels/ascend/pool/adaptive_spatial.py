"""Bounded plane-staged adaptive pooling with FP32 SIMD bin accumulation.

One output bin per lane; each plane is copied and widened once. Configuration
excludes global, identity and unsupported geometries so the caller retains its
existing fallback. This module performs no registration on import.
"""

import math


def configuration(planes, h, w, oh, ow, dtype):
    """Pure-host support gate and explicit UB budget, excluding compiler scratch."""
    if min(planes, h, w, oh, ow) <= 0:
        return {"path": "production-fallback"}
    if (
        dtype not in ("float16", "bfloat16")
        or not 1 < oh * ow <= 64
        or (oh, ow) == (h, w)
        or h * w > 4096
        or planes * max(h * w, oh * ow) >= 2**31
    ):
        return {"path": "production-fallback"}
    kh = (h + oh - math.gcd(h, oh) - 1) // oh + 1
    kw = (w + ow - math.gcd(w, ow) - 1) // ow + 1
    if kh * kw > 128:
        return {"path": "production-fallback"}
    stride = ((h * w + 63) // 64) * 64
    return {
        "path": "plane-simd-bins",
        "plane_stride": stride,
        "max_bin_height": kh,
        "max_bin_width": kw,
        "output_lanes": oh * ow,
        "stages": 1,
        "explicit_ub_bytes": stride * 6 + 3 * 64 * 4 + 64 * (2 + 4),
    }


def spatial_program(planes, h, w, oh, ow, dtype, maximum, with_indices):
    """Build the bounded plane-staged adaptive pooling PrimFunc."""
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    cfg = configuration(planes, h, w, oh, ow, dtype)
    if cfg["path"] != "plane-simd-bins":
        raise ValueError("unsupported plane-staged adaptive geometry; caller must select fallback")
    if with_indices and not maximum:
        raise ValueError("indices require max pooling")
    spatial, outputs = h * w, oh * ow
    stride = cfg["plane_stride"]
    kh, kw = cfg["max_bin_height"], cfg["max_bin_width"]
    blocks = min(64, planes)

    @T.prim_func
    def main(
        X: T.Tensor((planes, spatial), dtype),
        Y: T.Tensor((planes * outputs,), dtype),
        Indices: T.Tensor((planes * outputs if with_indices else 1,), "int64"),
    ):
        with T.Kernel(blocks) as core:
            raw = T.alloc_shared((stride,), dtype)
            data = T.alloc_shared((stride,), "float32")
            metadata = T.alloc_shared((3, 64), "int32")
            result = T.alloc_shared((64,), dtype)
            if with_indices:
                picks = T.alloc_shared((64,), "int32")
            # Immutable integer metadata: one short VF per core, not per plane
            # or per tap. No integer->float casts in this SIMT region.
            with T.SimtVF(threads=64):
                for lane in T.Parallel(64):
                    metadata[0, lane] = 0
                    metadata[1, lane] = 1
                    metadata[2, lane] = 1
                    if lane < outputs:
                        ph = lane // ow
                        pw = lane % ow
                        hs = ph * h // oh
                        ws = pw * w // ow
                        he = ((ph + 1) * h + oh - 1) // oh
                        we = ((pw + 1) * w + ow - 1) // ow
                        metadata[0, lane] = hs * w + ws
                        metadata[1, lane] = he - hs
                        metadata[2, lane] = we - ws
            # raw: MTE2 -> SIMD; data: SIMD write -> same-VF UB gather;
            # result: SIMD -> MTE3; picks: SIMD -> SIMT. Serial jobs avoid
            # inter-job overlap. Parent must inspect emitted dependency events.
            for step in T.serial((planes + blocks - 1) // blocks):
                plane = step * blocks + core
                if plane < planes:
                    T.copy(X[plane, :], raw[:spatial])
                    with T.SimdVF():
                        full = S.pset(32)
                        active = S.update_mask(outputs, 32)
                        zero = S.vdup(T.float32(0), "float32", full)
                        izero = S.vdup(T.int32(0), "int32", full)
                        for part in T.serial(stride // 64):
                            valid = S.update_mask(T.min(64, spatial - part * 64), 32)
                            widened = S.vcvt(
                                S.vld(raw[part * 64], dist="UNPK_B16"),
                                "float32",
                                valid,
                                part=0,
                            )
                            S.vsts(data[part * 64], widened, full)
                        S.mem_bar("VST_VLD")
                        base = S.vld(metadata[0, 0])
                        height = S.vld(metadata[1, 0])
                        width = S.vld(metadata[2, 0])
                        acc = S.alloc_local((1,), "float32")
                        if maximum:
                            # Initializing from the first actual tap preserves
                            # all--inf, ties and signed-zero payload selection.
                            acc[0] = S.vgather2(data[0], T.reinterpret(base, "uint32x64"), full)
                        else:
                            acc[0] = zero
                        if with_indices:
                            pick = S.alloc_local((1,), "int32")
                            pick[0] = base
                        for dy in T.serial(kh):
                            row_active = S.pand(
                                active, S.vcmps(height, T.cast(dy, "int32"), full, "gt"), full
                            )
                            for dx in T.Unroll(kw, explicit=True):
                                tap_active = S.pand(
                                    row_active,
                                    S.vcmps(width, T.int32(dx), full, "gt"),
                                    full,
                                )
                                location = S.vadds(base, T.cast(dy * w + dx, "int32"), full)
                                # Clamp inactive addresses BEFORE the gather.
                                offset = T.reinterpret(
                                    S.vsel(location, izero, tap_active), "uint32x64"
                                )
                                incoming = S.vgather2(data[0], offset, full)
                                if maximum:
                                    take = S.pand(
                                        tap_active,
                                        S.por(
                                            S.vcmp(incoming, acc[0], full, "gt"),
                                            S.vcmp(incoming, incoming, full, "ne"),
                                            full,
                                        ),
                                        full,
                                    )
                                    acc[0] = S.vsel(incoming, acc[0], take)
                                    if with_indices:
                                        pick[0] = S.vsel(location, pick[0], take)
                                else:
                                    acc[0] = S.vadd(
                                        acc[0], S.vsel(incoming, zero, tap_active), full
                                    )
                        if maximum:
                            narrowed = S.vcvt(acc[0], dtype, full, sat=False)
                        else:
                            counts = S.vcvt(S.vmul(height, width, full), "float32", full)
                            narrowed = S.vcvt(S.vdiv(acc[0], counts, full), dtype, full, sat=False)
                        S.vsts(result[0], narrowed, active, "PK_B32")
                        if with_indices:
                            S.vsts(picks[0], pick[0], active)
                    T.copy(result[:outputs], Y[plane * outputs : (plane + 1) * outputs])
                    if with_indices:
                        with T.SimtVF(threads=64):
                            for lane in T.Parallel(outputs):
                                Indices[plane * outputs + lane] = T.cast(picks[lane], "int64")

    return main.with_attr("tilelang_callee_allocated_outputs", 0)
