"""Exact ceil-video FP16 average pooling with four output rows per SIMD group.

The constructor must also check the exact public call (N=2,C=64), rather than
using the flattened window gate alone. No registration or mutation on import.
"""

import functools


def supports(window, dtype):
    return (
        dtype == "float16"
        and window.rows == 128
        and window.size == (8, 28, 28)
        and window.out == (5, 15, 15)
        and window.kernel == (2, 3, 3)
        and window.stride == (2, 2, 2)
        and window.pad == (1, 1, 1)
        and window.ceil_mode
        and not window.count_include_pad
        and window.divisor_override is None
    )


def program():
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    # 784 valid elements per input plane, rounded to a full 64-float vector.
    # Raw/output are double buffered; FP32 data is live within one SIMD stage.
    plane, pitch = 784, 832

    @T.prim_func
    def main(X: T.Tensor((802816,), "float16"), Y: T.Tensor((640, 225), "float16")):
        with T.Kernel(64) as core:
            raw = T.alloc_shared((2 * pitch,), "float16")
            data = T.alloc_shared((2 * pitch,), "float32")
            output = T.alloc_shared((4, 64), "float16")
            T.annotate_buffer_versions({raw: 2, data: 1, output: 2})
            for step in T.Pipelined(10, num_stages=2):
                job = step * 64 + core
                row = job // 5
                zstart = job % 5 * 2 - 1
                count_z = T.min(zstart + 2, 8) - T.max(zstart, 0)
                for kz in T.serial(2):
                    iz = zstart + kz
                    if iz >= 0 and iz < 8:
                        begin = (row * 8 + iz) * plane
                        T.copy(X[begin : begin + plane], raw[kz * pitch : kz * pitch + plane])
                with T.SimdVF():
                    full = S.pset(32)
                    zero = S.vdup(T.float32(0), "float32", full)
                    izero = S.vdup(T.int32(0), "int32", full)
                    ione = S.vdup(T.int32(1), "int32", full)
                    lane = S.vci(0, "int32")
                    row15 = S.vsel(ione, izero, S.vcmps(lane, T.int32(15), full, "ge"))
                    row30 = S.vsel(ione, izero, S.vcmps(lane, T.int32(30), full, "ge"))
                    row45 = S.vsel(ione, izero, S.vcmps(lane, T.int32(45), full, "ge"))
                    row_local = S.vadd(S.vadd(row15, row30, full), row45, full)
                    ox = S.vsub(lane, S.vmuls(row_local, T.int32(15), full), full)
                    xstart = S.vadds(S.vmuls(ox, T.int32(2), full), T.int32(-1), full)
                    active60 = S.vcmps(lane, T.int32(60), full, "lt")
                    count_x = S.vmaxs(
                        S.vsub(
                            S.vmins(S.vadds(xstart, T.int32(3), full), T.int32(28), full),
                            S.vmaxs(xstart, T.int32(0), full),
                            full,
                        ),
                        T.int32(0),
                        full,
                    )
                    xpos = S.alloc_local((3,), "int32")
                    xmask = S.alloc_local((3,), "bool")
                    offsets = S.alloc_local((9,), "uint32")
                    masks = S.alloc_local((9,), "bool")
                    acc = S.alloc_local((1,), "float32")
                    for kx in T.Unroll(3, explicit=True):
                        xpos[kx] = S.vadds(xstart, T.int32(kx), full)
                        xmask[kx] = S.pand(
                            S.vcmps(xpos[kx], T.int32(0), full, "ge"),
                            S.vcmps(xpos[kx], T.int32(28), full, "lt"),
                            full,
                        )
                    for kz in T.serial(2):
                        if zstart + kz >= 0 and zstart + kz < 8:
                            for v in T.serial(13):
                                wide = S.vcvt(
                                    S.vld(raw[kz * pitch + v * 64], dist="UNPK_B16"),
                                    "float32",
                                    full,
                                    part=0,
                                )
                                S.vsts(data[kz * pitch + v * 64], wide, full)
                    S.mem_bar("VST_VLD")
                    for group in T.serial(4):
                        oy = S.vadds(row_local, T.cast(group * 4, "int32"), full)
                        active = S.pand(active60, S.vcmps(oy, T.int32(15), full, "lt"), full)
                        ystart = S.vadds(S.vmuls(oy, T.int32(2), full), T.int32(-1), full)
                        count_y = S.vmaxs(
                            S.vsub(
                                S.vmins(S.vadds(ystart, T.int32(3), full), T.int32(28), full),
                                S.vmaxs(ystart, T.int32(0), full),
                                full,
                            ),
                            T.int32(0),
                            full,
                        )
                        # Each safe offset addresses the aligned plane base, never
                        # an unaligned 28-float row base. Reuse these across kz.
                        for ky in T.Unroll(3, explicit=True):
                            iy = S.vadds(ystart, T.int32(ky), full)
                            ymask = S.pand(
                                S.vcmps(iy, T.int32(0), full, "ge"),
                                S.vcmps(iy, T.int32(28), full, "lt"),
                                full,
                            )
                            for kx in T.Unroll(3, explicit=True):
                                masks[ky * 3 + kx] = S.pand(
                                    active, S.pand(ymask, xmask[kx], full), full
                                )
                                address = S.vadd(S.vmuls(iy, T.int32(28), full), xpos[kx], full)
                                offsets[ky * 3 + kx] = T.reinterpret(
                                    S.vsel(address, izero, masks[ky * 3 + kx]), "uint32x64"
                                )
                        acc[0] = zero
                        for kz in T.serial(2):
                            # No read from an unstaged depth plane, including its
                            # safe-offset zero. Zero padding contributes no NaNs.
                            if zstart + kz >= 0 and zstart + kz < 8:
                                for ky in T.Unroll(3, explicit=True):
                                    for kx in T.Unroll(3, explicit=True):
                                        gathered = S.vgather2(
                                            data[kz * pitch], offsets[ky * 3 + kx], full
                                        )
                                        value = S.vsel(gathered, zero, masks[ky * 3 + kx])
                                        acc[0] = S.vadd(acc[0], value, full)
                        counts = S.vmaxs(
                            S.vmuls(S.vmul(count_x, count_y, full), T.cast(count_z, "int32"), full),
                            T.int32(1),
                            full,
                        )
                        average = S.vdiv(acc[0], S.vcvt(counts, "float32", full), full)
                        narrowed = S.vcvt(average, "float16", sat=False)
                        S.vsts(output[group, 0], narrowed, active, "PK_B32")
                for group in T.serial(3):
                    T.copy(output[group, :60], Y[job, group * 60 : group * 60 + 60])
                T.copy(output[3, :45], Y[job, 180:225])

    return main.with_attr("tilelang_callee_allocated_outputs", 0)


@functools.lru_cache(maxsize=1)
def build(window, dtype):
    import tilelang

    assert supports(window, dtype)
    return tilelang.compile(program(), target="ascend", out_idx=-1)


def call_contract(call):
    """Only the validated public ceil-video FP16 call, including separate N/C."""
    return (
        call.n == 2
        and call.c_in == 64
        and tuple(call.size) == (8, 28, 28)
        and tuple(call.window) == (2, 3, 3)
        and tuple(call.stride) == (2, 2, 2)
        and tuple(call.pad) == (1, 1, 1)
        and call.ceil_mode
        and not call.count_include_pad
        and call.divisor_override is None
        and str(call.dtype).split(".")[-1] == "float16"
    )
