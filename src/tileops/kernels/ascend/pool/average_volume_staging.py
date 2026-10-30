"""Avg05 exact ceil-video: one contiguous input volume per channel job.

TileLang Ascend only. Original Avg02 gather/add/divide/cast/output layout retained.
program() is CPU-lowerable; build() is parent/device-only. No auto registration.
"""

import functools

VARIANT = "avg05-volume-staging"
CASES = {
    "ceilvideo": dict(
        n=2,
        c=64,
        size=(8, 28, 28),
        out=(5, 15, 15),
        kernel=(2, 3, 3),
        stride=(2, 2, 2),
        pad=(1, 1, 1),
        ceil=True,
        include=False,
        override=None,
        dtype="float16",
    ),
}


def call_contract(call):
    for name, c in CASES.items():
        if (
            call.n == c["n"]
            and call.c_in == c["c"]
            and tuple(call.size) == c["size"]
            and tuple(call.window) == c["kernel"]
            and tuple(call.stride) == c["stride"]
            and tuple(call.pad) == c["pad"]
            and call.ceil_mode == c["ceil"]
            and call.count_include_pad == c["include"]
            and call.divisor_override == c["override"]
            and str(call.dtype).split(".")[-1] == c["dtype"]
        ):
            return name
    return None


def supports(window, dtype, name):
    c = CASES[name]
    return (
        window.rows == c["n"] * c["c"]
        and window.size == c["size"]
        and window.out == c["out"]
        and window.kernel == c["kernel"]
        and window.stride == c["stride"]
        and window.pad == c["pad"]
        and window.ceil_mode == c["ceil"]
        and window.count_include_pad == c["include"]
        and window.divisor_override == c["override"]
        and dtype == c["dtype"]
    )


def volume_program():
    """128 whole-channel jobs; unchanged ordered arithmetic and padded output rows."""
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    pitch = 784
    raw_size = 6272
    depths = 5
    jobs = 128

    @T.prim_func
    def main(X: T.Tensor((802816,), "float16"), Y: T.Tensor((640, 15, 15), "float16")):
        with T.Kernel(64) as core:
            raw = T.alloc_shared((raw_size,), "float16")
            data = T.alloc_shared((raw_size,), "float32")
            output = T.alloc_shared((depths, 16, 16), "float16")
            T.annotate_buffer_versions({raw: 2, data: 1, output: 2})
            for step in T.Pipelined(jobs // 64, num_stages=2):
                job = step * 64 + core
                begin = job * raw_size
                T.copy(X[begin : begin + raw_size], raw[:])
                with T.SimdVF():
                    full = S.pset(32)
                    zero = S.vdup(T.float32(0), "float32", full)
                    izero = S.vdup(T.int32(0), "int32", full)
                    # 6272/64 = 98 full vectors: no padded or uninitialized lanes.
                    for v in T.serial(98):
                        wide = S.vcvt(S.vld(raw[v * 64], "UNPK_B16"), "float32", full, part=0)
                        S.vsts(data[v * 64], wide, full)
                    S.mem_bar("VST_VLD")
                    lane = S.vci(0, "int32")
                    local_y = S.vshrs(lane, T.int32(4), full)
                    ox = S.vsub(lane, S.vmuls(local_y, T.int32(16), full), full)
                    x0 = S.vadds(S.vmuls(ox, T.int32(2), full), T.int32(-1), full)
                    x1 = S.vadds(x0, T.int32(1), full)
                    mx0 = S.pand(
                        S.vcmps(x0, T.int32(0), full, "ge"),
                        S.vcmps(x0, T.int32(28), full, "lt"),
                        full,
                    )
                    mx1 = S.pand(
                        S.vcmps(x1, T.int32(0), full, "ge"),
                        S.vcmps(x1, T.int32(28), full, "lt"),
                        full,
                    )
                    nxt = S.vmins(S.vadds(lane, T.int32(1), full), T.int32(63), full)
                    cx = S.vmaxs(
                        S.vsub(
                            S.vmins(S.vadds(x0, T.int32(3), full), T.int32(28), full),
                            S.vmaxs(x0, T.int32(0), full),
                            full,
                        ),
                        T.int32(0),
                        full,
                    )
                    offsets = S.alloc_local((6,), "uint32")
                    masks = S.alloc_local((6,), "bool")
                    acc = S.alloc_local((1,), "float32")
                    # Static depth expansion avoids Bisheng nested-VF loop CMP failure.
                    for oz in T.Unroll(depths, explicit=True):
                        zstart = oz * 2 - 1
                        cz = T.min(zstart + 2, 8) - T.max(zstart, 0)
                        for group in T.serial(4):
                            oy = S.vadds(local_y, T.cast(group * 4, "int32"), full)
                            y0 = S.vadds(S.vmuls(oy, T.int32(2), full), T.int32(-1), full)
                            cy = S.vmaxs(
                                S.vsub(
                                    S.vmins(S.vadds(y0, T.int32(3), full), T.int32(28), full),
                                    S.vmaxs(y0, T.int32(0), full),
                                    full,
                                ),
                                T.int32(0),
                                full,
                            )
                            for ky in T.Unroll(3, explicit=True):
                                iy = S.vadds(y0, T.int32(ky), full)
                                my = S.pand(
                                    S.vcmps(iy, T.int32(0), full, "ge"),
                                    S.vcmps(iy, T.int32(28), full, "lt"),
                                    full,
                                )
                                masks[ky * 2] = S.pand(mx0, my, full)
                                masks[ky * 2 + 1] = S.pand(mx1, my, full)
                                base = S.vmuls(iy, T.int32(28), full)
                                p0 = S.vsel(S.vadd(base, x0, full), izero, masks[ky * 2])
                                p1 = S.vsel(S.vadd(base, x1, full), izero, masks[ky * 2 + 1])
                                offsets[ky * 2] = T.reinterpret(p0, "uint32x64")
                                offsets[ky * 2 + 1] = T.reinterpret(p1, "uint32x64")
                            acc[0] = zero
                            for kz in T.serial(2):
                                if zstart + kz >= 0 and zstart + kz < 8:
                                    for ky in T.Unroll(3, explicit=True):
                                        origin = (zstart + kz) * pitch
                                        a = S.vsel(
                                            S.vgather2(data[origin], offsets[ky * 2], full),
                                            zero,
                                            masks[ky * 2],
                                        )
                                        b = S.vsel(
                                            S.vgather2(data[origin], offsets[ky * 2 + 1], full),
                                            zero,
                                            masks[ky * 2 + 1],
                                        )
                                        acc[0] = S.vadd(acc[0], a, full)
                                        acc[0] = S.vadd(acc[0], b, full)
                                        acc[0] = S.vadd(acc[0], S.vselr(a, nxt), full)
                            counts = S.vmaxs(
                                S.vmuls(S.vmul(cx, cy, full), T.cast(cz, "int32"), full),
                                T.int32(1),
                                full,
                            )
                            avg = S.vdiv(acc[0], S.vcvt(counts, "float32", full), full)
                            S.vsts(
                                output[oz, group * 4, 0],
                                S.vcvt(avg, "float16", sat=False),
                                full,
                                "PK_B32",
                            )
                for oz in T.serial(depths):
                    target = job * depths + oz
                    T.copy(output[oz, :15, :15], Y[target, :, :])

    return main.with_attr("tilelang_callee_allocated_outputs", 0)


def program(name):
    assert name in CASES
    return volume_program()


@functools.lru_cache(maxsize=1)
def build(name):
    import tilelang

    return tilelang.compile(program(name), target="ascend", out_idx=-1)
