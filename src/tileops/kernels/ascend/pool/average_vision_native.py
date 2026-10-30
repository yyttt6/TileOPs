"""Vision5 only: native FP16 gathers, four rows, ordered FP32 accumulators.

No auto registration. program() is CPU-lowerable; build() is parent/device-only.
"""

import functools

CASES = {
    "vision5": dict(
        n=2,
        c=128,
        size=(56, 56),
        out=(28, 28),
        kernel=(5, 5),
        stride=(2, 2),
        pad=(2, 2),
        ceil=False,
        include=True,
        override=None,
        dtype="float16",
    ),
}


def call_contract(call):
    c = CASES["vision5"]
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
        return "vision5"
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


def program(name="vision5"):
    assert name == "vision5"
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    @T.prim_func
    def main(X: T.Tensor((802816,), "float16"), Y: T.Tensor((256, 14, 56), "float16")):
        with T.Kernel(64) as core:
            raw = T.alloc_shared((3136,), "float16")
            output = T.alloc_shared((14, 64), "float16")
            T.annotate_buffer_versions({raw: 2, output: 2})
            for step in T.Pipelined(4, num_stages=2):
                plane = step * 64 + core
                T.copy(X[plane * 3136 : (plane + 1) * 3136], raw)
                with T.SimdVF():
                    full16 = S.pset(16)
                    full = S.pset(32)
                    zero16 = S.vdup(T.float16(0), "float16", full16)
                    zero = S.vdup(T.float32(0), "float32", full)
                    izero = S.vdup(T.int16(0), "int16", full16)
                    lane16 = S.vci(0, "int16")
                    local_row = S.vshrs(lane16, T.int16(5), full16)
                    col = S.vsub(lane16, S.vmuls(local_row, T.int16(32), full16), full16)
                    dy = S.vmuls(local_row, T.int16(2), full16)
                    x0 = S.vadds(S.vmuls(col, T.int16(2), full16), T.int16(-2), full16)
                    x1 = S.vadds(x0, T.int16(1), full16)
                    mx0 = S.pand(
                        S.vcmps(x0, T.int16(0), full16, "ge"),
                        S.vcmps(x0, T.int16(56), full16, "lt"),
                        full16,
                    )
                    mx1 = S.pand(
                        S.vcmps(x1, T.int16(0), full16, "ge"),
                        S.vcmps(x1, T.int16(56), full16, "lt"),
                        full16,
                    )
                    offset0 = S.vsel(
                        S.vadd(S.vmuls(dy, T.int16(56), full16), x0, full16), izero, mx0
                    )
                    offset1 = S.vsel(
                        S.vadd(S.vmuls(dy, T.int16(56), full16), x1, full16), izero, mx1
                    )
                    lane = S.vci(0, "int32")
                    col32 = S.vsub(
                        lane, S.vmuls(S.vshrs(lane, T.int32(4), full), T.int32(16), full), full
                    )
                    # Clamp *each* 16-lane row. Active lanes 0..13 use halo14.
                    next_lane = S.vsel(
                        lane,
                        S.vadds(lane, T.int32(1), full),
                        S.vcmps(col32, T.int32(15), full, "eq"),
                    )
                    pack_index = S.vmins(
                        S.vsel(
                            S.vadds(lane, T.int32(4), full),
                            lane,
                            S.vcmps(lane, T.int32(28), full, "ge"),
                        ),
                        T.int32(63),
                        full,
                    )
                    packed_active = S.vcmps(lane, T.int32(56), full, "lt")
                    samples = S.alloc_local((2,), "float16")
                    acc = S.alloc_local((2,), "float32")
                    for group in T.serial(7):
                        acc[0] = zero
                        acc[1] = zero
                        for ky in T.serial(5):
                            if group > 0 and group < 6:
                                # Fixed aligned raw[0] base, including odd ky.
                                p0 = S.vadds(
                                    offset0, T.cast((group * 8 - 2 + ky) * 56, "int16"), full16
                                )
                                p1 = S.vadds(
                                    offset1, T.cast((group * 8 - 2 + ky) * 56, "int16"), full16
                                )
                                samples[0] = S.vsel(
                                    S.vgather2(raw[0], T.reinterpret(p0, "uint16x128"), full16),
                                    zero16,
                                    mx0,
                                )
                                samples[1] = S.vsel(
                                    S.vgather2(raw[0], T.reinterpret(p1, "uint16x128"), full16),
                                    zero16,
                                    mx1,
                                )
                            else:
                                iy = S.vadds(dy, T.cast(group * 8 - 2 + ky, "int16"), full16)
                                my = S.pand(
                                    S.vcmps(iy, T.int16(0), full16, "ge"),
                                    S.vcmps(iy, T.int16(56), full16, "lt"),
                                    full16,
                                )
                                m0 = S.pand(mx0, my, full16)
                                m1 = S.pand(mx1, my, full16)
                                base = S.vmuls(iy, T.int16(56), full16)
                                p0 = S.vsel(S.vadd(base, x0, full16), izero, m0)
                                p1 = S.vsel(S.vadd(base, x1, full16), izero, m1)
                                samples[0] = S.vsel(
                                    S.vgather2(raw[0], T.reinterpret(p0, "uint16x128"), full16),
                                    zero16,
                                    m0,
                                )
                                samples[1] = S.vsel(
                                    S.vgather2(raw[0], T.reinterpret(p1, "uint16x128"), full16),
                                    zero16,
                                    m1,
                                )
                            e0 = S.vcvt(samples[0], "float32", full16, part=0)
                            o0 = S.vcvt(samples[0], "float32", full16, part=1)
                            e1 = S.vcvt(samples[1], "float32", full16, part=0)
                            o1 = S.vcvt(samples[1], "float32", full16, part=1)
                            # Even output: kx0,1,2,3,4; then next ky, no reassociation.
                            acc[0] = S.vadd(acc[0], e0, full)
                            acc[0] = S.vadd(acc[0], e1, full)
                            acc[0] = S.vadd(acc[0], o0, full)
                            acc[0] = S.vadd(acc[0], o1, full)
                            next_e0 = S.vselr(e0, next_lane)
                            acc[0] = S.vadd(acc[0], next_e0, full)
                            # Odd output: kx0,1,2,3,4 in the identical order.
                            acc[1] = S.vadd(acc[1], o0, full)
                            acc[1] = S.vadd(acc[1], o1, full)
                            acc[1] = S.vadd(acc[1], next_e0, full)
                            acc[1] = S.vadd(acc[1], S.vselr(e1, next_lane), full)
                            acc[1] = S.vadd(acc[1], S.vselr(o0, next_lane), full)
                        avg_even = S.vmuls(acc[0], T.float32(1.0 / 25), full)
                        avg_odd = S.vmuls(acc[1], T.float32(1.0 / 25), full)
                        lo, hi = S.vintlv(avg_even, avg_odd)
                        packed_lo = S.vselr(lo, pack_index)
                        packed_hi = S.vselr(hi, pack_index)
                        S.vsts(
                            output[group * 2, 0],
                            S.vcvt(packed_lo, "float16", sat=False),
                            packed_active,
                            "PK_B32",
                        )
                        S.vsts(
                            output[group * 2 + 1, 0],
                            S.vcvt(packed_hi, "float16", sat=False),
                            packed_active,
                            "PK_B32",
                        )
                T.copy(output[:, :56], Y[plane, :, :])

    return main.with_attr("tilelang_callee_allocated_outputs", 0)


@functools.lru_cache(maxsize=1)
def build(name="vision5"):
    import tilelang

    return tilelang.compile(program(name), target="ascend", out_idx=-1)
