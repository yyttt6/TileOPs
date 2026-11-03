"""BF16 ceil-mode pooling with divisor override and register sample reuse."""

import functools

CASES = {
    "ceildiv2d": {
        "n": 3,
        "c": 96,
        "size": (55, 57),
        "out": (28, 29),
        "kernel": (3, 5),
        "stride": (2, 2),
        "pad": (1, 2),
        "ceil": True,
        "include": False,
        "override": 7,
        "dtype": "bfloat16",
    }
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


def spatial_program(name):
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    c = CASES[name]
    h, w = c["size"]
    kh, _ = c["kernel"]
    ph, _ = c["pad"]
    ow = c["out"][1]
    rows, dtype = c["n"] * c["c"], c["dtype"]
    plane = h * w
    padded = (plane + 63) // 64 * 64
    divisor = 25 if name == "vision5" else 7

    @T.prim_func
    def main(X: T.Tensor((rows * plane,), dtype), Y: T.Tensor((rows, 28, ow), dtype)):
        with T.Kernel(64) as core:
            raw = T.alloc_shared((padded,), dtype)
            data = T.alloc_shared((padded,), "float32")
            output = T.alloc_shared((28, 32), dtype)
            T.annotate_buffer_versions({raw: 2, data: 1, output: 2})
            for step in T.Pipelined((rows + 63) // 64, num_stages=2):
                row = step * 64 + core
                if row < rows:
                    T.copy(X[row * plane : (row + 1) * plane], raw[:plane])
                    with T.SimdVF():
                        full = S.pset(32)
                        zero = S.vdup(T.float32(0), "float32", full)
                        izero = S.vdup(T.int32(0), "int32", full)
                        for v in T.serial(padded // 64):
                            wide = S.vcvt(S.vld(raw[v * 64], "UNPK_B16"), "float32", full, part=0)
                            S.vsts(data[v * 64], wide, full)
                        S.mem_bar("VST_VLD")
                        lane = S.vci(0, "int32")
                        high = S.vcmps(lane, T.int32(32), full, "ge")
                        ox = S.vsel(S.vadds(lane, T.int32(-32), full), lane, high)
                        dy = S.vsel(S.vdup(T.int32(2), "int32", full), izero, high)
                        x0 = S.vadds(S.vmuls(ox, T.int32(2), full), T.int32(-2), full)
                        x1 = S.vadds(x0, T.int32(1), full)
                        mx0 = S.pand(
                            S.vcmps(x0, T.int32(0), full, "ge"),
                            S.vcmps(x0, T.int32(w), full, "lt"),
                            full,
                        )
                        mx1 = S.pand(
                            S.vcmps(x1, T.int32(0), full, "ge"),
                            S.vcmps(x1, T.int32(w), full, "lt"),
                            full,
                        )
                        # Active columns <=28: +2 stays in the same 32-lane row.
                        # Padded lanes are discarded; their shuffle indices also stay bounded.
                        next1 = S.vmins(S.vadds(lane, T.int32(1), full), T.int32(63), full)
                        next2 = S.vmins(S.vadds(lane, T.int32(2), full), T.int32(63), full)
                        offset0 = S.vsel(
                            S.vadd(S.vmuls(dy, T.int32(w), full), x0, full), izero, mx0
                        )
                        offset1 = S.vsel(
                            S.vadd(S.vmuls(dy, T.int32(w), full), x1, full), izero, mx1
                        )
                        samples = S.alloc_local((2,), "float32")
                        acc = S.alloc_local((1,), "float32")
                        for pair in T.serial(14):
                            acc[0] = zero
                            for ky in T.serial(kh):
                                if pair > 0 and pair < 13:
                                    # Interior Y rows are guaranteed valid. Only X masks remain.
                                    if w % 8 == 0:
                                        g0 = S.vgather2(
                                            data[(pair * 4 - ph + ky) * w],
                                            T.reinterpret(offset0, "uint32x64"),
                                            full,
                                        )
                                        g1 = S.vgather2(
                                            data[(pair * 4 - ph + ky) * w],
                                            T.reinterpret(offset1, "uint32x64"),
                                            full,
                                        )
                                    else:
                                        p0 = S.vadds(
                                            offset0, T.cast((pair * 4 - ph + ky) * w, "int32"), full
                                        )
                                        p1 = S.vadds(
                                            offset1, T.cast((pair * 4 - ph + ky) * w, "int32"), full
                                        )
                                        g0 = S.vgather2(
                                            data[0], T.reinterpret(p0, "uint32x64"), full
                                        )
                                        g1 = S.vgather2(
                                            data[0], T.reinterpret(p1, "uint32x64"), full
                                        )
                                    samples[0] = S.vsel(g0, zero, mx0)
                                    samples[1] = S.vsel(g1, zero, mx1)
                                else:
                                    iy = S.vadds(dy, T.cast(pair * 4 - ph + ky, "int32"), full)
                                    my = S.pand(
                                        S.vcmps(iy, T.int32(0), full, "ge"),
                                        S.vcmps(iy, T.int32(h), full, "lt"),
                                        full,
                                    )
                                    m0 = S.pand(mx0, my, full)
                                    m1 = S.pand(mx1, my, full)
                                    base = S.vmuls(iy, T.int32(w), full)
                                    p0 = S.vsel(S.vadd(base, x0, full), izero, m0)
                                    p1 = S.vsel(S.vadd(base, x1, full), izero, m1)
                                    samples[0] = S.vsel(
                                        S.vgather2(data[0], T.reinterpret(p0, "uint32x64"), full),
                                        zero,
                                        m0,
                                    )
                                    samples[1] = S.vsel(
                                        S.vgather2(data[0], T.reinterpret(p1, "uint32x64"), full),
                                        zero,
                                        m1,
                                    )
                                acc[0] = S.vadd(acc[0], samples[0], full)
                                acc[0] = S.vadd(acc[0], samples[1], full)
                                acc[0] = S.vadd(acc[0], S.vselr(samples[0], next1), full)
                                acc[0] = S.vadd(acc[0], S.vselr(samples[1], next1), full)
                                acc[0] = S.vadd(acc[0], S.vselr(samples[0], next2), full)
                            avg = S.vmuls(acc[0], T.float32(1.0 / divisor), full)
                            narrow = S.vcvt(avg, dtype, sat=False)
                            S.vsts(output[pair * 2, 0], narrow, full, "PK_B32")
                    T.copy(output[:, :ow], Y[row, :, :])

    return main.with_attr("tilelang_callee_allocated_outputs", 0)


def program():
    return spatial_program("ceildiv2d")


@functools.lru_cache(maxsize=1)
def build():
    import tilelang

    return tilelang.compile(program(), target="ascend", out_idx=-1)
