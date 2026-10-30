"""Exact AlexNet paired-row specialization; validated numeric program unchanged."""

import functools

DTYPES = ("float16", "bfloat16", "float32")


def supports_call(call):
    return (
        call.n == 32
        and call.c_in == 64
        and tuple(call.size) == (55, 55)
        and tuple(call.window) == (3, 3)
        and tuple(call.stride) == (2, 2)
        and tuple(call.pad) == (0, 0)
        and tuple(call.dilation) == (1, 1)
        and call.ceil_mode is True
        and str(call.dtype).split(".")[-1] in DTYPES
    )


def program(dtype="float16", indices=False):
    if dtype not in DTYPES:
        raise ValueError("AlexNet paired candidate supports float16/bfloat16/float32")
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    rows, plane, staged_plane = 2048, 3025, 3072
    index_size = rows * 729 if indices else 1
    pick_rows = 14 if indices else 1
    raw_elements = 64 if dtype == "float32" else staged_plane

    @T.prim_func
    def main(
        X: T.Tensor((rows * plane,), dtype),
        Y: T.Tensor((rows * 729,), dtype),
        I: T.Tensor((index_size,), "int64"),
    ):
        with T.Kernel(64) as core:
            raw = T.alloc_shared((raw_elements,), dtype)
            data = T.alloc_shared((staged_plane,), "float32")
            output = T.alloc_shared((14, 64), dtype)
            picks = T.alloc_shared((pick_rows, 64), "uint32")
            if dtype == "float32":
                T.annotate_buffer_versions({data: 2, output: 2, picks: 2})
            else:
                T.annotate_buffer_versions({raw: 2, data: 1, output: 2, picks: 2})
            for step in T.Pipelined(rows // 64, num_stages=2):
                row = step * 64 + core
                if dtype == "float32":
                    T.copy(X[row * plane : (row + 1) * plane], data[:plane])
                else:
                    T.copy(X[row * plane : (row + 1) * plane], raw[:plane])
                with T.SimdVF():
                    full = S.pset(32)
                    if dtype != "float32":
                        for part in T.serial(staged_plane // 64):
                            converted = S.vcvt(
                                S.vld(raw[part * 64], "UNPK_B16"), "float32", full, part=0
                            )
                            S.vsts(data[part * 64], converted, full)
                        S.mem_bar("VST_VLD")
                    lane = S.vci(0, "int32")
                    izero = S.vdup(T.int32(0), "int32", full)
                    uzero = S.vdup(T.uint32(0), "uint32", full)
                    high = S.vcmps(lane, T.int32(27), full, "ge")
                    subtract = S.vsel(S.vdup(T.int32(27), "int32", full), izero, high)
                    high_y = S.vsel(S.vdup(T.int32(110), "int32", full), izero, high)
                    xbase = S.vmuls(S.vsub(lane, subtract, full), T.int32(2), full)
                    base = S.vadd(high_y, xbase, full)
                    pair_mask = S.vcmps(lane, T.int32(54), full, "lt")
                    offsets = S.alloc_local((9,), "uint32")
                    for tap in T.Unroll(9, explicit=True):
                        relative = S.vadds(base, T.int32(tap // 3 * 55 + tap % 3), full)
                        offsets[tap] = T.reinterpret(
                            S.vsel(relative, izero, pair_mask), "uint32x64"
                        )
                    value = S.alloc_local((1,), "float32")
                    pick = S.alloc_local((1,), "uint32")
                    for pair in T.serial(14):
                        active = S.vcmps(
                            lane, T.cast(T.min(54, 729 - pair * 54), "int32"), full, "lt"
                        )
                        first = S.vadds(offsets[0], T.cast(pair * 4 * 55, "uint32"), full)
                        # Mask BEFORE the gather: the unused high output row in
                        # the last pair would otherwise reach input rows55/56.
                        first_safe = S.vsel(first, uzero, active)
                        value[0] = S.vgather2(data[0], first_safe, full)
                        if indices:
                            pick[0] = first_safe
                        for tap in T.Unroll(1, 9, explicit=True):
                            position = S.vadds(offsets[tap], T.cast(pair * 4 * 55, "uint32"), full)
                            safe = S.vsel(position, uzero, active)
                            incoming = S.vgather2(data[0], safe, full)
                            take = S.pand(
                                active,
                                S.por(
                                    S.vcmp(incoming, value[0], full, "gt"),
                                    S.vcmp(incoming, incoming, full, "ne"),
                                    full,
                                ),
                                full,
                            )
                            value[0] = S.vsel(incoming, value[0], take)
                            if indices:
                                pick[0] = S.vsel(position, pick[0], take)
                        if dtype == "float32":
                            S.vsts(output[pair, 0], value[0], active)
                        else:
                            narrowed = S.vcvt(value[0], dtype, full, sat=False)
                            S.vsts(output[pair, 0], narrowed, active, "PK_B32")
                        if indices:
                            S.vsts(picks[pair, 0], pick[0], active)
                # Public output stays densely packed at729 elements per plane.
                # DMA each full54-value pair, then the final27-value row.
                for pair in T.serial(13):
                    begin = row * 729 + pair * 54
                    T.copy(output[pair, :54], Y[begin : begin + 54])
                T.copy(output[13, :27], Y[row * 729 + 702 : (row + 1) * 729])
                if indices:
                    with T.SimtVF(threads=128):
                        for col in T.Parallel(729):
                            I[row * 729 + col] = T.cast(picks[col // 54, col % 54], "int64")

    return main.with_attr("tilelang_callee_allocated_outputs", 0)


@functools.lru_cache(maxsize=6)
def build(dtype="float16", indices=False):
    import tilelang

    return tilelang.compile(program(dtype, indices), target="ascend", out_idx=[1, 2])
