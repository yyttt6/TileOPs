"""Exact C3D-pool2 specialization; only values selected by production dispatch."""

import functools


def supports_call(call):
    return (
        call.n == 4
        and call.c_in == 128
        and tuple(call.size) == (16, 56, 56)
        and tuple(call.window) == (2, 2, 2)
        and tuple(call.stride) == (2, 2, 2)
        and tuple(call.pad) == (0, 0, 0)
        and tuple(call.dilation) == (1, 1, 1)
        and call.ceil_mode is False
        and str(call.dtype).split(".")[-1] == "float16"
    )


def program(indices=False):
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    pick_rows = 14 if indices else 1
    index_size = 4096 * 784 if indices else 1

    @T.prim_func
    def main(
        X: T.Tensor((512 * 16 * 56 * 56,), "float16"),
        Y: T.Tensor((4096, 14, 56), "float16"),
        I: T.Tensor((index_size,), "int64"),
    ):
        with T.Kernel(64) as core:
            raw = T.alloc_shared((6272,), "float16")
            data = T.alloc_shared((6272,), "float32")
            output = T.alloc_shared((14, 64), "float16")
            picks = T.alloc_shared((pick_rows, 64), "uint32")
            T.annotate_buffer_versions({raw: 2, data: 1, output: 2, picks: 2})
            for step in T.Pipelined(64, num_stages=2):
                job = step * 64 + core
                T.copy(X[job * 6272 : (job + 1) * 6272], raw)
                with T.SimdVF():
                    full = S.pset(32)
                    for part in T.serial(98):
                        wide = S.vcvt(S.vld(raw[part * 64], "UNPK_B16"), "float32", full, part=0)
                        S.vsts(data[part * 64], wide, full)
                    S.mem_bar("VST_VLD")
                    lane = S.vci(0, "int32")
                    zero = S.vdup(T.int32(0), "int32", full)
                    high = S.vcmps(lane, T.int32(28), full, "ge")
                    subtract = S.vsel(S.vdup(T.int32(28), "int32", full), zero, high)
                    high_y = S.vsel(S.vdup(T.int32(112), "int32", full), zero, high)
                    base = S.vadd(
                        high_y, S.vmuls(S.vsub(lane, subtract, full), T.int32(2), full), full
                    )
                    active = S.vcmps(lane, T.int32(56), full, "lt")
                    offsets = S.alloc_local((8,), "uint32")
                    for tap in T.Unroll(8, explicit=True):
                        relative = S.vadds(
                            base, T.int32(tap // 4 * 3136 + tap // 2 % 2 * 56 + tap % 2), full
                        )
                        offsets[tap] = T.reinterpret(S.vsel(relative, zero, active), "uint32x64")
                    best = S.alloc_local((1,), "float32")
                    pick = S.alloc_local((1,), "uint32")
                    for pair in T.serial(14):
                        first = S.vadds(offsets[0], T.cast(pair * 224, "uint32"), full)
                        best[0] = S.vgather2(data[0], first, full)
                        if indices:
                            pick[0] = first
                        for tap in T.Unroll(1, 8, explicit=True):
                            pos = S.vadds(offsets[tap], T.cast(pair * 224, "uint32"), full)
                            incoming = S.vgather2(data[0], pos, full)
                            take = S.pand(
                                active,
                                S.por(
                                    S.vcmp(incoming, best[0], full, "gt"),
                                    S.vcmp(incoming, incoming, full, "ne"),
                                    full,
                                ),
                                full,
                            )
                            best[0] = S.vsel(incoming, best[0], take)
                            if indices:
                                pick[0] = S.vsel(pos, pick[0], take)
                        narrowed = S.vcvt(best[0], "float16", full, sat=False)
                        S.vsts(output[pair, 0], narrowed, active, "PK_B32")
                        if indices:
                            S.vsts(picks[pair, 0], pick[0], active)
                T.copy(output[:, :56], Y[job, :, :])
                if indices:
                    with T.SimtVF(threads=128):
                        for col in T.Parallel(784):
                            I[job * 784 + col] = (
                                T.cast(picks[col // 56, col % 56], "int64") + (job % 8) * 6272
                            )

    return main.with_attr("tilelang_callee_allocated_outputs", 0)


@functools.lru_cache(maxsize=2)
def build(indices=False):
    import tilelang

    return tilelang.compile(program(indices), target="ascend", out_idx=[1, 2])
