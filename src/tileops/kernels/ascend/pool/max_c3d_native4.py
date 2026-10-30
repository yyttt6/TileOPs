"""C4: C3D pool2 indices, four rows / 128 native FP16 lanes and index DMA."""

import functools


def program(indices=True):
    if not indices:
        raise ValueError("C4 gate is indices-only; keep production values winner")
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    @T.prim_func
    def main(
        X: T.Tensor((512 * 16 * 3136,), "float16"),
        Y: T.Tensor((4096, 784), "float16"),
        I: T.Tensor((4096 * 784,), "int64"),
    ):
        with T.Kernel(64) as core:
            raw = T.alloc_shared((6272,), "float16")
            output = T.alloc_shared((7, 128), "float16")
            picks = T.alloc_shared((7, 128), "uint16")
            ids = T.alloc_shared((784,), "int64")
            T.annotate_buffer_versions({raw: 2, output: 2, picks: 2, ids: 2})
            for step in T.Pipelined(64, num_stages=2):
                job = step * 64 + core
                T.copy(X[job * 6272 : (job + 1) * 6272], raw)
                with T.SimdVF():
                    full = S.pset(16)
                    lane = S.vci(0, "int16")
                    zero = S.vdup(T.int16(0), "int16", full)
                    base = S.alloc_local((1,), "int16")
                    base[0] = S.vmuls(lane, T.int16(2), full)
                    # Each additional group moves down two input rows (112)
                    # and resets the x ramp (-56), net +56 per threshold.
                    for group in T.Unroll(1, 4, explicit=True):
                        high = S.vcmps(lane, T.int16(group * 28), full, "ge")
                        base[0] = S.vadd(
                            base[0], S.vsel(S.vdup(T.int16(56), "int16", full), zero, high), full
                        )
                    active = S.vcmps(lane, T.int16(112), full, "lt")
                    offsets = S.alloc_local((8,), "uint16")
                    for tap in T.Unroll(8, explicit=True):
                        relative = S.vadds(
                            base[0], T.int16(tap // 4 * 3136 + tap // 2 % 2 * 56 + tap % 2), full
                        )
                        offsets[tap] = T.reinterpret(S.vsel(relative, zero, active), "uint16x128")
                    best = S.alloc_local((1,), "float16")
                    pick = S.alloc_local((1,), "uint16")
                    for group in T.serial(7):
                        first = S.vadds(offsets[0], T.cast(group * 448, "uint16"), full)
                        best[0] = S.vgather2(raw[0], first, full)
                        pick[0] = first
                        for tap in T.Unroll(1, 8, explicit=True):
                            pos = S.vadds(offsets[tap], T.cast(group * 448, "uint16"), full)
                            incoming = S.vgather2(raw[0], pos, full)
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
                            pick[0] = S.vsel(pos, pick[0], take)
                        S.vsts(output[group, 0], best[0], active, "NORM_B16")
                        S.vsts(picks[group, 0], pick[0], active, "NORM_B16")
                T.copy(output[:, :112], Y[job, :])
                with T.SimtVF(threads=128):
                    for col in T.Parallel(784):
                        ids[col] = T.cast(picks[col // 112, col % 112], "int64") + (job % 8) * 6272
                T.copy(ids, I[job * 784 : (job + 1) * 784])

    return main.with_attr("tilelang_callee_allocated_outputs", 0)


@functools.lru_cache(None)
def build(indices=True):
    import tilelang

    return tilelang.compile(program(indices), target="ascend", out_idx=[1, 2])
