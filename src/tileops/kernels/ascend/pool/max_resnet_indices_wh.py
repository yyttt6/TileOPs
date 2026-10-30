"""One exact ResNet indices design for FP16/BF16; no automatic registration.

Contiguous native16 loads -> ordered W/H folds -> register row packing.
One N*C channel per job, 64 cores, raw/output two-stage buffers. Full stores
own independent 128-lane slots; only the first112 lanes per slot reach GM.
Int16 W/H winners carry plane offsets, -1 unset. SIMD uint32 low/high words
are stored through a real int64 view. Raw, values and indices use two slots.
"""

import functools


def program(dtype="float16"):
    if dtype not in ("float16", "bfloat16"):
        raise ValueError("Exact ResNet FP16/BF16 indices only")
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    @T.prim_func
    def main(
        X: T.Tensor((2048 * 12544,), dtype),
        Y: T.Tensor((2048, 3136), dtype),
        I: T.Tensor((2048, 3136), "int64"),
    ):
        with T.Kernel(64) as core:
            raw = T.alloc_shared((12672,), dtype)
            output = T.alloc_shared((28, 128), dtype)
            words = T.alloc_shared((28, 256), "uint32")
            wide_ids = T.view(words, (28, 128), "int64")
            T.annotate_buffer_versions({raw: 2, output: 2, words: 2})
            for step in T.Pipelined(32, num_stages=2):
                channel = step * 64 + core
                T.copy(X[channel * 12544 : (channel + 1) * 12544], raw[:12544])
                with T.SimdVF():
                    full = S.pset(16)
                    lane = S.vci(0, "int16")
                    zero = S.vdup(T.int16(0), "int16", full)
                    neginf = S.vdup(T.cast(float("-inf"), dtype), dtype, full)
                    unset = S.vdup(T.int16(-1), "int16", full)
                    S.vsts(raw[12544], neginf, full, "NORM_B16")
                    S.mem_bar("VST_VLD")
                    high = S.vcmps(lane, T.int16(64), full, "ge")
                    ox = S.vsub(lane, S.vsel(S.vdup(T.int16(64), "int16", full), zero, high), full)
                    row_offset = S.vsel(S.vdup(T.int16(112), "int16", full), zero, high)
                    x0 = S.vadds(S.vmuls(ox, T.int16(2), full), T.int16(-1), full)
                    valid = S.alloc_local((3,), "bool")
                    lo = S.alloc_local((2,), "int16")
                    hi = S.alloc_local((2,), "int16")
                    upper = S.alloc_local((2,), "bool")
                    for kx in T.Unroll(3, explicit=True):
                        xpos = S.vadds(x0, T.int16(kx), full)
                        valid[kx] = S.pand(
                            S.vcmps(xpos, T.int16(0), full, "ge"),
                            S.vcmps(xpos, T.int16(112), full, "lt"),
                            full,
                        )
                    for kx in T.Unroll(2, explicit=True):
                        relative = S.vadds(S.vadd(row_offset, x0, full), T.int16(kx), full)
                        safe = S.vsel(relative, zero, valid[kx])
                        lo[kx] = S.vmins(safe, T.int16(127), full)
                        hi[kx] = S.vmaxs(S.vadds(safe, T.int16(-128), full), T.int16(0), full)
                        upper[kx] = S.vcmps(safe, T.int16(128), full, "ge")
                    next_lane = S.vmins(S.vadds(lane, T.int16(1), full), T.int16(127), full)
                    odd_lane = S.vmins(S.vadds(lane, T.int16(64), full), T.int16(127), full)
                    second_row_lane = S.vmaxs(S.vadds(lane, T.int16(-56), full), T.int16(0), full)
                    second_row = S.vcmps(lane, T.int16(56), full, "ge")
                    previous_odd = S.alloc_local((1,), dtype)
                    previous_id = S.alloc_local((1,), "int16")
                    samples = S.alloc_local((3,), dtype)
                    horizontal = S.alloc_local((1,), dtype)
                    horizontal_id = S.alloc_local((1,), "int16")
                    rows = S.alloc_local((2,), dtype)
                    row_ids = S.alloc_local((2,), "int16")
                    halves = S.alloc_local((2,), "uint32")
                    full32 = S.pset(32)
                    lane32 = T.reinterpret(S.vci(0, "int32"), "uint32x64")
                    zero32 = S.vdup(T.uint32(0), "uint32", full32)
                    half_lane32 = S.vshrs(lane32, T.uint32(1), full32)
                    even_word = S.vcmp(
                        lane32, S.vmuls(half_lane32, T.uint32(2), full32), full32, "eq"
                    )
                    previous_odd[0] = neginf
                    previous_id[0] = unset
                    for pair in T.serial(28):
                        for sub in T.Unroll(2, explicit=True):
                            # pair_y produces output y=pair_y from input2y-1,2y,2y+1.
                            pair_y = pair * 2 + sub
                            first = S.vld(raw[pair_y * 224])
                            second = S.vld(raw[pair_y * 224 + 128])
                            for kx in T.Unroll(2, explicit=True):
                                samples[kx] = S.vsel(
                                    S.vselr(second, hi[kx]), S.vselr(first, lo[kx]), upper[kx]
                                )
                            # The ox56 source lane is retained as ResNet's x111 halo.
                            samples[2] = S.vselr(samples[0], next_lane)
                            horizontal[0] = S.vsel(samples[0], neginf, valid[0])
                            first_id = S.vadds(
                                S.vadd(row_offset, x0, full), T.cast(pair_y * 224, "int16"), full
                            )
                            horizontal_id[0] = S.vsel(first_id, unset, valid[0])
                            for kx in T.Unroll(2, explicit=True):
                                incoming = samples[kx + 1]
                                better = S.por(
                                    S.vcmp(incoming, horizontal[0], full, "gt"),
                                    S.vcmp(incoming, incoming, full, "ne"),
                                    full,
                                )
                                take = S.pand(
                                    valid[kx + 1],
                                    S.por(
                                        better,
                                        S.vcmps(horizontal_id[0], T.int16(0), full, "lt"),
                                        full,
                                    ),
                                    full,
                                )
                                horizontal[0] = S.vsel(incoming, horizontal[0], take)
                                horizontal_id[0] = S.vsel(
                                    S.vadds(first_id, T.int16(kx + 1), full), horizontal_id[0], take
                                )
                            odd_row = S.vselr(horizontal[0], odd_lane)
                            odd_id = S.vselr(horizontal_id[0], odd_lane)
                            better_even = S.por(
                                S.vcmp(horizontal[0], previous_odd[0], full, "gt"),
                                S.vcmp(horizontal[0], horizontal[0], full, "ne"),
                                full,
                            )
                            take_even = S.pand(
                                S.vcmps(horizontal_id[0], T.int16(0), full, "ge"),
                                S.por(
                                    better_even,
                                    S.vcmps(previous_id[0], T.int16(0), full, "lt"),
                                    full,
                                ),
                                full,
                            )
                            rows[sub] = S.vsel(horizontal[0], previous_odd[0], take_even)
                            row_ids[sub] = S.vsel(horizontal_id[0], previous_id[0], take_even)
                            better_odd = S.por(
                                S.vcmp(odd_row, rows[sub], full, "gt"),
                                S.vcmp(odd_row, odd_row, full, "ne"),
                                full,
                            )
                            take_odd = S.pand(
                                S.vcmps(odd_id, T.int16(0), full, "ge"),
                                S.por(
                                    better_odd, S.vcmps(row_ids[sub], T.int16(0), full, "lt"), full
                                ),
                                full,
                            )
                            rows[sub] = S.vsel(odd_row, rows[sub], take_odd)
                            row_ids[sub] = S.vsel(odd_id, row_ids[sub], take_odd)
                            previous_odd[0] = odd_row
                            previous_id[0] = odd_id
                        # [row0[0:56],row1[0:56],16 unused defined lanes].
                        # Each full store owns exactly output[pair,0:128].
                        packed = S.vsel(S.vselr(rows[1], second_row_lane), rows[0], second_row)
                        S.vsts(output[pair, 0], packed, full, "NORM_B16")
                        packed_ids = S.vsel(
                            S.vselr(row_ids[1], second_row_lane), row_ids[0], second_row
                        )
                        unsigned_ids = T.reinterpret(packed_ids, "uint16x128")
                        halves[0] = S.vunpack(unsigned_ids, part="LOWER")
                        halves[1] = S.vunpack(unsigned_ids, part="HIGHER")
                        for half in T.Unroll(2, explicit=True):
                            for chunk in T.Unroll(2, explicit=True):
                                from_lane = S.vadds(half_lane32, T.uint32(chunk * 32), full32)
                                low_word = S.vselr(halves[half], from_lane)
                                packed_words = S.vsel(low_word, zero32, even_word)
                                S.vsts(words[pair, half * 128 + chunk * 64], packed_words, full32)
                T.copy(output[:, :112], Y[channel, :])
                T.copy(wide_ids[:, :112], I[channel, :])

    return main.with_attr("tilelang_callee_allocated_outputs", 0)


@functools.lru_cache(maxsize=2)
def build(dtype="float16"):
    import tilelang

    return tilelang.compile(program(dtype), target="ascend", out_idx=[1, 2])


DTYPES = ("float16", "bfloat16")


def supports_call(call, indices=True):
    return (
        indices is True
        and call.n == 32
        and call.c_in == 64
        and tuple(call.size) == (112, 112)
        and tuple(call.window) == (3, 3)
        and tuple(call.stride) == (2, 2)
        and tuple(call.pad) == (1, 1)
        and tuple(call.dilation) == (1, 1)
        and call.ceil_mode is False
        and str(call.dtype).split(".")[-1] in DTYPES
    )
