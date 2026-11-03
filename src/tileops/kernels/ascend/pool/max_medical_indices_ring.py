"""One exact Medical BF16 indices candidate: W/H/depth ring with provenance.

128 N*C jobs, 64 cores. Each real input plane is read once. Two contiguous
native16 vector loads plus register selection implement W; H stays in registers.
Three padded 2D planes retain previous odd/current even/current odd results.
Ordered folds preserve first finite ties/signed zero and last NaN payload.
W/H retain int16 plane offsets, with -1 unset. D uses uint16 window offsets
0..37631 (65535 unset), then widens and adds the signed depth base modulo 2^32.
All emitted indices are 0..401407; low/high32 interleave preserves int64 ABI.
"""

import functools


def program():
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    @T.prim_func
    def main(
        X: T.Tensor((128 * 32 * 112 * 112,), "bfloat16"),
        Y: T.Tensor((128, 17, 3249), "bfloat16"),
        I: T.Tensor((128, 17, 3249), "int64"),
    ):
        with T.Kernel(64) as core:
            # Two real planes + 128 defined guard elements for the final vld.
            raw = T.alloc_shared((25216,), "bfloat16")
            planes = T.alloc_shared((3, 58, 64), "bfloat16")
            plane_ids = T.alloc_shared((3, 58, 64), "int16")
            output = T.alloc_shared((29, 128), "bfloat16")
            words = T.alloc_shared((29, 256), "uint32")
            wide_ids = T.view(words, (29, 128), "int64")
            T.annotate_buffer_versions({raw: 2, planes: 1, plane_ids: 1, output: 2, words: 1})
            for channel_step in T.serial(2):
                channel = channel_step * 64 + core
                for oz in T.Pipelined(17, num_stages=2):
                    if oz < 16:
                        begin = (channel * 32 + oz * 2) * 12544
                        T.copy(X[begin : begin + 25088], raw[:25088])
                        with T.SimdVF():
                            full = S.pset(16)
                            lane = S.vci(0, "int16")
                            zero = S.vdup(T.int16(0), "int16", full)
                            neginf = S.vdup(T.bfloat16(float("-inf")), "bfloat16", full)
                            unset = S.vdup(T.int16(-1), "int16", full)
                            # Both vector-load footprints are now defined even at z=31.
                            S.vsts(raw[25088], neginf, full, "NORM_B16")
                            S.mem_bar("VST_VLD")
                            high = S.vcmps(lane, T.int16(64), full, "ge")
                            ox = S.vsub(
                                lane, S.vsel(S.vdup(T.int16(64), "int16", full), zero, high), full
                            )
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
                                hi[kx] = S.vmaxs(
                                    S.vadds(safe, T.int16(-128), full), T.int16(0), full
                                )
                                upper[kx] = S.vcmps(safe, T.int16(128), full, "ge")
                            next_lane = S.vmins(S.vadds(lane, T.int16(1), full), T.int16(127), full)
                            odd_lane = S.vmins(S.vadds(lane, T.int16(64), full), T.int16(127), full)
                            low_half = S.vcmps(lane, T.int16(64), full, "lt")
                            previous_odd = S.alloc_local((1,), "bfloat16")
                            previous_id = S.alloc_local((1,), "int16")
                            samples = S.alloc_local((3,), "bfloat16")
                            horizontal = S.alloc_local((1,), "bfloat16")
                            horizontal_id = S.alloc_local((1,), "int16")
                            vertical = S.alloc_local((1,), "bfloat16")
                            vertical_id = S.alloc_local((1,), "int16")
                            for parity in T.Unroll(2, explicit=True):
                                slot = 2 if parity == 0 else oz % 2
                                previous_odd[0] = neginf
                                previous_id[0] = unset
                                for pair_y in T.serial(56):
                                    first = S.vld(raw[parity * 12544 + pair_y * 224])
                                    second = S.vld(raw[parity * 12544 + pair_y * 224 + 128])
                                    for kx in T.Unroll(2, explicit=True):
                                        samples[kx] = S.vsel(
                                            S.vselr(second, hi[kx]),
                                            S.vselr(first, lo[kx]),
                                            upper[kx],
                                        )
                                    samples[2] = S.vselr(samples[0], next_lane)
                                    horizontal[0] = S.vsel(samples[0], neginf, valid[0])
                                    first_id = S.vadds(
                                        S.vadd(row_offset, x0, full),
                                        T.cast(pair_y * 224, "int16"),
                                        full,
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
                                            S.vadds(first_id, T.int16(kx + 1), full),
                                            horizontal_id[0],
                                            take,
                                        )
                                    # Lower half = even input row; gather odd row into lower half.
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
                                    vertical[0] = S.vsel(horizontal[0], previous_odd[0], take_even)
                                    vertical_id[0] = S.vsel(
                                        horizontal_id[0], previous_id[0], take_even
                                    )
                                    better_odd = S.por(
                                        S.vcmp(odd_row, vertical[0], full, "gt"),
                                        S.vcmp(odd_row, odd_row, full, "ne"),
                                        full,
                                    )
                                    take_odd = S.pand(
                                        S.vcmps(odd_id, T.int16(0), full, "ge"),
                                        S.por(
                                            better_odd,
                                            S.vcmps(vertical_id[0], T.int16(0), full, "lt"),
                                            full,
                                        ),
                                        full,
                                    )
                                    vertical[0] = S.vsel(odd_row, vertical[0], take_odd)
                                    vertical_id[0] = S.vsel(odd_id, vertical_id[0], take_odd)
                                    S.vsts(
                                        planes[slot, pair_y, 0], vertical[0], low_half, "NORM_B16"
                                    )
                                    S.vsts(
                                        plane_ids[slot, pair_y, 0],
                                        vertical_id[0],
                                        low_half,
                                        "NORM_B16",
                                    )
                                    previous_odd[0] = odd_row
                                    previous_id[0] = odd_id
                                # Ceil output y=56 contains only real input y=111.
                                # Upper half is -inf, defining the padded row 57 as well.
                                S.vsts(planes[slot, 56, 0], previous_odd[0], full, "NORM_B16")
                                S.vsts(plane_ids[slot, 56, 0], previous_id[0], full, "NORM_B16")
                    with T.SimdVF():
                        full = S.pset(16)
                        lane = S.vci(0, "int16")
                        zero = S.vdup(T.int16(0), "int16", full)
                        shift = S.vsel(
                            S.vdup(T.int16(7), "int16", full),
                            zero,
                            S.vcmps(lane, T.int16(57), full, "ge"),
                        )
                        packed_lane = S.vmins(S.vadd(lane, shift, full), T.int16(127), full)
                        best = S.alloc_local((1,), "bfloat16")
                        delta = S.alloc_local((1,), "uint16")
                        unset_delta = S.vdup(T.uint16(65535), "uint16", full)
                        full32 = S.pset(32)
                        lane32 = T.reinterpret(S.vci(0, "int32"), "uint32x64")
                        zero32 = S.vdup(T.uint32(0), "uint32", full32)
                        half_lane32 = S.vshrs(lane32, T.uint32(1), full32)
                        even_word = S.vcmp(
                            lane32, S.vmuls(half_lane32, T.uint32(2), full32), full32, "eq"
                        )
                        halves = S.alloc_local((2,), "uint32")
                        # Explicit VST->VLD ordering across the plane producer/consumer.
                        S.mem_bar("VST_VLD")
                        for pair in T.serial(29):
                            if oz < 16:
                                even = S.vld(planes[2, pair * 2, 0])
                                odd = S.vld(planes[oz % 2, pair * 2, 0])
                                even_id = S.vld(plane_ids[2, pair * 2, 0])
                                odd_id = S.vld(plane_ids[oz % 2, pair * 2, 0])
                                even_delta = S.vadds(
                                    T.reinterpret(even_id, "uint16x128"), T.uint16(12544), full
                                )
                                odd_delta = S.vadds(
                                    T.reinterpret(odd_id, "uint16x128"), T.uint16(25088), full
                                )
                                if oz == 0:
                                    best[0] = even
                                    delta[0] = S.vsel(
                                        even_delta,
                                        unset_delta,
                                        S.vcmps(even_id, T.int16(0), full, "ge"),
                                    )
                                else:
                                    previous = S.vld(planes[1 - oz % 2, pair * 2, 0])
                                    previous_id = S.vld(plane_ids[1 - oz % 2, pair * 2, 0])
                                    previous_delta = T.reinterpret(previous_id, "uint16x128")
                                    better_even = S.por(
                                        S.vcmp(even, previous, full, "gt"),
                                        S.vcmp(even, even, full, "ne"),
                                        full,
                                    )
                                    take_even = S.pand(
                                        S.vcmps(even_id, T.int16(0), full, "ge"),
                                        S.por(
                                            better_even,
                                            S.vcmps(previous_id, T.int16(0), full, "lt"),
                                            full,
                                        ),
                                        full,
                                    )
                                    best[0] = S.vsel(even, previous, take_even)
                                    delta[0] = S.vsel(even_delta, previous_delta, take_even)
                                better_odd = S.por(
                                    S.vcmp(odd, best[0], full, "gt"),
                                    S.vcmp(odd, odd, full, "ne"),
                                    full,
                                )
                                take_odd = S.pand(
                                    S.vcmps(odd_id, T.int16(0), full, "ge"),
                                    S.por(
                                        better_odd,
                                        S.vcmps(delta[0], T.uint16(65535), full, "eq"),
                                        full,
                                    ),
                                    full,
                                )
                                best[0] = S.vsel(odd, best[0], take_odd)
                                delta[0] = S.vsel(odd_delta, delta[0], take_odd)
                            else:
                                # Ceil output z=16 contains only real input z=31 (odd slot 1).
                                best[0] = S.vld(planes[1, pair * 2, 0])
                                delta[0] = T.reinterpret(
                                    S.vld(plane_ids[1, pair * 2, 0]), "uint16x128"
                                )
                            active = S.vcmps(
                                lane, T.cast(T.min(114, 3249 - pair * 114), "int16"), full, "lt"
                            )
                            S.vsts(
                                output[pair, 0], S.vselr(best[0], packed_lane), active, "NORM_B16"
                            )
                            packed_delta = S.vselr(delta[0], packed_lane)
                            halves[0] = S.vunpack(packed_delta, part="LOWER")
                            halves[1] = S.vunpack(packed_delta, part="HIGHER")
                            for half in T.Unroll(2, explicit=True):
                                index32 = S.vadds(
                                    halves[half], T.cast((oz * 2 - 1) * 12544, "uint32"), full32
                                )
                                for chunk in T.Unroll(2, explicit=True):
                                    source_lane = S.vadds(half_lane32, T.uint32(chunk * 32), full32)
                                    low_word = S.vselr(index32, source_lane)
                                    packed_words = S.vsel(low_word, zero32, even_word)
                                    S.vsts(
                                        words[pair, half * 128 + chunk * 64], packed_words, full32
                                    )
                    # 28 bursts of 228 B + one 114 B tail, versus 57 x 114 B.
                    T.copy(output[:28, :114], Y[channel, oz, :3192])
                    T.copy(output[28, :57], Y[channel, oz, 3192:3249])
                    T.copy(wide_ids[:28, :114], I[channel, oz, :3192])
                    T.copy(wide_ids[28, :57], I[channel, oz, 3192:3249])

    return main.with_attr("tilelang_callee_allocated_outputs", 0)


@functools.lru_cache(None)
def build():
    import tilelang

    return tilelang.compile(program(), target="ascend", out_idx=[1, 2])


def supports_call(call, indices=True):
    return (
        indices is True
        and call.n == 2
        and call.c_in == 64
        and tuple(call.size) == (32, 112, 112)
        and tuple(call.window) == (3, 3, 3)
        and tuple(call.stride) == (2, 2, 2)
        and tuple(call.pad) == (1, 1, 1)
        and tuple(call.dilation) == (1, 1, 1)
        and call.ceil_mode is True
        and str(call.dtype).split(".")[-1] == "bfloat16"
    )
