"""Exact windowed FP32 Radix Select for the mainline TileLang Ascend backend."""

import tilelang
import tilelang.ascend.language as T


@tilelang.jit(target="ascend", out_idx=[])
def topk_select_ascend(R: int, N: int, K: int, W: int, G: int):
    def score_key(value):
        bits = T.reinterpret(value, T.uint32)
        magnitude = bits & T.Cast(T.uint32, 0x7FFFFFFF)
        ordered = T.if_then_else(
            (bits & T.Cast(T.uint32, 0x80000000)) != 0, ~bits, bits | T.Cast(T.uint32, 0x80000000)
        )
        zeroed = T.if_then_else(magnitude == 0, T.Cast(T.uint32, 0x80000000), ordered)
        return T.if_then_else(
            magnitude > T.Cast(T.uint32, 0x7F800000), T.Cast(T.uint32, 0xFFFFFFFF), zeroed
        )

    def coarse_byte(value):
        bits = T.reinterpret(T.Cast(T.float16, value), T.uint16)
        magnitude = bits & T.Cast(T.uint16, 0x7FFF)
        ordered = T.if_then_else(
            (bits & T.Cast(T.uint16, 0x8000)) != 0, ~bits, bits | T.Cast(T.uint16, 0x8000)
        )
        zeroed = T.if_then_else(magnitude == 0, T.Cast(T.uint16, 0x8000), ordered)
        canonical = T.if_then_else(
            magnitude > T.Cast(T.uint16, 0x7C00), T.Cast(T.uint16, 0xFFFF), zeroed
        )
        return T.Cast(T.int32, (canonical >> 8) & T.Cast(T.uint16, 255))

    def byte_shift(round):
        return 24 - T.max(round - 1, 0) * 8

    M = 2048
    H = 128
    rounds = 5

    @T.prim_func
    def main(
        x: T.Tensor((R // G, N, G), T.float32),
        starts: T.Tensor((R // G,), T.int32),
        ends: T.Tensor((R // G,), T.int32),
        out: T.Tensor((R, K), T.int32),
    ):
        with T.Kernel(W) as worker:
            hist = T.alloc_shared((257,), T.int32)
            group_offsets = T.alloc_shared((16,), T.int32)
            local_prefix = T.alloc_shared((256,), T.int32)
            # Remaining quota, committed output count, threshold, active count,
            # accepted count, previous candidate count, next candidate count.
            state = T.alloc_shared((7,), T.int32)
            cached_key = T.alloc_shared((2, M), T.uint32)
            cached_index = T.alloc_shared((2, M), T.int32)
            prefix = T.alloc_shared((3,), T.uint32)
            tie_prefix = T.alloc_shared((H,), T.int32)
            for row_iter in T.serial(T.ceildiv(R, W)):
                row = worker + row_iter * W
                if row < R:
                    with T.SimtVF(threads=H):
                        for i in T.Parallel(K):
                            out[row, i] = T.if_then_else(
                                (ends[row // G] - starts[row // G] <= K)
                                and (starts[row // G] + i < ends[row // G]),
                                starts[row // G] + i,
                                N,
                            )
                        for i in T.Parallel(7):
                            state[i] = 0
                        for i in T.Parallel(3):
                            prefix[i] = 0
                    with T.SimtVF(threads=H):
                        for thread in T.Parallel(H):
                            if thread == 0:
                                state[0] = T.if_then_else(
                                    ends[row // G] - starts[row // G] > K, K, 0
                                )
                    for round in T.serial(rounds):
                        with T.SimtVF(threads=H):
                            for i in T.Parallel(257):
                                hist[i] = 0
                            for thread in T.Parallel(H):
                                if thread == 0:
                                    state[5] = state[6]
                                    state[6] = 0
                                    state[2] = -1
                                    state[3] = 0
                                    state[4] = 0
                        with T.SimtVF(threads=H):
                            cached_count = T.alloc_var(T.int32)
                            cached_count = state[5]
                            if (round == 0) or (cached_count > M):
                                for tile in T.serial(T.ceildiv(N, H)):
                                    for lane in T.Parallel(H):
                                        i = tile * H + lane
                                        if (
                                            state[0] > 0
                                            and i < N
                                            and i >= starts[row // G]
                                            and i < ends[row // G]
                                        ):
                                            value = x[row // G, i, row % G]
                                            key = T.if_then_else(
                                                round == 0,
                                                T.Cast(T.uint32, 0),
                                                score_key(value),
                                            )
                                            candidate = i
                                            if (
                                                round == 0
                                                or coarse_byte(value) == T.Cast(T.int32, prefix[2])
                                            ) and (key & prefix[1]) == prefix[0]:
                                                digit = T.alloc_var(T.int32)
                                                digit = T.if_then_else(
                                                    round == 0,
                                                    coarse_byte(value),
                                                    T.Cast(
                                                        T.int32,
                                                        (key >> (byte_shift(round)))
                                                        & T.Cast(T.uint32, 255),
                                                    ),
                                                )
                                                T.assume(digit >= 0)
                                                T.assume(digit < 256)
                                                T.atomic_add(hist[T.max(0, T.min(digit, 255))], 1)
                            else:
                                if round % 2 == 0:
                                    for tile in T.serial(T.ceildiv(T.min(cached_count, M), H)):
                                        for lane in T.Parallel(H):
                                            i = tile * H + lane
                                            if state[0] > 0 and i < M and i < cached_count:
                                                key = cached_key[1, T.max(0, T.min(i, M - 1))]
                                                candidate = cached_index[
                                                    1, T.max(0, T.min(i, M - 1))
                                                ]
                                                digit_byte = T.alloc_var(T.uint8)
                                                digit_byte = T.Cast(
                                                    T.uint8, key >> (byte_shift(round))
                                                )
                                                digit = T.Cast(T.int32, digit_byte)
                                                T.assume(digit >= 0)
                                                T.assume(digit < 256)
                                                T.atomic_add(hist[T.max(0, T.min(digit, 255))], 1)
                                else:
                                    for tile in T.serial(T.ceildiv(T.min(cached_count, M), H)):
                                        for lane in T.Parallel(H):
                                            i = tile * H + lane
                                            if state[0] > 0 and i < M and i < cached_count:
                                                key = cached_key[0, T.max(0, T.min(i, M - 1))]
                                                candidate = cached_index[
                                                    0, T.max(0, T.min(i, M - 1))
                                                ]
                                                digit_byte = T.alloc_var(T.uint8)
                                                digit_byte = T.Cast(
                                                    T.uint8, key >> (byte_shift(round))
                                                )
                                                digit = T.Cast(T.int32, digit_byte)
                                                T.assume(digit >= 0)
                                                T.assume(digit < 256)
                                                T.atomic_add(hist[T.max(0, T.min(digit, 255))], 1)
                        # Keep producer and consumer work in separate VF regions.
                        with T.SimtVF(threads=H):
                            for group in T.Parallel(16):
                                total = T.alloc_var(T.int32)
                                total = 0
                                for j in T.serial(16):
                                    bucket = 255 - group * 16 - j
                                    local_prefix[bucket] = total
                                    total = total + hist[bucket]
                                group_offsets[group] = total
                        with T.SimtVF(threads=H):
                            for thread in T.Parallel(H):
                                if thread == 0 and state[0] > 0:
                                    total = T.alloc_var(T.int32)
                                    count = T.alloc_var(T.int32)
                                    total = 0
                                    for group in T.serial(16):
                                        count = group_offsets[group]
                                        group_offsets[group] = total
                                        total = total + count
                                    state[3] = total
                                    state[4] = total
                        with T.SimtVF(threads=H):
                            for bucket in T.Parallel(256):
                                if state[0] > 0:
                                    before = T.alloc_var(T.int32)
                                    through = T.alloc_var(T.int32)
                                    before = (
                                        local_prefix[bucket] + group_offsets[(255 - bucket) // 16]
                                    )
                                    through = before + hist[bucket]
                                    hist[bucket] = before
                                    # Only one nonempty bucket can contain the quota.
                                    if before <= state[0] and through > state[0]:
                                        state[2] = bucket
                                        state[4] = before
                        with T.SimtVF(threads=H):
                            for thread in T.Parallel(H):
                                if thread == 0:
                                    state[0] = state[0] - state[4]
                        with T.SimtVF(threads=H):
                            cached_count = T.alloc_var(T.int32)
                            cached_count = state[5]
                            if (round == 0) or (cached_count > M):
                                for tile in T.serial(T.ceildiv(N, H)):
                                    for lane in T.Parallel(H):
                                        i = tile * H + lane
                                        if (
                                            state[3] > 0
                                            and i < N
                                            and i >= starts[row // G]
                                            and i < ends[row // G]
                                        ):
                                            value = x[row // G, i, row % G]
                                            key = T.if_then_else(
                                                round == 0,
                                                T.Cast(T.uint32, 0),
                                                score_key(value),
                                            )
                                            candidate = i
                                            if (
                                                round == 0
                                                or coarse_byte(value) == T.Cast(T.int32, prefix[2])
                                            ) and (key & prefix[1]) == prefix[0]:
                                                digit = T.alloc_var(T.int32)
                                                digit = T.if_then_else(
                                                    round == 0,
                                                    coarse_byte(value),
                                                    T.Cast(
                                                        T.int32,
                                                        (key >> (byte_shift(round)))
                                                        & T.Cast(T.uint32, 255),
                                                    ),
                                                )
                                                T.assume(digit >= 0)
                                                T.assume(digit < 256)
                                                if digit > state[2]:
                                                    pos = (
                                                        T.atomic_add(
                                                            hist[T.max(0, T.min(digit, 255))],
                                                            1,
                                                            return_prev=True,
                                                        )
                                                        + state[1]
                                                    )
                                                    if pos < K:
                                                        out[row, pos] = candidate
                                                elif digit == state[2] and state[0] > 0:
                                                    slot = T.atomic_add(
                                                        state[6], 1, return_prev=True
                                                    )
                                                    if slot >= 0 and slot < M:
                                                        cached_index[
                                                            round % 2, T.max(0, T.min(slot, M - 1))
                                                        ] = candidate
                                                        cached_key[
                                                            round % 2, T.max(0, T.min(slot, M - 1))
                                                        ] = T.if_then_else(
                                                            round == 0,
                                                            score_key(value),
                                                            key,
                                                        )
                            else:
                                if round % 2 == 0:
                                    for tile in T.serial(T.ceildiv(T.min(cached_count, M), H)):
                                        for lane in T.Parallel(H):
                                            i = tile * H + lane
                                            if state[3] > 0 and i < M and i < cached_count:
                                                key = cached_key[1, T.max(0, T.min(i, M - 1))]
                                                candidate = cached_index[
                                                    1, T.max(0, T.min(i, M - 1))
                                                ]
                                                digit_byte = T.alloc_var(T.uint8)
                                                digit_byte = T.Cast(
                                                    T.uint8, key >> (byte_shift(round))
                                                )
                                                digit = T.Cast(T.int32, digit_byte)
                                                T.assume(digit >= 0)
                                                T.assume(digit < 256)
                                                if digit > state[2]:
                                                    pos = (
                                                        T.atomic_add(
                                                            hist[T.max(0, T.min(digit, 255))],
                                                            1,
                                                            return_prev=True,
                                                        )
                                                        + state[1]
                                                    )
                                                    if pos < K:
                                                        out[row, pos] = candidate
                                                elif digit == state[2] and state[0] > 0:
                                                    slot = T.atomic_add(
                                                        state[6], 1, return_prev=True
                                                    )
                                                    if slot >= 0 and slot < M:
                                                        cached_index[
                                                            0, T.max(0, T.min(slot, M - 1))
                                                        ] = candidate
                                                        cached_key[
                                                            0, T.max(0, T.min(slot, M - 1))
                                                        ] = key
                                else:
                                    for tile in T.serial(T.ceildiv(T.min(cached_count, M), H)):
                                        for lane in T.Parallel(H):
                                            i = tile * H + lane
                                            if state[3] > 0 and i < M and i < cached_count:
                                                key = cached_key[0, T.max(0, T.min(i, M - 1))]
                                                candidate = cached_index[
                                                    0, T.max(0, T.min(i, M - 1))
                                                ]
                                                digit_byte = T.alloc_var(T.uint8)
                                                digit_byte = T.Cast(
                                                    T.uint8, key >> (byte_shift(round))
                                                )
                                                digit = T.Cast(T.int32, digit_byte)
                                                T.assume(digit >= 0)
                                                T.assume(digit < 256)
                                                if digit > state[2]:
                                                    pos = (
                                                        T.atomic_add(
                                                            hist[T.max(0, T.min(digit, 255))],
                                                            1,
                                                            return_prev=True,
                                                        )
                                                        + state[1]
                                                    )
                                                    if pos < K:
                                                        out[row, pos] = candidate
                                                elif digit == state[2] and state[0] > 0:
                                                    slot = T.atomic_add(
                                                        state[6], 1, return_prev=True
                                                    )
                                                    if slot >= 0 and slot < M:
                                                        cached_index[
                                                            1, T.max(0, T.min(slot, M - 1))
                                                        ] = candidate
                                                        cached_key[
                                                            1, T.max(0, T.min(slot, M - 1))
                                                        ] = key
                        with T.SimtVF(threads=H):
                            for thread in T.Parallel(H):
                                if thread == 0:
                                    state[1] = state[1] + state[4]
                                    if state[2] >= 0:
                                        if round == 0:
                                            prefix[2] = T.Cast(T.uint32, state[2])
                                        else:
                                            shift = 24 - (round - 1) * 8
                                            prefix[0] = prefix[0] | (
                                                T.Cast(T.uint32, state[2]) << shift
                                            )
                                            prefix[1] = prefix[1] | (T.Cast(T.uint32, 255) << shift)
                    # Contiguous thread segments preserve ascending absolute-index tie selection.
                    with T.SimtVF(threads=H):
                        for thread in T.Parallel(H):
                            count = T.alloc_var(T.int32)
                            count = 0
                            for offset in T.serial(T.ceildiv(N, H)):
                                i = thread * T.ceildiv(N, H) + offset
                                if (  # noqa: SIM102 -- guard the score load before evaluating its key
                                    state[0] > 0
                                    and i < N
                                    and i >= starts[row // G]
                                    and i < ends[row // G]
                                ):
                                    if score_key(x[row // G, i, row % G]) == prefix[0]:
                                        count = count + 1
                            tie_prefix[thread] = count
                    with T.SimtVF(threads=H):
                        for thread in T.Parallel(H):
                            if thread == 0:
                                base = T.alloc_var(T.int32)
                                value = T.alloc_var(T.int32)
                                base = 0
                                for i in T.serial(H):
                                    value = tie_prefix[i]
                                    tie_prefix[i] = base
                                    base = base + value
                    with T.SimtVF(threads=H):
                        for thread in T.Parallel(H):
                            position = T.alloc_var(T.int32)
                            position = state[1] + tie_prefix[thread]
                            for offset in T.serial(T.ceildiv(N, H)):
                                i = thread * T.ceildiv(N, H) + offset
                                if (  # noqa: SIM102 -- guard the score load before evaluating its key
                                    state[0] > 0
                                    and i < N
                                    and i >= starts[row // G]
                                    and i < ends[row // G]
                                ):
                                    if score_key(x[row // G, i, row % G]) == prefix[0]:
                                        if position < state[1] + state[0]:
                                            out[row, position] = i
                                        position = position + 1
                    with T.SimtVF(threads=H):
                        for thread in T.Parallel(H):
                            if thread == 0:
                                state[1] = state[1] + state[0]
                                state[0] = 0

    return main
