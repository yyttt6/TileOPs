"""Bounded-memory streaming radix selection with early output."""

import tilelang
import tilelang.ascend.language as T
from tvm import tirx

STREAMING_WORKSPACE_LIMIT = 96 * 1024


def streaming_workspace_bytes(k):
    return 4 * (16384 + (k + 63) // 64 * 64) + 4096 + 2048 * 8


@tilelang.jit(target="ascend", out_idx=[])
def topk_select_streaming(R: int, N: int, K: int, W: int):
    assert 0 < K <= N < 2147483647 and 0 < W <= R
    NP = 16384
    assert streaming_workspace_bytes(K) <= STREAMING_WORKSPACE_LIMIT
    CAP = 2048
    KP = (K + 63) // 64 * 64
    VF_TILES = 16

    @T.prim_func
    def main(
        bits: T.Tensor((R, N), "uint32"),
        starts: T.Tensor((R,), "int32"),
        ends: T.Tensor((R,), "int32"),
        ramp: T.Tensor((64,), "uint32"),
        out: T.Tensor((R, K), "uint32"),
    ):
        with T.Kernel(W) as worker:
            for iteration in T.serial(T.ceildiv(R, W)):
                row = worker + iteration * W
                if row < R:
                    keys = T.alloc_shared((NP,), "uint32")
                    result = T.alloc_shared((KP,), "uint32")
                    lanes = T.alloc_shared((64,), "uint32")
                    hist = T.alloc_shared((2, 128), "uint16")
                    sums = T.alloc_shared((256,), "uint32")
                    candidates = T.alloc_shared((CAP,), "uint32")
                    candidate_ids = T.alloc_shared((CAP,), "uint32")
                    candidate_count = T.alloc_var("int32", init=0)
                    collected = T.alloc_var("int32", init=0)
                    counts = T.alloc_shared((64,), "uint32")
                    prefix = T.alloc_var("uint32", init=0)
                    prefix_mask = T.alloc_var("uint32", init=0)
                    remaining = T.alloc_var("int32", init=K)
                    selected_bin = T.alloc_var("int32", init=0)
                    above_offset = T.alloc_var("int32", init=0)
                    equal_offset = T.alloc_var("int32", init=0)
                    begin = T.alloc_var("int32", init=starts[row])
                    end = T.alloc_var("int32", init=ends[row])
                    T.copy(ramp, lanes)
                    # Short windows write indices and padding. Long windows write
                    # K - remaining above-threshold indices and remaining ties.
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        for tile in T.serial(KP // 64):
                            T.simd.vsts(result[tile * 64], T.simd.vdup(T.uint32(N), "uint32", m), m)
                    if end - begin <= K:
                        with T.SimdVF():
                            m = T.simd.pset(32)
                            for tile in T.serial(KP // 64):
                                ids = T.simd.vadd(
                                    T.simd.vld(lanes[0]),
                                    T.simd.vdup(T.cast(begin + tile * 64, "uint32"), "uint32", m),
                                    m,
                                )
                                valid = T.simd.vcmps(
                                    ids, T.cast(T.min(end, begin + K), "uint32"), m, op="lt"
                                )
                                T.simd.vsts(
                                    result[tile * 64],
                                    T.simd.vsel(ids, T.simd.vdup(T.uint32(N), "uint32", m), valid),
                                    m,
                                )
                    else:
                        for round in T.serial(4):
                            with T.SimdVF():
                                m = T.simd.pset(32)
                                for part in T.serial(4):
                                    T.simd.vsts(
                                        sums[part * 64], T.simd.vdup(T.uint32(0), "uint32", m), m
                                    )
                            if candidate_count > 0 and candidate_count <= CAP:
                                if collected == 0:
                                    for chunk in T.serial(T.ceildiv(N, NP)):
                                        T.copy(
                                            bits[
                                                row,
                                                chunk * NP : chunk * NP + T.min(NP, N - chunk * NP),
                                            ],
                                            keys[: T.min(NP, N - chunk * NP)],
                                        )
                                        with T.SimdVF():
                                            m = T.simd.pset(32)
                                            one_lane = T.simd.pset(32, "PAT_VL1")
                                            sign = T.simd.vdup(T.uint32(0x80000000), "uint32", m)
                                            magnitude_mask = T.simd.vdup(
                                                T.uint32(0x7FFFFFFF), "uint32", m
                                            )
                                            ones = T.simd.vdup(T.uint32(0xFFFFFFFF), "uint32", m)
                                            one = T.simd.vdup(T.uint32(1), "uint32", m)
                                            zero = T.simd.vdup(T.uint32(0), "uint32", m)
                                            high = T.simd.vdup(prefix_mask, "uint32", m)
                                            lane = T.simd.vld(lanes[0])
                                            early_offset = T.simd.alloc_var("uint32")
                                            early_offset = T.simd.vdup(
                                                T.cast(above_offset, "uint32"), "uint32", m
                                            )
                                            offset = T.simd.alloc_var("uint32")
                                            offset = T.simd.vdup(
                                                T.cast(collected, "uint32"), "uint32", m
                                            )
                                            for tile in T.serial(
                                                T.ceildiv(T.min(NP, N - chunk * NP), 64)
                                            ):
                                                ids = T.simd.vadd(
                                                    lane,
                                                    T.simd.vdup(
                                                        T.cast(chunk * NP + tile * 64, "uint32"),
                                                        "uint32",
                                                        m,
                                                    ),
                                                    m,
                                                )
                                                b = T.simd.vld(keys[tile * 64])
                                                magnitude = T.simd.vand(b, magnitude_mask, m)
                                                neg = T.simd.vcmps(
                                                    b, T.uint32(0x80000000), m, op="ge"
                                                )
                                                k = T.simd.vsel(
                                                    T.simd.vxor(b, ones, m),
                                                    T.simd.vxor(b, sign, m),
                                                    neg,
                                                )
                                                k = T.simd.vsel(
                                                    sign,
                                                    k,
                                                    T.simd.vcmps(
                                                        magnitude, T.uint32(0), m, op="eq"
                                                    ),
                                                )
                                                k = T.simd.vsel(
                                                    ones,
                                                    k,
                                                    T.simd.vcmps(
                                                        magnitude, T.uint32(0x7F800000), m, op="gt"
                                                    ),
                                                )
                                                visible = T.simd.pand(
                                                    T.simd.vcmps(
                                                        ids, T.cast(begin, "uint32"), m, op="ge"
                                                    ),
                                                    T.simd.vcmps(
                                                        ids, T.cast(end, "uint32"), m, op="lt"
                                                    ),
                                                    m,
                                                )
                                                matched = T.simd.vcmps(
                                                    T.simd.vand(k, high, m), prefix, m, op="eq"
                                                )
                                                keep = T.simd.pand(visible, matched, m)
                                                values = T.simd.vsqz(k, keep)
                                                count = T.simd.vdupv(
                                                    T.simd.vcadd(T.simd.vsel(one, zero, keep), m), m
                                                )
                                                valid = T.simd.vcmp(lane, count, m, op="lt")
                                                positions = T.simd.vsel(
                                                    T.simd.vadd(lane, offset, m), zero, valid
                                                )
                                                T.simd.vscatter(
                                                    values,
                                                    T.access_ptr(candidates[0], "w", extent=CAP),
                                                    positions,
                                                    valid,
                                                )
                                                T.simd.vscatter(
                                                    T.simd.vsqz(ids, keep),
                                                    T.access_ptr(candidate_ids[0], "w", extent=CAP),
                                                    positions,
                                                    valid,
                                                )
                                                higher = T.simd.vcmps(
                                                    T.simd.vand(k, high, m), prefix, m, op="gt"
                                                )
                                                take = T.simd.pand(visible, higher, m)
                                                early_count = T.simd.vdupv(
                                                    T.simd.vcadd(T.simd.vsel(one, zero, take), m), m
                                                )
                                                early_mask = T.simd.vcmp(
                                                    lane, early_count, m, op="lt"
                                                )
                                                early_pos = T.simd.vsel(
                                                    T.simd.vadd(lane, early_offset, m),
                                                    zero,
                                                    early_mask,
                                                )
                                                T.simd.vscatter(
                                                    T.simd.vsqz(ids, take),
                                                    T.access_ptr(result[0], "w", extent=K),
                                                    early_pos,
                                                    early_mask,
                                                )
                                                early_offset = T.simd.vadd(
                                                    early_offset, early_count, m
                                                )
                                                offset = T.simd.vadd(offset, count, m)
                                            T.simd.vsts(
                                                counts[0],
                                                offset,
                                                one_lane,
                                                dist="ONEPT_B32",
                                                extent=1,
                                            )
                                            T.simd.vsts(
                                                counts[1],
                                                early_offset,
                                                one_lane,
                                                dist="ONEPT_B32",
                                                extent=1,
                                            )
                                        above_offset = T.cast(counts[1], "int32")
                                        collected = T.cast(counts[0], "int32")
                                with T.SimdVF():
                                    m32 = T.simd.pset(32)
                                    m16 = T.simd.pset(16)
                                    m8 = T.simd.pset(8)
                                    f0 = T.simd.alloc_var("uint16")
                                    f1 = T.simd.alloc_var("uint16")
                                    f0 = T.simd.vdup(T.uint16(0), "uint16", m16)
                                    f1 = T.simd.vdup(T.uint16(0), "uint16", m16)
                                    ones32 = T.simd.vdup(T.uint32(1), "uint32", m32)
                                    zero32 = T.simd.vdup(T.uint32(0), "uint32", m32)
                                    zero8 = T.simd.vdup(T.uint8(0), "uint8", m8)
                                    byte_mask = T.simd.vdup(T.uint32(255), "uint32", m32)
                                    high_mask = T.simd.vdup(prefix_mask, "uint32", m32)
                                    for tile in T.serial(T.ceildiv(candidate_count, 64)):
                                        ids = T.simd.vadd(
                                            T.simd.vld(lanes[0]),
                                            T.simd.vdup(T.cast(tile * 64, "uint32"), "uint32", m32),
                                            m32,
                                        )
                                        visible = T.simd.vcmps(
                                            ids, T.cast(candidate_count, "uint32"), m32, op="lt"
                                        )
                                        k = T.simd.vld(candidates[tile * 64])
                                        matched = T.simd.vcmps(
                                            T.simd.vand(k, high_mask, m32), prefix, m32, op="eq"
                                        )
                                        active32 = T.simd.pand(visible, matched, m32)
                                        digit32 = T.simd.vand(
                                            T.simd.vshrs(k, 24 - round * 8, m32), byte_mask, m32
                                        )
                                        active_values = T.simd.vsel(ones32, zero32, active32)
                                        # Same narrowing layout for digit and 0/1 validity. Only
                                        # nonzero validity lanes participate, regardless of part placement.
                                        digits = T.simd.vcvt(digit32, "uint8", m8, part=0)
                                        tags = T.simd.vcvt(active_values, "uint8", m8, part=0)
                                        active8 = T.simd.vcmp(tags, zero8, m8, op="ne")
                                        T.simd.dhistv2(f0, digits, active8, bin=0)
                                        T.simd.dhistv2(f1, digits, active8, bin=1)
                                    T.simd.vsts(hist[0, 0], f0, m16, dist="NORM_B16")
                                    T.simd.vsts(hist[1, 0], f1, m16, dist="NORM_B16")
                                with T.SimdVF():
                                    m16 = T.simd.pset(16)
                                    m32 = T.simd.pset(32)
                                    for bank in T.serial(2):
                                        v = T.simd.vld(hist[bank, 0])
                                        even = T.simd.vcvt(v, "uint32", m16, part=0)
                                        odd = T.simd.vcvt(v, "uint32", m16, part=1)
                                        T.simd.vsts(
                                            sums[bank * 128],
                                            T.simd.vadd(T.simd.vld(sums[bank * 128]), even, m32),
                                            m32,
                                        )
                                        T.simd.vsts(
                                            sums[bank * 128 + 64],
                                            T.simd.vadd(
                                                T.simd.vld(sums[bank * 128 + 64]), odd, m32
                                            ),
                                            m32,
                                        )
                            else:
                                for chunk in T.serial(T.ceildiv(N, NP)):
                                    # Full contiguous copies overwrite every lane in the staging buffer.
                                    if N % NP != 0:
                                        with T.SimdVF():
                                            m = T.simd.pset(32)
                                            for tile in T.serial(NP // 64):
                                                T.simd.vsts(
                                                    keys[tile * 64],
                                                    T.simd.vdup(T.uint32(0), "uint32", m),
                                                    m,
                                                )
                                    T.copy(
                                        bits[
                                            row, chunk * NP : chunk * NP + T.min(NP, N - chunk * NP)
                                        ],
                                        keys[: T.min(NP, N - chunk * NP)],
                                    )
                                    with T.SimdVF():
                                        m32 = T.simd.pset(32)
                                        m16 = T.simd.pset(16)
                                        m8 = T.simd.pset(8)
                                        f0 = T.simd.alloc_var("uint16")
                                        f1 = T.simd.alloc_var("uint16")
                                        f0 = T.simd.vdup(T.uint16(0), "uint16", m16)
                                        f1 = T.simd.vdup(T.uint16(0), "uint16", m16)
                                        ones32 = T.simd.vdup(T.uint32(1), "uint32", m32)
                                        zero32 = T.simd.vdup(T.uint32(0), "uint32", m32)
                                        zero8 = T.simd.vdup(T.uint8(0), "uint8", m8)
                                        byte_mask = T.simd.vdup(T.uint32(255), "uint32", m32)
                                        high_mask = T.simd.vdup(prefix_mask, "uint32", m32)
                                        fused_sign = T.simd.vdup(
                                            T.uint32(0x80000000), "uint32", m32
                                        )
                                        fused_magnitude_mask = T.simd.vdup(
                                            T.uint32(0x7FFFFFFF), "uint32", m32
                                        )
                                        fused_ones = T.simd.vdup(
                                            T.uint32(0xFFFFFFFF), "uint32", m32
                                        )
                                        for tile in T.serial(NP // 64):
                                            k = T.simd.vld(keys[tile * 64])
                                            magnitude = T.simd.vand(k, fused_magnitude_mask, m32)
                                            neg = T.simd.vcmps(
                                                k, T.uint32(0x80000000), m32, op="ge"
                                            )
                                            k = T.simd.vsel(
                                                T.simd.vxor(k, fused_ones, m32),
                                                T.simd.vxor(k, fused_sign, m32),
                                                neg,
                                            )
                                            k = T.simd.vsel(
                                                fused_sign,
                                                k,
                                                T.simd.vcmps(magnitude, T.uint32(0), m32, op="eq"),
                                            )
                                            k = T.simd.vsel(
                                                fused_ones,
                                                k,
                                                T.simd.vcmps(
                                                    magnitude, T.uint32(0x7F800000), m32, op="gt"
                                                ),
                                            )
                                            ids = T.simd.vadd(
                                                T.simd.vld(lanes[0]),
                                                T.simd.vdup(
                                                    T.cast(chunk * NP + tile * 64, "uint32"),
                                                    "uint32",
                                                    m32,
                                                ),
                                                m32,
                                            )
                                            visible = T.simd.vcmps(
                                                ids, T.cast(begin, "uint32"), m32, op="ge"
                                            )
                                            visible = T.simd.pand(
                                                visible,
                                                T.simd.vcmps(
                                                    ids, T.cast(end, "uint32"), m32, op="lt"
                                                ),
                                                m32,
                                            )
                                            matched = T.simd.vcmps(
                                                T.simd.vand(k, high_mask, m32), prefix, m32, op="eq"
                                            )
                                            active32 = T.simd.pand(visible, matched, m32)
                                            digit32 = T.simd.vand(
                                                T.simd.vshrs(k, 24 - round * 8, m32), byte_mask, m32
                                            )
                                            active_values = T.simd.vsel(ones32, zero32, active32)
                                            # Same narrowing layout for digit and 0/1 validity. Only
                                            # nonzero validity lanes participate, regardless of part placement.
                                            digits = T.simd.vcvt(digit32, "uint8", m8, part=0)
                                            tags = T.simd.vcvt(active_values, "uint8", m8, part=0)
                                            active8 = T.simd.vcmp(tags, zero8, m8, op="ne")
                                            T.simd.dhistv2(f0, digits, active8, bin=0)
                                            T.simd.dhistv2(f1, digits, active8, bin=1)
                                        T.simd.vsts(hist[0, 0], f0, m16, dist="NORM_B16")
                                        T.simd.vsts(hist[1, 0], f1, m16, dist="NORM_B16")
                                    with T.SimdVF():
                                        m16 = T.simd.pset(16)
                                        m32 = T.simd.pset(32)
                                        for bank in T.serial(2):
                                            v = T.simd.vld(hist[bank, 0])
                                            even = T.simd.vcvt(v, "uint32", m16, part=0)
                                            odd = T.simd.vcvt(v, "uint32", m16, part=1)
                                            T.simd.vsts(
                                                sums[bank * 128],
                                                T.simd.vadd(
                                                    T.simd.vld(sums[bank * 128]), even, m32
                                                ),
                                                m32,
                                            )
                                            T.simd.vsts(
                                                sums[bank * 128 + 64],
                                                T.simd.vadd(
                                                    T.simd.vld(sums[bank * 128 + 64]), odd, m32
                                                ),
                                                m32,
                                            )
                            selected_bin = -1
                            bucket_cursor = T.alloc_var("int32", init=255)
                            while selected_bin < 0:
                                position = (
                                    (bucket_cursor // 128) * 128
                                    + (bucket_cursor % 2) * 64
                                    + (bucket_cursor % 128) // 2
                                )
                                bucket_count = T.cast(sums[position], "int32")
                                if remaining > bucket_count:
                                    remaining = remaining - bucket_count
                                    bucket_cursor = bucket_cursor - 1
                                else:
                                    selected_bin = bucket_cursor
                            prefix = prefix | (T.cast(selected_bin, "uint32") << (24 - round * 8))
                            prefix_mask = prefix_mask | (T.uint32(255) << (24 - round * 8))
                            if round == 0 or (round == 1 and candidate_count > CAP):
                                position = (
                                    (selected_bin // 128) * 128
                                    + (selected_bin % 2) * 64
                                    + (selected_bin % 128) // 2
                                )
                                candidate_count = T.cast(sums[position], "int32")
                        if collected > 0:
                            for group in T.serial(T.ceildiv(T.ceildiv(collected, 64), VF_TILES)):
                                with T.SimdVF():
                                    m = T.simd.pset(32)
                                    one_lane = T.simd.pset(32, "PAT_VL1")
                                    one = T.simd.vdup(T.uint32(1), "uint32", m)
                                    zero = T.simd.vdup(T.uint32(0), "uint32", m)
                                    lane_ids = T.simd.vld(lanes[0])
                                    gt_offset = T.simd.alloc_var("uint32")
                                    eq_offset = T.simd.alloc_var("uint32")
                                    gt_offset = T.simd.vdup(
                                        T.cast(above_offset, "uint32"), "uint32", m
                                    )
                                    eq_offset = T.simd.vdup(
                                        T.cast(equal_offset, "uint32"), "uint32", m
                                    )
                                    quota_total = T.simd.vdup(
                                        T.cast(remaining, "uint32"), "uint32", m
                                    )
                                    eq_base = T.simd.vdup(
                                        T.cast(K - remaining, "uint32"), "uint32", m
                                    )
                                    for inner in T.serial(VF_TILES):
                                        tile = group * VF_TILES + inner
                                        if tile < T.ceildiv(collected, 64):
                                            k = T.simd.vld(candidates[tile * 64])
                                            ids = T.simd.vld(candidate_ids[tile * 64])
                                            offsets = T.simd.vadd(
                                                lane_ids,
                                                T.simd.vdup(
                                                    T.cast(tile * 64, "uint32"), "uint32", m
                                                ),
                                                m,
                                            )
                                            visible = T.simd.vcmps(
                                                offsets, T.cast(collected, "uint32"), m, op="lt"
                                            )
                                            gt = T.simd.pand(
                                                visible, T.simd.vcmps(k, prefix, m, op="gt"), m
                                            )
                                            eq = T.simd.pand(
                                                visible, T.simd.vcmps(k, prefix, m, op="eq"), m
                                            )
                                            rank = T.simd.vusqz(eq, "int32")
                                            rank_unsigned = tirx.reinterpret("uint32x64", rank)
                                            quota = T.simd.vsub(quota_total, eq_offset, m)
                                            eq = T.simd.pand(
                                                eq, T.simd.vcmp(rank_unsigned, quota, m, op="lt"), m
                                            )
                                            g = T.simd.vsqz(ids, gt)
                                            e = T.simd.vsqz(ids, eq)
                                            gc = T.simd.vdupv(
                                                T.simd.vcadd(T.simd.vsel(one, zero, gt), m), m
                                            )
                                            ec = T.simd.vdupv(
                                                T.simd.vcadd(T.simd.vsel(one, zero, eq), m), m
                                            )
                                            gmask = T.simd.vcmp(lane_ids, gc, m, op="lt")
                                            emask = T.simd.vcmp(lane_ids, ec, m, op="lt")
                                            gpos = T.simd.vadd(lane_ids, gt_offset, m)
                                            epos = T.simd.vadd(
                                                lane_ids, T.simd.vadd(eq_base, eq_offset, m), m
                                            )
                                            gpos = T.simd.vsel(gpos, zero, gmask)
                                            epos = T.simd.vsel(epos, zero, emask)
                                            T.simd.vscatter(
                                                g,
                                                T.access_ptr(result[0], "w", extent=K),
                                                gpos,
                                                gmask,
                                            )
                                            T.simd.vscatter(
                                                e,
                                                T.access_ptr(result[0], "w", extent=K),
                                                epos,
                                                emask,
                                            )
                                            gt_offset = T.simd.vadd(gt_offset, gc, m)
                                            eq_offset = T.simd.vadd(eq_offset, ec, m)
                                    T.simd.vsts(
                                        counts[0], gt_offset, one_lane, dist="ONEPT_B32", extent=1
                                    )
                                    T.simd.vsts(
                                        counts[1], eq_offset, one_lane, dist="ONEPT_B32", extent=1
                                    )
                                above_offset = T.cast(counts[0], "int32")
                                equal_offset = T.cast(counts[1], "int32")
                        else:
                            for chunk in T.serial(T.ceildiv(N, NP)):
                                # Full contiguous copies overwrite every lane in the staging buffer.
                                if N % NP != 0:
                                    with T.SimdVF():
                                        m = T.simd.pset(32)
                                        for tile in T.serial(NP // 64):
                                            T.simd.vsts(
                                                keys[tile * 64],
                                                T.simd.vdup(T.uint32(0), "uint32", m),
                                                m,
                                            )
                                T.copy(
                                    bits[row, chunk * NP : chunk * NP + T.min(NP, N - chunk * NP)],
                                    keys[: T.min(NP, N - chunk * NP)],
                                )
                                for group in T.serial(T.ceildiv(NP // 64, VF_TILES)):
                                    with T.SimdVF():
                                        m = T.simd.pset(32)
                                        one_lane = T.simd.pset(32, "PAT_VL1")
                                        one = T.simd.vdup(T.uint32(1), "uint32", m)
                                        zero = T.simd.vdup(T.uint32(0), "uint32", m)
                                        lane_ids = T.simd.vld(lanes[0])
                                        gt_offset = T.simd.alloc_var("uint32")
                                        eq_offset = T.simd.alloc_var("uint32")
                                        gt_offset = T.simd.vdup(
                                            T.cast(above_offset, "uint32"), "uint32", m
                                        )
                                        eq_offset = T.simd.vdup(
                                            T.cast(equal_offset, "uint32"), "uint32", m
                                        )
                                        quota_total = T.simd.vdup(
                                            T.cast(remaining, "uint32"), "uint32", m
                                        )
                                        eq_base = T.simd.vdup(
                                            T.cast(K - remaining, "uint32"), "uint32", m
                                        )
                                        fused_sign = T.simd.vdup(T.uint32(0x80000000), "uint32", m)
                                        fused_magnitude_mask = T.simd.vdup(
                                            T.uint32(0x7FFFFFFF), "uint32", m
                                        )
                                        fused_ones = T.simd.vdup(T.uint32(0xFFFFFFFF), "uint32", m)
                                        for inner in T.serial(VF_TILES):
                                            tile = group * VF_TILES + inner
                                            if tile < NP // 64:
                                                k = T.simd.vld(keys[tile * 64])
                                                magnitude = T.simd.vand(k, fused_magnitude_mask, m)
                                                neg = T.simd.vcmps(
                                                    k, T.uint32(0x80000000), m, op="ge"
                                                )
                                                k = T.simd.vsel(
                                                    T.simd.vxor(k, fused_ones, m),
                                                    T.simd.vxor(k, fused_sign, m),
                                                    neg,
                                                )
                                                k = T.simd.vsel(
                                                    fused_sign,
                                                    k,
                                                    T.simd.vcmps(
                                                        magnitude, T.uint32(0), m, op="eq"
                                                    ),
                                                )
                                                k = T.simd.vsel(
                                                    fused_ones,
                                                    k,
                                                    T.simd.vcmps(
                                                        magnitude, T.uint32(0x7F800000), m, op="gt"
                                                    ),
                                                )
                                                ids = T.simd.vadd(
                                                    lane_ids,
                                                    T.simd.vdup(
                                                        T.cast(chunk * NP + tile * 64, "uint32"),
                                                        "uint32",
                                                        m,
                                                    ),
                                                    m,
                                                )
                                                visible = T.simd.vcmps(
                                                    ids, T.cast(begin, "uint32"), m, op="ge"
                                                )
                                                visible = T.simd.pand(
                                                    visible,
                                                    T.simd.vcmps(
                                                        ids, T.cast(end, "uint32"), m, op="lt"
                                                    ),
                                                    m,
                                                )
                                                gt = T.simd.pand(
                                                    visible, T.simd.vcmps(k, prefix, m, op="gt"), m
                                                )
                                                eq = T.simd.pand(
                                                    visible, T.simd.vcmps(k, prefix, m, op="eq"), m
                                                )
                                                rank = T.simd.vusqz(eq, "int32")
                                                rank_unsigned = tirx.reinterpret("uint32x64", rank)
                                                quota = T.simd.vsub(quota_total, eq_offset, m)
                                                eq = T.simd.pand(
                                                    eq,
                                                    T.simd.vcmp(rank_unsigned, quota, m, op="lt"),
                                                    m,
                                                )
                                                g = T.simd.vsqz(ids, gt)
                                                e = T.simd.vsqz(ids, eq)
                                                gc = T.simd.vdupv(
                                                    T.simd.vcadd(T.simd.vsel(one, zero, gt), m), m
                                                )
                                                ec = T.simd.vdupv(
                                                    T.simd.vcadd(T.simd.vsel(one, zero, eq), m), m
                                                )
                                                gmask = T.simd.vcmp(lane_ids, gc, m, op="lt")
                                                emask = T.simd.vcmp(lane_ids, ec, m, op="lt")
                                                gpos = T.simd.vadd(lane_ids, gt_offset, m)
                                                epos = T.simd.vadd(
                                                    lane_ids, T.simd.vadd(eq_base, eq_offset, m), m
                                                )
                                                gpos = T.simd.vsel(gpos, zero, gmask)
                                                epos = T.simd.vsel(epos, zero, emask)
                                                T.simd.vscatter(
                                                    g,
                                                    T.access_ptr(result[0], "w", extent=K),
                                                    gpos,
                                                    gmask,
                                                )
                                                T.simd.vscatter(
                                                    e,
                                                    T.access_ptr(result[0], "w", extent=K),
                                                    epos,
                                                    emask,
                                                )
                                                gt_offset = T.simd.vadd(gt_offset, gc, m)
                                                eq_offset = T.simd.vadd(eq_offset, ec, m)
                                        T.simd.vsts(
                                            counts[0],
                                            gt_offset,
                                            one_lane,
                                            dist="ONEPT_B32",
                                            extent=1,
                                        )
                                        T.simd.vsts(
                                            counts[1],
                                            eq_offset,
                                            one_lane,
                                            dist="ONEPT_B32",
                                            extent=1,
                                        )
                                    above_offset = T.cast(counts[0], "int32")
                                    equal_offset = T.cast(counts[1], "int32")
                    T.copy(result[:K], out[row, :])

    return main
