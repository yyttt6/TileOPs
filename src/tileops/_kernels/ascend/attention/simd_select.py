"""Resident SIMD radix selection for rows that fit in unified buffer."""

import tilelang
import tilelang.ascend.language as T
from tvm import tirx

SIMD_WORKSPACE_LIMIT = 96 * 1024


def simd_workspace_bytes(n, k, groups):
    np = (n + 63) // 64 * 64
    kp = (k + 63) // 64 * 64
    return 4 * (np + kp + (np * groups if groups > 1 else 0)) + 4096


@tilelang.jit(target="ascend", out_idx=[])
def topk_select_simd(R: int, N: int, K: int, G: int, W: int):
    assert 0 < K <= N < 65536 and G > 0 and 0 < W <= R
    assert simd_workspace_bytes(N, K, G) <= SIMD_WORKSPACE_LIMIT
    VF_TILES = 16
    NP = (N + 63) // 64 * 64
    KP = (K + 63) // 64 * 64

    @T.prim_func
    def main(
        bits: T.Tensor((R // G, N * G), "uint32"),
        starts: T.Tensor((R // G,), "int32"),
        ends: T.Tensor((R // G,), "int32"),
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
                    counts = T.alloc_shared((64,), "uint32")
                    prefix = T.alloc_var("uint32", init=0)
                    prefix_mask = T.alloc_var("uint32", init=0)
                    remaining = T.alloc_var("int32", init=K)
                    selected_bin = T.alloc_var("int32", init=0)
                    above_offset = T.alloc_var("int32", init=0)
                    equal_offset = T.alloc_var("int32", init=0)
                    begin = T.alloc_var("int32", init=starts[row // G])
                    end = T.alloc_var("int32", init=ends[row // G])
                    T.copy(ramp, lanes)
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
                        with T.SimdVF():
                            m = T.simd.pset(32)
                            for tile in T.serial(NP // 64):
                                T.simd.vsts(
                                    keys[tile * 64], T.simd.vdup(T.uint32(0), "uint32", m), m
                                )
                        if G == 1:
                            T.copy(bits[row, :N], keys[:N])
                        else:
                            packed = T.alloc_shared((NP * G,), "uint32")
                            with T.SimdVF():
                                m = T.simd.pset(32)
                                for tile in T.serial(NP * G // 64):
                                    T.simd.vsts(
                                        packed[tile * 64], T.simd.vdup(T.uint32(0), "uint32", m), m
                                    )
                            T.copy(bits[row // G, : N * G], packed[: N * G])
                            with T.SimdVF():
                                m = T.simd.pset(32)
                                for tile in T.serial(NP // 64):
                                    ids = T.simd.vadd(
                                        T.simd.vld(lanes[0]),
                                        T.simd.vdup(T.cast(tile * 64, "uint32"), "uint32", m),
                                        m,
                                    )
                                    offsets = T.simd.vadd(
                                        T.simd.vmul(ids, T.simd.vdup(T.uint32(G), "uint32", m), m),
                                        T.simd.vdup(T.cast(row % G, "uint32"), "uint32", m),
                                        m,
                                    )
                                    values = T.simd.vgather2(
                                        T.access_ptr(packed[0], "r", extent=NP * G), offsets, m
                                    )
                                    T.simd.vsts(keys[tile * 64], values, m)
                        with T.SimdVF():
                            m = T.simd.pset(32)
                            sign = T.simd.vdup(T.uint32(0x80000000), "uint32", m)
                            magnitude_mask = T.simd.vdup(T.uint32(0x7FFFFFFF), "uint32", m)
                            ones = T.simd.vdup(T.uint32(0xFFFFFFFF), "uint32", m)
                            for tile in T.serial(NP // 64):
                                b = T.simd.vld(keys[tile * 64])
                                magnitude = T.simd.vand(b, magnitude_mask, m)
                                neg = T.simd.vcmps(b, T.uint32(0x80000000), m, op="ge")
                                k = T.simd.vsel(
                                    T.simd.vxor(b, ones, m), T.simd.vxor(b, sign, m), neg
                                )
                                k = T.simd.vsel(
                                    sign, k, T.simd.vcmps(magnitude, T.uint32(0), m, op="eq")
                                )
                                k = T.simd.vsel(
                                    ones,
                                    k,
                                    T.simd.vcmps(magnitude, T.uint32(0x7F800000), m, op="gt"),
                                )
                                T.simd.vsts(keys[tile * 64], k, m)
                        for round in T.serial(4):
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
                                for tile in T.serial(NP // 64):
                                    k = T.simd.vld(keys[tile * 64])
                                    ids = T.simd.vadd(
                                        T.simd.vld(lanes[0]),
                                        T.simd.vdup(T.cast(tile * 64, "uint32"), "uint32", m32),
                                        m32,
                                    )
                                    visible = T.simd.vcmps(
                                        ids, T.cast(begin, "uint32"), m32, op="ge"
                                    )
                                    visible = T.simd.pand(
                                        visible,
                                        T.simd.vcmps(ids, T.cast(end, "uint32"), m32, op="lt"),
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
                            selected_bin = -1
                            bucket_cursor = T.alloc_var("int32", init=255)
                            while selected_bin < 0:
                                bucket_count = T.cast(
                                    hist[bucket_cursor // 128, bucket_cursor % 128], "int32"
                                )
                                if remaining > bucket_count:
                                    remaining = remaining - bucket_count
                                    bucket_cursor = bucket_cursor - 1
                                else:
                                    selected_bin = bucket_cursor
                            prefix = prefix | (T.cast(selected_bin, "uint32") << (24 - round * 8))
                            prefix_mask = prefix_mask | (T.uint32(255) << (24 - round * 8))
                        for group in T.serial(T.ceildiv(NP // 64, VF_TILES)):
                            with T.SimdVF():
                                m = T.simd.pset(32)
                                one_lane = T.simd.pset(32, "PAT_VL1")
                                one = T.simd.vdup(T.uint32(1), "uint32", m)
                                zero = T.simd.vdup(T.uint32(0), "uint32", m)
                                lane_ids = T.simd.vld(lanes[0])
                                gt_offset = T.simd.alloc_var("uint32")
                                eq_offset = T.simd.alloc_var("uint32")
                                gt_offset = T.simd.vdup(T.cast(above_offset, "uint32"), "uint32", m)
                                eq_offset = T.simd.vdup(T.cast(equal_offset, "uint32"), "uint32", m)
                                quota_total = T.simd.vdup(T.cast(remaining, "uint32"), "uint32", m)
                                eq_base = T.simd.vdup(T.cast(K - remaining, "uint32"), "uint32", m)
                                for inner in T.serial(VF_TILES):
                                    tile = group * VF_TILES + inner
                                    if tile < NP // 64:
                                        k = T.simd.vld(keys[tile * 64])
                                        ids = T.simd.vadd(
                                            lane_ids,
                                            T.simd.vdup(T.cast(tile * 64, "uint32"), "uint32", m),
                                            m,
                                        )
                                        visible = T.simd.vcmps(
                                            ids, T.cast(begin, "uint32"), m, op="ge"
                                        )
                                        visible = T.simd.pand(
                                            visible,
                                            T.simd.vcmps(ids, T.cast(end, "uint32"), m, op="lt"),
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
                                            g, T.access_ptr(result[0], "w", extent=K), gpos, gmask
                                        )
                                        T.simd.vscatter(
                                            e, T.access_ptr(result[0], "w", extent=K), epos, emask
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
                    T.copy(result[:K], out[row, :])

    return main
