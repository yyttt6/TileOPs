"""Stable compaction into valid physical KV cache row numbers."""

import functools

import torch


def metadata_row_tile(rows, topk, table_width):
    # Five scan arrays, the table, one limit and eight padded counts per row.
    extent = max(128, 1 << (topk - 1).bit_length())
    bytes_per_row = 4 * (5 * extent + ((table_width + 7) // 8) * 8 + 9)
    # Keep headroom for compiler allocations in the vector UB.
    budget = 192 * 1024
    if rows >= 256 and topk <= 1024 and rows % 4 == 0 and 4 * bytes_per_row <= budget:
        return 4
    return 1 if bytes_per_row <= budget else 0


def metadata_builder(
    batch, seq, splits, topk, table_width, page_size, num_blocks, full_row_fast=False
):
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    rows = batch * seq * splits
    extent = max(128, 1 << (topk - 1).bit_length())
    output_width = ((topk + 127) // 128) * 128
    levels = (extent - 1).bit_length()
    row_tile = metadata_row_tile(rows, topk, table_width)
    if not row_tile:
        raise ValueError("Paged index metadata exceeds the UB budget")
    if full_row_fast and (row_tile != 1 or extent != topk or topk % 64 or page_size != 64):
        raise ValueError("full-row path requires one complete, aligned index row per AIV")
    jobs = rows // row_tile
    cores = min(jobs, num_blocks)

    @T.macro
    def compact_scan(ids, limits, table_ub, flags, scan, scratch, packed, length):
        with T.SimtVF(threads=min(256, extent)):
            for r, i in T.Parallel(row_tile, extent):
                limit = limits[r]
                valid = (
                    (i < topk)
                    & (ids[r, i] >= 0)
                    & (ids[r, i] < limit)
                    & (ids[r, i] < table_width * page_size)
                )
                flags[r, i] = T.if_then_else(valid, 1, 0)
                scan[r, i] = T.if_then_else(valid, 1, 0)
        # Alternate source and destination. Each phase reads a
        # complete, immutable previous phase.
        for pair in T.unroll(levels // 2):
            first = 1 << (2 * pair)
            second = first * 2
            with T.SimtVF(threads=min(256, extent)):
                for r, i in T.Parallel(row_tile, extent):
                    scratch[r, i] = scan[r, i] + T.if_then_else(
                        i >= first, scan[r, T.max(i - first, 0)], 0
                    )
            with T.SimtVF(threads=min(256, extent)):
                for r, i in T.Parallel(row_tile, extent):
                    scan[r, i] = scratch[r, i] + T.if_then_else(
                        i >= second, scratch[r, T.max(i - second, 0)], 0
                    )
        if levels % 2:
            distance = 1 << (levels - 1)
            with T.SimtVF(threads=min(256, extent)):
                for r, i in T.Parallel(row_tile, extent):
                    scratch[r, i] = scan[r, i] + T.if_then_else(
                        i >= distance, scan[r, T.max(i - distance, 0)], 0
                    )
        with T.SimtVF(threads=min(256, extent)):
            for r, i in T.Parallel(row_tile, extent):
                if flags[r, i] != 0:
                    index = ids[r, i]
                    rank = T.if_then_else(levels % 2 == 1, scratch[r, i], scan[r, i])
                    packed[r, T.max(rank - 1, 0)] = (
                        table_ub[r, index // page_size] * page_size + index % page_size
                    )
        # Dedicated VF avoids mixing the scatter's lane domain with
        # a second, smaller parallel loop. Each output row is 32-byte
        # aligned and carries eight identical count values.
        with T.SimtVF(threads=32):
            for r, column in T.Parallel(row_tile, 8):
                length[r, column] = T.if_then_else(
                    levels % 2 == 1, scratch[r, extent - 1], scan[r, extent - 1]
                )

    @T.prim_func
    def main(
        Input: T.Tensor((rows, topk), "int32"),
        Limits: T.Tensor((batch,), "int32"),
        Table: T.Tensor((batch, table_width), "int32"),
        Selected: T.Tensor((rows, output_width), "int32"),
        Lengths: T.Tensor((rows, 8), "int32"),
    ):
        with T.Kernel(cores) as bx:
            table_ub = T.alloc_shared((row_tile, ((table_width + 7) // 8) * 8), "int32")
            limits = T.alloc_shared((row_tile,), "int32")
            ids = T.alloc_shared((row_tile, extent), "int32")
            flags = T.alloc_shared((row_tile, extent), "int32")
            scan = T.alloc_shared((row_tile, extent), "int32")
            scratch = T.alloc_shared((row_tile, extent), "int32")
            packed = T.alloc_shared((row_tile, extent), "int32")
            length = T.alloc_shared((row_tile, 8), "int32")
            if full_row_fast:
                bounds = T.alloc_shared((8,), "int32")
            for task in T.Serial(T.ceildiv(jobs - bx, cores)):
                row = (task * cores + bx) * row_tile
                for r in T.Serial(row_tile):
                    request = (row + r) // (seq * splits)
                    limits[r] = Limits[request]
                    T.copy(Table[request, :], table_ub[r, :table_width])
                if not full_row_fast:
                    with T.SimtVF(threads=min(256, extent)):
                        for r, i in T.Parallel(row_tile, extent):
                            ids[r, i] = output_width
                            packed[r, i] = output_width
                T.copy(Input[row : row + row_tile, :], ids[:, :topk])
                if full_row_fast:
                    with T.SimdVF():
                        full = S.pset(32, "PAT_ALL")
                        first_lane = S.pset(32, "PAT_VL1")
                        lowest = T.alloc_var("int32x64")
                        highest = T.alloc_var("int32x64")
                        lowest = S.vld(ids[0, 0], "NORM")
                        highest = lowest
                        for block in range(1, topk // 64):
                            chunk = S.vld(ids[0, block * 64], "NORM")
                            lowest = S.vmin(lowest, chunk, full)
                            highest = S.vmax(highest, chunk, full)
                        S.vsts(bounds[0], S.vcmin(lowest, full), first_lane, "ONEPT_B32")
                        S.vsts(bounds[1], S.vcmax(highest, full), first_lane, "ONEPT_B32")
                    if bounds[0] >= 0 and bounds[1] < T.min(limits[0], table_width * page_size):
                        with T.SimdVF():
                            full = S.pset(32, "PAT_ALL")
                            low_bits = S.vdup(T.int32(63), "int32", full)
                            for block in range(topk // 64):
                                index = S.vld(ids[0, block * 64], "NORM")
                                page_id = T.reinterpret(
                                    S.vshrs(index, T.int32(6), full), "uint32x64"
                                )
                                page = S.vgather2(table_ub[0, 0], page_id, full)
                                offset = S.vand(index, low_bits, full)
                                physical = S.vadd(S.vshls(page, T.int32(6), full), offset, full)
                                S.vsts(packed[0, block * 64], physical, full, "NORM_B32")
                        with T.SimtVF(threads=32):
                            for lane in T.Parallel(8):
                                length[0, lane] = topk
                    else:
                        with T.SimtVF(threads=256):
                            for i in T.Parallel(topk):
                                packed[0, i] = output_width
                        compact_scan(ids, limits, table_ub, flags, scan, scratch, packed, length)
                else:
                    compact_scan(ids, limits, table_ub, flags, scan, scratch, packed, length)
                T.copy(packed[:, :output_width], Selected[row : row + row_tile, :])
                T.copy(length, Lengths[row : row + row_tile, :])

    return main


@functools.lru_cache(maxsize=64)
def _compile(batch, seq, splits, part, table_width, page_size, cores, device, full_row_fast=False):
    import tilelang

    target = tilelang.tvm.target.Target("ascend")
    with target:
        program = metadata_builder(
            batch, seq, splits, part, table_width, page_size, cores, full_row_fast
        )
        return tilelang.compile(program, target=target, out_idx=[])


def metadata(index, limits, table, page_size, splits, vector_cores, *, full_row_fast=False):
    b, s, k = index.shape
    width = ((k + 127) // 128) * 128
    if width != k:
        index = torch.nn.functional.pad(index, (0, width - k), value=-1)
    part = width // splits
    rows = b * s * splits
    selected = index.new_empty((rows, part))
    counts = index.new_empty((rows, 8))
    kernel = _compile(
        b,
        s,
        splits,
        part,
        table.shape[1],
        page_size,
        vector_cores,
        index.device.index,
        full_row_fast,
    )
    kernel(
        index.reshape(rows, part).contiguous(),
        limits.contiguous(),
        table.contiguous(),
        selected,
        counts,
    )
    return selected.reshape(rows, 1, 1, part), counts.reshape(rows, 1, 8)
