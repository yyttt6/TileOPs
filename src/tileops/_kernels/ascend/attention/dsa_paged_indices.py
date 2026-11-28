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


def metadata_builder(batch, seq, splits, topk, table_width, page_size, num_blocks):
    import tilelang.ascend.language as T

    rows = batch * seq * splits
    extent = max(128, 1 << (topk - 1).bit_length())
    output_width = ((topk + 127) // 128) * 128
    levels = (extent - 1).bit_length()
    row_tile = metadata_row_tile(rows, topk, table_width)
    if not row_tile:
        raise ValueError("Paged index metadata exceeds the UB budget")
    jobs = rows // row_tile
    cores = min(jobs, num_blocks)

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
            for task in T.Serial(T.ceildiv(jobs - bx, cores)):
                row = (task * cores + bx) * row_tile
                for r in T.Serial(row_tile):
                    request = (row + r) // (seq * splits)
                    limits[r] = Limits[request]
                    T.copy(Table[request, :], table_ub[r, :table_width])
                with T.SimtVF(threads=min(256, extent)):
                    for r, i in T.Parallel(row_tile, extent):
                        ids[r, i] = output_width
                        packed[r, i] = output_width
                T.copy(Input[row : row + row_tile, :], ids[:, :topk])
                with T.SimtVF(threads=min(256, extent)):
                    for r, i in T.Parallel(row_tile, extent):
                        limit = limits[r]
                        valid = (i < topk) & (ids[r, i] >= 0) & (ids[r, i] < limit) & (ids[r, i] < table_width * page_size)
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
                T.copy(packed[:, :output_width], Selected[row : row + row_tile, :])
                T.copy(length, Lengths[row : row + row_tile, :])

    return main


@functools.lru_cache(maxsize=64)
def _compile(batch, seq, splits, part, table_width, page_size, cores, device):
    import tilelang
    target = tilelang.tvm.target.Target("ascend")
    with target:
        program = metadata_builder(batch, seq, splits, part, table_width, page_size, cores)
        return tilelang.compile(program, target=target, out_idx=[])


def metadata(index, limits, table, page_size, splits, vector_cores):
    b, s, k = index.shape
    width = ((k + 127) // 128) * 128
    if width != k:
        index = torch.nn.functional.pad(index, (0, width - k), value=-1)
    part = width // splits
    rows = b * s * splits
    selected = index.new_empty((rows, part))
    counts = index.new_empty((rows, 8))
    kernel = _compile(b, s, splits, part, table.shape[1], page_size, vector_cores, index.device.index)
    kernel(index.reshape(rows, part).contiguous(), limits.contiguous(), table.contiguous(), selected, counts)
    return selected.reshape(rows, 1, 1, part), counts.reshape(rows, 1, 8)
