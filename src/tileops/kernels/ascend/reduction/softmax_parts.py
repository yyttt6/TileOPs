"""Contiguous single-row tiles for large softmax partial statistics."""

import functools

import tilelang
import tilelang.ascend.language as T

from tileops.kernels.ascend.reduction.logsumexp import _native_stats


@functools.lru_cache(maxsize=256)
def build(m, n, dtype, outer, inner):
    extent = inner if outer else n
    width = max(
        32,
        min(
            4096 if m <= 64 else 16384 if dtype == "float32" else 32768,
            tilelang.next_power_of_2(extent),
        ),
    )
    parts = (extent + width - 1) // width
    chunks = parts * (outer if outer else 1)
    rows = 1
    pitch = min(width, (extent + 15) // 16 * 16)
    native = extent % 128 == 0 and m % rows == 0
    tasks = ((m + rows - 1) // rows) * chunks
    blocks = min(64, tasks)
    shape = (outer, m, inner) if outer else (m, n)

    @T.prim_func
    def main(X: T.Tensor(shape, dtype), Y: T.Tensor((m, chunks, 8), "float32")):
        with T.Kernel(blocks) as bx:
            x = T.alloc_shared((rows, pitch), dtype)
            y = T.alloc_shared((rows, 8), "float32")
            T.annotate_buffer_versions({x: 2, y: 2})
            for step in T.Pipelined((tasks + blocks - 1) // blocks, num_stages=2):
                task = step * blocks + bx
                if task < tasks:
                    row = (task // chunks) * rows
                    chunk = task % chunks
                    begin = (chunk % parts) * width
                    valid_rows = T.min(rows, m - row)
                    count = T.min(width, extent - begin)
                    if outer:
                        T.copy(
                            X[chunk // parts, row : row + valid_rows, begin : begin + count],
                            x[:valid_rows, :count],
                            l2_cache_ctrl=4,
                        )
                    else:
                        T.copy(
                            X[row : row + valid_rows, begin : begin + count],
                            x[:valid_rows, :count],
                            l2_cache_ctrl=4,
                        )
                    if native:
                        _native_stats(x, y, rows, count, dtype)
                    else:
                        with T.SimtVF(threads=min(256, width)):
                            maximum = T.alloc_reducer((rows,), "float32", op="max")
                            T.reducer_init(maximum)
                            for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                if r < valid_rows and i < count:
                                    T.reducer_update(maximum[r], T.cast(x[r, i], "float32"))
                            mx = T.alloc_fragment((rows,), "float32")
                            T.finalize_reducer(maximum, mx)
                            total = T.alloc_reducer((rows,), "float32", op="sum")
                            T.reducer_init(total)
                            for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                if r < valid_rows and i < count:
                                    value = T.cast(x[r, i], "float32")
                                    shift = T.if_then_else(
                                        T.abs(mx[r]) == T.infinity("float32"), T.float32(0), mx[r]
                                    )
                                    T.reducer_update(total[r], T.exp(value - shift))
                            sm = T.alloc_fragment((rows,), "float32")
                            T.finalize_reducer(total, sm)
                            for r in T.Parallel(rows):
                                y[r, 0] = mx[r]
                                y[r, 1] = sm[r]
                    T.copy(
                        y[:valid_rows, :2], Y[row : row + valid_rows, chunk, :2], l2_cache_ctrl=0
                    )

    return tilelang.compile(main, target="ascend", out_idx=[1]), chunks
