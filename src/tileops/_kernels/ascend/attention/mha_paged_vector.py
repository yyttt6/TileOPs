"""Split-KV paged attention on Ascend vector cores."""

import functools
import math

import tilelang
import tilelang.ascend.language as T


@functools.lru_cache(maxsize=64)
def paged_programs(call, cores):
    b, qlen, heads, dim = call.batch, call.max_seqlen_q, call.heads, call.dim
    page, width, pool = call.page_size, call.max_pages_per_req, call.seqlen_kv
    dtype = str(call.dtype).removeprefix("torch.")
    tile = math.gcd(page, 64)
    splits = width * page // tile
    rows = b * qlen * heads
    tasks = rows * splits
    blocks = min(cores, tasks)
    # One warp handles four independent 8-lane dot products.
    lanes = 8
    threads = tile * lanes
    groups = 8
    stride = dim + 8
    scale = dim**-0.5
    empty_max = -3.402823466e38

    @T.prim_func
    def paged_partials(
        Q: T.Tensor((b, qlen, heads, dim), dtype),
        K: T.Tensor((pool, heads, dim), dtype),
        V: T.Tensor((pool, heads, dim), dtype),
        Length: T.Tensor((b,), "int32"),
        Table: T.Tensor((b, width), "int32"),
        Part: T.Tensor((rows, splits, stride), "float32"),
    ):
        with T.Kernel(blocks) as block:
            kbuf = T.alloc_shared((tile, dim), dtype)
            vbuf = T.alloc_shared((tile, dim), dtype)
            query = T.alloc_shared((dim,), dtype)
            scores = T.alloc_shared((tile,), "float32")
            sums = T.alloc_shared((groups, dim), "float32")
            for wave in T.serial((tasks + blocks - 1 - block) // blocks):
                task = block + wave * blocks
                split, row = task % splits, task // splits
                head, seq, batch = row % heads, row // heads % qlen, row // (heads * qlen)
                live = Length[batch]
                end = live - qlen + seq + 1 if call.is_causal else live
                start = split * tile
                if start < end:
                    phys = Table[batch, start // page] * page + start % page
                    T.copy(K[phys : phys + tile, head, :], kbuf)
                    T.copy(V[phys : phys + tile, head, :], vbuf)
                    T.copy(Q[batch, seq, head, :], query)
                    with T.SimtVF(threads=threads):
                        t = T.get_thread_binding()
                        key, lane = t // lanes, t % lanes
                        dot = T.alloc_var("float32", init=0)
                        for j in T.unroll(dim // lanes):
                            col = j * lanes + lane
                            dot += T.cast(query[col], "float32") * T.cast(kbuf[key, col], "float32")
                        for level in T.unroll(3):
                            dot += T.call_pure_extern(
                                "float32", "asc_shfl_xor", dot, 1 << level
                            )  # lint: extern-target allow: Ascend SIMT lane reduction.
                        if lane == 0:
                            scores[key] = T.if_then_else(
                                start + key < end, dot * scale, T.float32(empty_max)
                            )
                    with T.SimtVF(threads=32):
                        lane = T.get_thread_binding()
                        mx = T.alloc_var("float32", init=empty_max)
                        for j in T.unroll((tile + 31) // 32):
                            if lane + j * 32 < tile:
                                mx = T.max(mx, scores[lane + j * 32])
                        mx = T.warp_reduce_max(mx)
                        den = T.alloc_var("float32", init=0)
                        for j in T.unroll((tile + 31) // 32):
                            if lane + j * 32 < tile:
                                w = T.exp(scores[lane + j * 32] - mx)
                                scores[lane + j * 32] = w
                                den += w
                        den = T.warp_reduce_sum(den)
                        if lane == 0:
                            Part[row, split, dim] = mx
                            Part[row, split, dim + 1] = den
                    with T.SimtVF(threads=groups * dim):
                        t = T.get_thread_binding()
                        group, col = t // dim, t % dim
                        val = T.alloc_var("float32", init=0)
                        for j in T.unroll(tile // groups):
                            key = group + j * groups
                            if start + key < end:
                                val += scores[key] * T.cast(vbuf[key, col], "float32")
                        sums[group, col] = val
                    with T.SimtVF(threads=dim):
                        col = T.get_thread_binding()
                        val = T.alloc_var("float32", init=0)
                        for group in T.unroll(groups):
                            val += sums[group, col]
                        Part[row, split, col] = val
                else:
                    with T.SimtVF(threads=dim):
                        col = T.get_thread_binding()
                        Part[row, split, col] = T.float32(0)
                        if col == 0:
                            Part[row, split, dim] = T.float32(empty_max)
                            Part[row, split, dim + 1] = T.float32(0)

    @T.prim_func
    def paged_merge(
        Part: T.Tensor((rows, splits, stride), "float32"),
        Out: T.Tensor((b, qlen, heads, dim), dtype),
    ):
        with T.Kernel(min(cores, rows)) as block:
            buf = T.alloc_shared((splits, stride), "float32")
            for row in T.serial(block, rows, min(cores, rows)):
                T.copy(Part[row, :, :], buf)
                with T.SimtVF(threads=dim):
                    col = T.get_thread_binding()
                    mx = T.alloc_var("float32", init=empty_max)
                    for i in T.serial(splits):
                        mx = T.max(mx, buf[i, dim])
                    den = T.alloc_var("float32", init=0)
                    val = T.alloc_var("float32", init=0)
                    for i in T.serial(splits):
                        w = T.exp(buf[i, dim] - mx)
                        val += w * buf[i, col]
                        den += w * buf[i, dim + 1]
                    Out[row // (heads * qlen), row // heads % qlen, row % heads, col] = T.cast(
                        val / T.max(den, T.float32(1e-30)), dtype
                    )

    return (
        tilelang.compile(paged_partials, target="ascend", out_idx=[5]),
        tilelang.compile(paged_merge, target="ascend", out_idx=[1]),
    )
