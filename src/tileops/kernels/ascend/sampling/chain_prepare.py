"""Parallel Philox acceptance tests with sparse probability gathers."""

import functools

import tilelang
import tilelang.ascend.language as T


@functools.lru_cache(maxsize=256)
def _prepare_parallel(batch, vocab, drafts):
    width = max(64, tilelang.next_power_of_2(drafts + 1))
    blocks = min(batch, 64)

    @T.prim_func
    def main(
        D: T.Tensor((batch, drafts, vocab), "float32"),
        I: T.Tensor((batch, drafts), "int32"),
        Q: T.Tensor((batch, drafts + 1, vocab), "float32"),
        S: T.Tensor((1,), "int64"),
        O: T.Tensor((1,), "int64"),
        A: T.Tensor((batch,), "int32"),
        U: T.Tensor((batch,), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            seed = T.alloc_shared((1,), "int64")
            offset = T.alloc_shared((1,), "int64")
            ids = T.alloc_shared((max(8, drafts),), "int32")
            accepted = T.alloc_shared((8,), "int32")
            uniform = T.alloc_shared((8,), "float32")
            T.copy(S, seed)
            T.copy(O, offset)
            for step in T.serial(T.ceildiv(batch, blocks)):
                row = step * blocks + bx
                if row < batch:
                    T.copy(I[row, :], ids[:drafts])
                    with T.SimtVF(threads=256):
                        c0 = T.alloc_fragment((width,), "uint32")
                        c1 = T.alloc_fragment((width,), "uint32")
                        c2 = T.alloc_fragment((width,), "uint32")
                        c3 = T.alloc_fragment((width,), "uint32")
                        k0 = T.alloc_fragment((width,), "uint32")
                        k1 = T.alloc_fragment((width,), "uint32")
                        for j in T.Parallel(width):
                            c0[j] = T.cast(j, "uint32")
                            c1[j] = T.cast(row, "uint32")
                            c2[j] = T.cast(offset[0], "uint32")
                            c3[j] = T.cast(T.cast(offset[0], "uint64") >> 32, "uint32")
                            k0[j] = T.cast(seed[0], "uint32")
                            k1[j] = T.cast(T.cast(seed[0], "uint64") >> 32, "uint32")
                        for _round in T.serial(10):
                            for j in T.Parallel(width):
                                a = T.cast(c0[j], "uint64") * T.uint64(0xD2511F53)
                                b = T.cast(c2[j], "uint64") * T.uint64(0xCD9E8D57)
                                t0 = T.cast(b >> 32, "uint32") ^ c1[j] ^ k0[j]
                                t2 = T.cast(a >> 32, "uint32") ^ c3[j] ^ k1[j]
                                c0[j] = t0
                                c1[j] = T.cast(b, "uint32")
                                c2[j] = t2
                                c3[j] = T.cast(a, "uint32")
                                k0[j] = k0[j] + T.uint32(0x9E3779B9)
                                k1[j] = k1[j] + T.uint32(0xBB67AE85)
                        first = T.alloc_reducer((1,), "int32", op="min")
                        T.reducer_init(first)
                        for j in T.Parallel(width):
                            u = T.cast(c0[j] >> 8, "float32") * T.float32(2.0**-24)
                            if j < drafts:
                                token = ids[j]
                                if not (u * D[row, j, token] < Q[row, j, token]):
                                    T.reducer_update(first[0], j)
                            if j == drafts:
                                uniform[0] = u
                        count = T.alloc_fragment((1,), "int32")
                        T.finalize_reducer(first, count)
                        accepted[0] = T.min(count[0], drafts)
                    T.copy(accepted[:1], A[row : row + 1])
                    T.copy(uniform[:1], U[row : row + 1])

    return tilelang.compile(main, target="ascend", out_idx=[5, 6])
