import functools

import tilelang
import tilelang.ascend.language as T


@functools.lru_cache(maxsize=128)
def build_batched(batch, seq, heads, dim, dtype, interleaved):
    half = dim // 2
    positions = max(1, min(8, 16384 // (heads * dim)))
    tiles = (seq + positions - 1) // positions
    jobs = batch * tiles
    grid = min(64, jobs)

    @T.prim_func
    def main(
        X: T.Tensor((batch, seq, heads, dim), dtype),
        C: T.Tensor((seq, half), dtype),
        S: T.Tensor((seq, half), dtype),
        Y: T.Tensor((batch, seq, heads, dim), dtype),
    ):
        with T.Kernel(grid) as bx:
            x = T.alloc_shared((positions, heads, dim), dtype)
            y = T.alloc_shared((positions, heads, dim), dtype)
            c = T.alloc_shared((positions, half), dtype)
            s = T.alloc_shared((positions, half), dtype)
            T.annotate_buffer_versions({x: 2, y: 2, c: 2, s: 2})
            for step in T.Pipelined((jobs + grid - 1) // grid, num_stages=2):
                job = step * grid + bx
                b = job // tiles
                pos = job % tiles * positions
                T.copy(X[b, pos : pos + positions, :, :], x[:positions, :, :], l2_cache_ctrl=4)
                T.copy(C[pos : pos + positions, :], c[:positions, :])
                T.copy(S[pos : pos + positions, :], s[:positions, :])
                with T.SimtVF(threads=1024):
                    for r, h, k in T.Parallel(positions, heads, half):
                        i = 2 * k if interleaved else k
                        j = 2 * k + 1 if interleaved else k + half
                        av = T.cast(x[r, h, i], "float32")
                        bv = T.cast(x[r, h, j], "float32")
                        cv = T.cast(c[r, k], "float32")
                        sv = T.cast(s[r, k], "float32")
                        y[r, h, i] = av * cv - bv * sv
                        y[r, h, j] = bv * cv + av * sv
                T.copy(y[:positions, :, :], Y[b, pos : pos + positions, :, :], l2_cache_ctrl=0)

    return tilelang.compile(main, target="ascend", out_idx=[3])
