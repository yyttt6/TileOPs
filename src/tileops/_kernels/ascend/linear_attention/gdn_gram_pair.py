"""Two Gram products sharing the K operand and a single Cube launch."""

import functools

import tilelang
import tilelang.ascend.language as T


@functools.lru_cache(maxsize=96)
def build(J, C, K, dtype):
    grid = min(32, J)

    @T.prim_func
    def main(
        Q: T.Tensor((J, C, K), dtype),
        KK: T.Tensor((J, C, K), dtype),
        Gram: T.Tensor((J, C, C), "float32"),
        QK: T.Tensor((J, C, C), "float32"),
    ):
        with T.Kernel(grid) as bx:
            k = T.alloc_l1((C, K), dtype)
            q = T.alloc_l1((C, K), dtype)
            ka = T.alloc_l0a((C, K), dtype)
            qa = T.alloc_l0a((C, K), dtype)
            kb = T.alloc_l0b((C, K), dtype)
            gram = T.alloc_l0c((C, C), "float32")
            qk = T.alloc_l0c((C, C), "float32")
            for step in T.Pipelined((J + grid - 1) // grid, num_stages=2):
                j = step * grid + bx
                if j < J:
                    T.copy(KK[j, :, :], k)
                    T.copy(Q[j, :, :], q)
                    T.copy(k, ka)
                    T.copy(k, kb)
                    T.copy(q, qa)
                    T.gemm(ka, kb, gram, transpose_B=True, clear_accum=True)
                    T.copy(gram, Gram[j, :, :])
                    T.gemm(qa, kb, qk, transpose_B=True, clear_accum=True)
                    T.copy(qk, QK[j, :, :])

    return tilelang.compile(main, target="ascend", out_idx=[2, 3])
