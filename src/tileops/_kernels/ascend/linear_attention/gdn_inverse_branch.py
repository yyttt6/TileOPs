"""Stable lower-triangular gated inverse; row beta is applied after GEMM."""

import tilelang.ascend.language as T


def inverse(J, dtype, C=32, gate_threads=256):
    assert C == 32
    grid = min(64, J)

    @T.prim_func
    def main(
        Gram: T.Tensor((J, C, C), "float32"),
        QK: T.Tensor((J, C, C), "float32"),
        Beta: T.Tensor((J, C), "float32"),
        Prefix: T.Tensor((J, C), "float32"),
        B: T.Tensor((J, C, C), dtype),
        Att: T.Tensor((J, C, C), dtype),
    ):
        with T.Kernel(grid) as bx:
            gram = T.alloc_shared((C, C), "float32")
            raw = T.alloc_shared((C, C), "float32")
            beta = T.alloc_shared((C,), "float32")
            g = T.alloc_shared((C,), "float32")
            for step in T.serial((J + grid - 1) // grid):
                job = step * grid + bx
                if job < J:
                    T.copy(Gram[job, :, :], raw)
                    T.copy(Beta[job, :], beta)
                    T.copy(Prefix[job, :], g)
                    with T.SimtVF(threads=gate_threads):
                        for i, j in T.Parallel(C, C):
                            if j < i:
                                gram[i, j] = raw[i, j] * T.exp(g[i] - g[j])
                            else:
                                gram[i, j] = 0.0
                            if j <= i:
                                Att[job, i, j] = QK[job, i, j] * T.exp(g[i] - g[j])
                            else:
                                Att[job, i, j] = 0.0
                    with T.SimtVF(threads=1024):
                        tid = T.get_thread_binding()
                        i = tid // 32
                        j = tid % 32
                        value = T.alloc_local((1,), "float32")
                        value[0] = T.if_then_else(
                            i == j,
                            T.float32(1),
                            T.if_then_else(i > j, -beta[i] * gram[i, j], T.float32(0)),
                        )
                        for pivot in T.unroll(C - 1):
                            u = C - 1 - pivot
                            if i > u:
                                coefficient = (
                                    T.call_extern(  # lint: extern-target allow: Ascend shuffle
                                        "float32", "asc_shfl", value[0], u, 32
                                    )
                                    * beta[u]
                                )
                                value[0] = T.if_then_else(
                                    j < u, value[0] - coefficient * gram[u, j], value[0]
                                )
                        B[job, i, j] = (
                            value[0]
                            * beta[j]
                            / T.if_then_else(beta[i] != 0.0, beta[i], T.float32(1))
                        )

    return main
