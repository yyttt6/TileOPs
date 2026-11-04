"""Persistent GEMM with explicit L0 staging and dynamic M/N tail views."""

import tilelang
import tilelang.ascend.language as T

# One entry per served shape; every field names a value the schedule fixes per case.
_CONFIGS = {
    (1024, 1024, 1024): {
        "block_m": 128,
        "block_n": 128,
        "block_k": 128,
        "stages": 3,
        "blocks": 32,
    },
    (1, 24576, 4096): {
        "block_m": 16,
        "block_n": 128,
        "block_k": 128,
        "stages": 3,
        "blocks": 32,
    },
    (128, 4096, 12288): {
        "block_m": 128,
        "block_n": 128,
        "block_k": 128,
        "stages": 3,
        "blocks": 32,
    },
}


def build(
    m: int,
    n: int,
    k: int,
    trans_b: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    blocks: int,
):
    """Build one ``D = A @ B`` with ``B`` stored ``(n, k)`` when ``trans_b``.

    The scheduler hands out M/N tile remainders to the copy and store views, so a
    partial M or N tile only writes lanes the output discards; the K loop shrinks
    the last copy and its GEMM the same way.
    """
    parameters = (m, n, k, block_m, block_n, block_k, stages, blocks)
    if any(type(value) is not int or value <= 0 for value in parameters):
        raise ValueError("dimensions, physical tiles, stages and blocks must be positive integers")
    if 4 * max(block_m, block_n) * block_k > 64 * 1024:
        raise ValueError("two 16-bit L0 versions exceed the 64 KiB per-operand capacity")
    if 2 * (block_m + block_n) * block_k * stages > 512 * 1024:
        raise ValueError("L1 operand versions exceed the 512 KiB capacity")
    if 4 * block_m * block_n > 256 * 1024:
        raise ValueError("FP32 accumulator exceeds the 256 KiB L0C capacity")
    grid = min(blocks, ((m + block_m - 1) // block_m) * ((n + block_n - 1) // block_n))
    if dtype not in ("bfloat16", "float16"):
        raise ValueError(f"this schedule serves 16-bit floats, got {dtype}")
    b_shape = (n, k) if trans_b else (k, n)

    @T.prim_func
    def main(
        A: T.Tensor((m, k), dtype),
        B: T.Tensor(b_shape, dtype),
        D: T.Tensor((m, n), dtype),
    ):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(grid) as bx:
            # One block per output tile; M/N remainders shrink the copies.
            sched = T.AscendBatchedTileScheduler(
                block_m=block_m,
                block_n=block_n,
                num_cores=grid,
                shape_m=m,
                shape_n=n,
                num_groups=1,
            )
            a_l1 = T.alloc_l1((block_m, block_k), dtype)
            b_l1 = T.alloc_l1((block_n, block_k), dtype)
            a_l0 = T.alloc_l0a((block_m, block_k), dtype)
            b_l0 = T.alloc_l0b((block_n, block_k), dtype)
            acc = T.alloc_l0c((block_m, block_n), "float32")
            T.annotate_buffer_versions({a_l1: stages, b_l1: stages, a_l0: 2, b_l0: 2})
            sched.init(bx)
            while sched.valid():
                mi, ni, am, an = sched.tile()
                # Two-stage L1 pipeline; each K tile is staged to L0A/L0B.
                for ki in T.Pipelined(T.ceildiv(k, block_k), num_stages=stages):
                    start = ki * block_k
                    ak = T.min(k - start, block_k)
                    T.copy(A[mi : mi + am, start : start + ak], a_l1[0:am, 0:ak])
                    if trans_b:
                        T.copy(B[ni : ni + an, start : start + ak], b_l1[0:an, 0:ak])
                    else:
                        T.copy(
                            B[start : start + ak, ni : ni + an],
                            b_l1[0:an, 0:ak],
                            transpose=True,
                        )
                    T.copy(a_l1[0:am, 0:ak], a_l0[0:am, 0:ak])
                    T.copy(b_l1[0:an, 0:ak], b_l0[0:an, 0:ak])
                    T.gemm(
                        a_l0[0:am, 0:ak],
                        b_l0[0:an, 0:ak],
                        acc[0:am, 0:an],
                        transpose_B=True,
                        clear_accum=ki == 0,
                    )
                T.copy(acc[0:am, 0:an], D[mi : mi + am, ni : ni + an])
                sched.next_block()

    return main


def compile_kernel(m: int, n: int, k: int, trans_b: bool, dtype: str):
    """Compile the schedule configured for ``(m, n, k)``, its layout and dtype."""
    return tilelang.compile(
        build(m, n, k, trans_b=trans_b, dtype=dtype, **_CONFIGS[m, n, k]),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )
