"""Persistent GEMM feeding the matrix unit from L1, with dynamic M/N tail views."""

import tilelang
import tilelang.ascend.language as T

# One entry per served shape; every field names a value the schedule fixes per case.
_CONFIGS = {
    (1, 2112, 7168): {
        "block_m": 16,
        "block_n": 80,
        "block_k": 512,
        "stages": 2,
        "blocks": 32,
    },
    (2, 2112, 7168): {
        "block_m": 16,
        "block_n": 80,
        "block_k": 512,
        "stages": 2,
        "blocks": 32,
    },
    (256, 21843, 1024): {
        "block_m": 256,
        "block_n": 256,
        "block_k": 256,
        "stages": 2,
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

    GEMM reads both operands from L1; the unit flags order the final MAC. The
    scheduler hands out M/N tile remainders to the copy and store views, so a
    partial M or N tile only writes lanes the output discards. K must be a whole
    number of ``block_k`` tiles: this schedule reads a fixed K span per GEMM.
    """
    parameters = (m, n, k, block_m, block_n, block_k, stages, blocks)
    if any(type(value) is not int or value <= 0 for value in parameters):
        raise ValueError("dimensions, physical tiles, stages and blocks must be positive integers")
    if any(tile % 16 for tile in (block_m, block_n, block_k)):
        raise ValueError("16-aligned physical tiles required")
    if k % block_k:
        raise ValueError("this schedule requires whole K tiles; M and N tails are supported")
    if dtype != "bfloat16":
        raise ValueError(f"this schedule serves bfloat16 only, got {dtype}")
    if 2 * (block_m + block_n) * block_k * stages > 512 * 1024:
        raise ValueError("L1 operand versions exceed the 512 KiB capacity")
    if 4 * block_m * block_n > 256 * 1024:
        raise ValueError("FP32 accumulator exceeds the 256 KiB L0C capacity")
    grid = min(blocks, ((m + block_m - 1) // block_m) * ((n + block_n - 1) // block_n))
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
            acc = T.alloc_l0c((block_m, block_n), "float32")
            T.annotate_buffer_versions({a_l1: stages, b_l1: stages})
            sched.init(bx)
            while sched.valid():
                mi, ni, am, an = sched.tile()
                for ki in T.Pipelined(T.ceildiv(k, block_k), num_stages=stages):
                    start = ki * block_k
                    ak = block_k
                    T.copy(A[mi : mi + am, start : start + ak], a_l1[0:am, 0:ak])
                    if trans_b:
                        T.copy(B[ni : ni + an, start : start + ak], b_l1[0:an, 0:ak])
                    else:
                        T.copy(
                            B[start : start + ak, ni : ni + an],
                            b_l1[0:an, 0:ak],
                            transpose=True,
                        )
                    # Cube reads both operands from L1; unit flags order the final MAC.
                    # Inactive M/N lanes only produce discarded output lanes.
                    T.gemm(
                        a_l1,
                        b_l1,
                        acc,
                        transpose_B=True,
                        clear_accum=ki == 0,
                        unit_flag_ctrl=T.Select(ki == T.ceildiv(k, block_k) - 1, 3, 2),
                    )
                T.copy(acc[0:am, 0:an], D[mi : mi + am, ni : ni + an], unit_flag_ctrl=3)
                sched.next_block()

    return main


def compile_kernel(m: int, n: int, k: int, trans_b: bool, dtype: str):
    """Compile the schedule configured for ``(m, n, k)``; it serves BF16 NT."""
    if not trans_b:
        raise ValueError("this schedule serves trans_b=True only")
    return tilelang.compile(
        build(m, n, k, trans_b=trans_b, dtype=dtype, **_CONFIGS[m, n, k]),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )
