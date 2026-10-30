"""Decode QKV-A GEMM: 27 tiles of 128x80 (last 32 columns), K=7168.

Each GEMM is 128x80x128. Event ids: MTE2_MTE1 GM->L1, MTE1_M L1->L0, M cube,
MTE1_MTE2 L0 free, M_FIX L0C->GM.
"""

import tilelang
import tilelang.ascend.language as T


@T.macro
def swizzle(tile_idx):
    m_tile = T.alloc_var("int32")
    n_tile = T.alloc_var("int32")
    m_tile = tile_idx // 27
    n_tile = tile_idx % 27
    return m_tile, n_tile


@T.prim_func
def main(
    A: T.Tensor((128, 7168), "bfloat16"),
    B: T.Tensor((2112, 7168), "bfloat16"),
    D: T.Tensor((128, 2112), "bfloat16"),
):
    T.func_attr({"tilelang_callee_allocated_outputs": 0})
    with T.Kernel(27) as bx:  # noqa: SIM117 - Keep the measured TileLang scope structure.
        with T.Cube():
            a_l0 = T.alloc_l0a((2, 128, 128), "bfloat16")
            a_l1 = T.alloc_l1((2, 128, 256), "bfloat16")
            b_l0 = T.alloc_l0b((2, 80, 128), "bfloat16")
            b_l1 = T.alloc_l1((2, 80, 256), "bfloat16")
            c_l0 = T.alloc_l0c((128, 80), "float32")
            for tile_idx in T.Persistent([27], 27, bx):
                m_tile, n_tile = swizzle(tile_idx)
                T.assume(m_tile >= 0)
                T.assume(m_tile < 1)
                T.assume(n_tile >= 0)
                T.assume(n_tile < 27)
                m_base = m_tile * 128
                n_base = n_tile * 80
                # 80-column tiles; the last is a 32-column tail.
                if n_tile < 26:
                    T.copy(A[m_base : m_base + 128, 0:256], a_l1[0, :128, :256])
                    T.copy(
                        B[n_base : n_base + 80, 0:256],
                        b_l1[0, :80, :256],
                        l2_cache_ctrl="NOTALLOC_KEEP",
                    )
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.copy(A[m_base : m_base + 128, 256:512], a_l1[1, :128, :256])
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:128], a_l0[0, :128, :])
                    T.copy(b_l1[0, 0:80, 0:128], b_l0[0, :80, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :80, :],
                        c_l0[:128, :80],
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.copy(b_l1[0, 0:80, 128:256], b_l0[1, :80, :])
                    T.copy(a_l1[0, 0:128, 128:256], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    # Steady state: two 256-wide K tiles per cycle.
                    for cycle_0 in range(13):
                        T.ascend_set_flag("MTE1_MTE2", 0)
                        T.copy(
                            B[n_base : n_base + 80, 256 + 512 * cycle_0 : 512 + 512 * cycle_0],
                            b_l1[1, :80, :256],
                            l2_cache_ctrl="NOTALLOC_KEEP",
                        )
                        T.ascend_set_flag("MTE2_MTE1", 1)
                        T.ascend_wait_flag("MTE1_M", 1)
                        T.gemm(
                            a_l0[1, :128, :],
                            b_l0[1, :80, :],
                            c_l0[:128, :80],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", 1)
                        T.ascend_wait_flag("MTE1_MTE2", 0)
                        T.copy(
                            A[m_base : m_base + 128, 512 + 512 * cycle_0 : 768 + 512 * cycle_0],
                            a_l1[0, :128, :256],
                        )
                        T.ascend_wait_flag("M_MTE1", 0)
                        T.ascend_wait_flag("MTE2_MTE1", 1)
                        T.copy(a_l1[1, 0:128, 0:128], a_l0[0, :128, :])
                        T.copy(b_l1[1, 0:80, 0:128], b_l0[0, :80, :])
                        T.ascend_set_flag("MTE1_M", 0)
                        T.ascend_wait_flag("MTE1_M", 0)
                        T.gemm(
                            a_l0[0, :128, :],
                            b_l0[0, :80, :],
                            c_l0[:128, :80],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", 0)
                        T.ascend_wait_flag("M_MTE1", 1)
                        T.copy(b_l1[1, 0:80, 128:256], b_l0[1, :80, :])
                        T.copy(a_l1[1, 0:128, 128:256], a_l0[1, :128, :])
                        T.ascend_set_flag("MTE1_M", 1)
                        T.ascend_set_flag("MTE1_MTE2", 1)
                        T.copy(
                            B[n_base : n_base + 80, 512 + 512 * cycle_0 : 768 + 512 * cycle_0],
                            b_l1[0, :80, :256],
                            l2_cache_ctrl="NOTALLOC_KEEP",
                        )
                        T.ascend_set_flag("MTE2_MTE1", 0)
                        T.ascend_wait_flag("MTE1_M", 1)
                        T.gemm(
                            a_l0[1, :128, :],
                            b_l0[1, :80, :],
                            c_l0[:128, :80],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", 1)
                        T.ascend_wait_flag("MTE1_MTE2", 1)
                        T.copy(
                            A[m_base : m_base + 128, 768 + 512 * cycle_0 : 1024 + 512 * cycle_0],
                            a_l1[1, :128, :256],
                        )
                        T.ascend_wait_flag("M_MTE1", 0)
                        T.ascend_wait_flag("MTE2_MTE1", 0)
                        T.copy(a_l1[0, 0:128, 0:128], a_l0[0, :128, :])
                        T.copy(b_l1[0, 0:80, 0:128], b_l0[0, :80, :])
                        T.ascend_set_flag("MTE1_M", 0)
                        T.ascend_wait_flag("MTE1_M", 0)
                        T.gemm(
                            a_l0[0, :128, :],
                            b_l0[0, :80, :],
                            c_l0[:128, :80],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", 0)
                        T.ascend_wait_flag("M_MTE1", 1)
                        T.copy(b_l1[0, 0:80, 128:256], b_l0[1, :80, :])
                        T.copy(a_l1[0, 0:128, 128:256], a_l0[1, :128, :])
                        T.ascend_set_flag("MTE1_M", 1)
                    T.copy(
                        B[n_base : n_base + 80, 6912:7168],
                        b_l1[1, :80, :256],
                        l2_cache_ctrl="NOTALLOC_KEEP",
                    )
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :80, :],
                        c_l0[:128, :80],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:128], a_l0[0, :128, :])
                    T.copy(b_l1[1, 0:80, 0:128], b_l0[0, :80, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :80, :],
                        c_l0[:128, :80],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(b_l1[1, 0:80, 128:256], b_l0[1, :80, :])
                    T.copy(a_l1[1, 0:128, 128:256], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :80, :],
                        c_l0[:128, :80],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_FIX", 0)
                    # Store the finished tile.
                    T.ascend_wait_flag("M_FIX", 0)
                    T.copy(c_l0[:128, :80], D[m_base + 0 : m_base + 128, n_base + 0 : n_base + 80])
                    T.ascend_pipe_barrier("PIPE_ALL")
                # 32-column N tail.
                elif n_tile == 26:
                    T.copy(A[m_base : m_base + 128, 0:256], a_l1[0, :128, :256])
                    T.copy(
                        B[n_base : n_base + 32, 0:256],
                        b_l1[0, :32, :256],
                        l2_cache_ctrl="NOTALLOC_KEEP",
                    )
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.copy(A[m_base : m_base + 128, 256:512], a_l1[1, :128, :256])
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:128], a_l0[0, :128, :])
                    T.copy(b_l1[0, 0:32, 0:128], b_l0[0, :32, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.copy(b_l1[0, 0:32, 128:256], b_l0[1, :32, :])
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :32, :],
                        c_l0[:128, :32],
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 128:256], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    # Same loop on the 32-column tail.
                    for cycle_0 in range(13):
                        T.ascend_set_flag("MTE1_MTE2", 0)
                        T.ascend_wait_flag("MTE1_M", 1)
                        T.gemm(
                            a_l0[1, :128, :],
                            b_l0[1, :32, :],
                            c_l0[:128, :32],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", 1)
                        T.copy(
                            B[n_base : n_base + 32, 256 + 512 * cycle_0 : 512 + 512 * cycle_0],
                            b_l1[1, :32, :256],
                            l2_cache_ctrl="NOTALLOC_KEEP",
                        )
                        T.ascend_set_flag("MTE2_MTE1", 1)
                        T.ascend_wait_flag("MTE1_MTE2", 0)
                        T.copy(
                            A[m_base : m_base + 128, 512 + 512 * cycle_0 : 768 + 512 * cycle_0],
                            a_l1[0, :128, :256],
                        )
                        T.ascend_wait_flag("M_MTE1", 0)
                        T.ascend_wait_flag("MTE2_MTE1", 1)
                        T.copy(a_l1[1, 0:128, 0:128], a_l0[0, :128, :])
                        T.copy(b_l1[1, 0:32, 0:128], b_l0[0, :32, :])
                        T.ascend_set_flag("MTE1_M", 0)
                        T.ascend_wait_flag("M_MTE1", 1)
                        T.copy(b_l1[1, 0:32, 128:256], b_l0[1, :32, :])
                        T.ascend_wait_flag("MTE1_M", 0)
                        T.gemm(
                            a_l0[0, :128, :],
                            b_l0[0, :32, :],
                            c_l0[:128, :32],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", 0)
                        T.copy(a_l1[1, 0:128, 128:256], a_l0[1, :128, :])
                        T.ascend_set_flag("MTE1_M", 1)
                        T.ascend_set_flag("MTE1_MTE2", 1)
                        T.ascend_wait_flag("MTE1_M", 1)
                        T.gemm(
                            a_l0[1, :128, :],
                            b_l0[1, :32, :],
                            c_l0[:128, :32],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", 1)
                        T.copy(
                            B[n_base : n_base + 32, 512 + 512 * cycle_0 : 768 + 512 * cycle_0],
                            b_l1[0, :32, :256],
                            l2_cache_ctrl="NOTALLOC_KEEP",
                        )
                        T.ascend_set_flag("MTE2_MTE1", 0)
                        T.ascend_wait_flag("MTE1_MTE2", 1)
                        T.copy(
                            A[m_base : m_base + 128, 768 + 512 * cycle_0 : 1024 + 512 * cycle_0],
                            a_l1[1, :128, :256],
                        )
                        T.ascend_wait_flag("M_MTE1", 0)
                        T.ascend_wait_flag("MTE2_MTE1", 0)
                        T.copy(a_l1[0, 0:128, 0:128], a_l0[0, :128, :])
                        T.copy(b_l1[0, 0:32, 0:128], b_l0[0, :32, :])
                        T.ascend_set_flag("MTE1_M", 0)
                        T.ascend_wait_flag("M_MTE1", 1)
                        T.copy(b_l1[0, 0:32, 128:256], b_l0[1, :32, :])
                        T.ascend_wait_flag("MTE1_M", 0)
                        T.gemm(
                            a_l0[0, :128, :],
                            b_l0[0, :32, :],
                            c_l0[:128, :32],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", 0)
                        T.copy(a_l1[0, 0:128, 128:256], a_l0[1, :128, :])
                        T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :32, :],
                        c_l0[:128, :32],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.copy(
                        B[n_base : n_base + 32, 6912:7168],
                        b_l1[1, :32, :256],
                        l2_cache_ctrl="NOTALLOC_KEEP",
                    )
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:128], a_l0[0, :128, :])
                    T.copy(b_l1[1, 0:32, 0:128], b_l0[0, :32, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(b_l1[1, 0:32, 128:256], b_l0[1, :32, :])
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :32, :],
                        c_l0[:128, :32],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.copy(a_l1[1, 0:128, 128:256], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :32, :],
                        c_l0[:128, :32],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_FIX", 0)
                    T.ascend_wait_flag("M_FIX", 0)
                    T.copy(c_l0[:128, :32], D[m_base + 0 : m_base + 128, n_base + 0 : n_base + 32])
                    T.ascend_pipe_barrier("PIPE_ALL")


def compile_kernel(m: int, n: int, k: int, trans_b: bool, dtype: str):
    """Compile this schedule; it serves one fixed BF16 NT shape."""
    if (m, n, k) != (128, 2112, 7168) or not trans_b or dtype != "bfloat16":
        raise ValueError(
            f"this schedule serves BF16 NT (128, 2112, 7168), got {(m, n, k)}, "
            f"trans_b={trans_b}, dtype={dtype}"
        )
    return tilelang.compile(
        main,
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_auto_schedule": False, "tl.disable_shared_memory_reuse": True},
    )
