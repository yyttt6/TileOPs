"""M=16 MLP-up GEMM: 32 tiles of 16x128, K=7168.

Each GEMM is 16x128x128. Event ids: MTE2_MTE1 GM->L1, MTE1_M L1->L0, M cube,
MTE1_MTE2 L0 free, M_FIX L0C->GM.
"""

import tilelang
import tilelang.ascend.language as T


@T.macro
def swizzle(tile_idx):
    m_tile = T.alloc_var("int32")
    n_tile = T.alloc_var("int32")
    m_tile = tile_idx // 32
    n_tile = tile_idx % 32
    return m_tile, n_tile


@T.prim_func
def main(
    A: T.Tensor((16, 7168), "bfloat16"),
    B: T.Tensor((4096, 7168), "bfloat16"),
    D: T.Tensor((16, 4096), "bfloat16"),
):
    with T.Kernel(32) as bx:  # noqa: SIM117 - Keep the measured TileLang scope structure.
        with T.Cube():
            a_l0 = T.alloc_l0a((1, 16, 128), "bfloat16")
            a_l1 = T.alloc_l1((1, 16, 256), "bfloat16")
            b_l0 = T.alloc_l0b((2, 128, 128), "bfloat16")
            b_l1 = T.alloc_l1((2, 128, 256), "bfloat16")
            c_l0 = T.alloc_l0c((16, 128), "float32")
            for tile_idx in T.Persistent([32], 32, bx):
                m_tile, n_tile = swizzle(tile_idx)
                T.assume(m_tile >= 0)
                T.assume(m_tile < 1)
                T.assume(n_tile >= 0)
                T.assume(n_tile < 32)
                m_base = m_tile * 16
                n_base = n_tile * 128
                T.copy(
                    B[n_base : n_base + 128, 0:256], b_l1[0, :, :], l2_cache_ctrl="NOTALLOC_KEEP"
                )
                T.ascend_set_flag("MTE2_MTE1", 1)
                T.copy(A[m_base : m_base + 16, 0:256], a_l1[0, :, :])
                T.ascend_set_flag("MTE2_MTE1", 0)
                T.copy(
                    B[n_base : n_base + 128, 256:512], b_l1[1, :, :], l2_cache_ctrl="NOTALLOC_KEEP"
                )
                T.ascend_set_flag("MTE2_MTE1", 2)
                T.ascend_wait_flag("MTE2_MTE1", 1)
                T.copy(b_l1[0, :, 0:128], b_l0[0, :, :])
                T.ascend_set_flag("MTE1_M", 1)
                T.ascend_wait_flag("MTE2_MTE1", 0)
                T.copy(a_l1[0, :, 0:128], a_l0[0, :, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.copy(b_l1[0, :, 128:256], b_l0[1, :, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.ascend_set_flag("MTE1_MTE2", 1)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 1)
                T.gemm(a_l0[0, :, :], b_l0[0, :, :], c_l0, transpose_B=True, clear_accum=True)
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 1)
                T.ascend_wait_flag("M_MTE1", 0)
                T.copy(a_l1[0, :, 128:256], a_l0[0, :, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_set_flag("MTE1_MTE2", 0)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(a_l0[0, :, :], b_l0[1, :, :], c_l0, transpose_B=True, clear_accum=False)
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 2)
                T.ascend_wait_flag("MTE1_MTE2", 0)
                T.copy(A[m_base : m_base + 16, 256:512], a_l1[0, :, :])
                T.ascend_set_flag("MTE2_MTE1", 0)
                T.ascend_wait_flag("MTE1_MTE2", 1)
                T.copy(
                    B[n_base : n_base + 128, 512:768], b_l1[0, :, :], l2_cache_ctrl="NOTALLOC_KEEP"
                )
                T.ascend_set_flag("MTE2_MTE1", 1)
                T.ascend_wait_flag("M_MTE1", 1)
                T.ascend_wait_flag("MTE2_MTE1", 2)
                T.copy(b_l1[1, :, 0:128], b_l0[0, :, :])
                T.ascend_set_flag("MTE1_M", 1)
                T.ascend_wait_flag("M_MTE1", 0)
                T.ascend_wait_flag("MTE2_MTE1", 0)
                T.copy(a_l1[0, :, 0:128], a_l0[0, :, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("M_MTE1", 2)
                T.copy(b_l1[1, :, 128:256], b_l0[1, :, :])
                T.ascend_set_flag("MTE1_M", 2)
                # Steady state: one 256-wide K tile per rotation.
                for rotation_0 in range(25):
                    T.ascend_set_flag("MTE1_MTE2", 1 + (1 + rotation_0) % 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(a_l0[0, :, :], b_l0[0, :, :], c_l0, transpose_B=True, clear_accum=False)
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, :, 128:256], a_l0[0, :, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(a_l0[0, :, :], b_l0[1, :, :], c_l0, transpose_B=True, clear_accum=False)
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(
                        A[
                            m_base : m_base + 16,
                            512 + 256 * rotation_0 : (512 + 256 * rotation_0) + 256,
                        ],
                        a_l1[0, :, :],
                    )
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 1 + (1 + rotation_0) % 2)
                    T.copy(
                        B[
                            n_base : n_base + 128,
                            768 + 256 * rotation_0 : (768 + 256 * rotation_0) + 256,
                        ],
                        b_l1[(1 + rotation_0) % 2, :, :],
                        l2_cache_ctrl="NOTALLOC_KEEP",
                    )
                    T.ascend_set_flag("MTE2_MTE1", 1 + (1 + rotation_0) % 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 1 + rotation_0 % 2)
                    T.copy(b_l1[rotation_0 % 2, :, 0:128], b_l0[0, :, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, :, 0:128], a_l0[0, :, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(b_l1[rotation_0 % 2, :, 128:256], b_l0[1, :, :])
                    T.ascend_set_flag("MTE1_M", 2)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 1)
                T.gemm(a_l0[0, :, :], b_l0[0, :, :], c_l0, transpose_B=True, clear_accum=False)
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 1)
                T.ascend_wait_flag("M_MTE1", 0)
                T.copy(a_l1[0, :, 128:256], a_l0[0, :, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_set_flag("MTE1_MTE2", 0)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(a_l0[0, :, :], b_l0[1, :, :], c_l0, transpose_B=True, clear_accum=False)
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 2)
                T.ascend_wait_flag("MTE1_MTE2", 0)
                T.copy(A[m_base : m_base + 16, 6912:7168], a_l1[0, :, :])
                T.ascend_set_flag("MTE2_MTE1", 0)
                T.ascend_wait_flag("M_MTE1", 1)
                T.ascend_wait_flag("MTE2_MTE1", 2)
                T.copy(b_l1[1, :, 0:128], b_l0[0, :, :])
                T.ascend_set_flag("MTE1_M", 1)
                T.ascend_wait_flag("M_MTE1", 0)
                T.ascend_wait_flag("MTE2_MTE1", 0)
                T.copy(a_l1[0, :, 0:128], a_l0[0, :, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("M_MTE1", 2)
                T.copy(b_l1[1, :, 128:256], b_l0[1, :, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 1)
                T.gemm(a_l0[0, :, :], b_l0[0, :, :], c_l0, transpose_B=True, clear_accum=False)
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_wait_flag("M_MTE1", 0)
                T.copy(a_l1[0, :, 128:256], a_l0[0, :, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(a_l0[0, :, :], b_l0[1, :, :], c_l0, transpose_B=True, clear_accum=False)
                T.ascend_set_flag("M_FIX", 0)
                # Store the finished tile.
                T.ascend_wait_flag("M_FIX", 0)
                T.copy(c_l0, D[m_base : m_base + 16, n_base : n_base + 128])
                T.ascend_pipe_barrier("PIPE_ALL")


def compile_kernel(m: int, n: int, k: int, trans_b: bool, dtype: str):
    """Compile this schedule; it serves one fixed BF16 NT shape."""
    if (m, n, k) != (16, 4096, 7168) or not trans_b or dtype != "bfloat16":
        raise ValueError(
            f"this schedule serves BF16 NT (16, 4096, 7168), got {(m, n, k)}, "
            f"trans_b={trans_b}, dtype={dtype}"
        )
    return tilelang.compile(
        main,
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_auto_schedule": False, "tl.disable_shared_memory_reuse": True},
    )
