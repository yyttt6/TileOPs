"""Prefill QKV-A GEMM: 16x9 persistent tiles of 256x240, K=7168.

Four L1 versions feed two 128-row accumulators through five phases (2, 2, 49,
2, 3). Event ids: MTE2_MTE1 GM->L1, MTE1_M L1->L0, M cube, MTE1_MTE2 L0 free,
M_FIX L0C->GM.
"""

import tilelang
import tilelang.ascend.language as T


@T.macro
def swizzle(tile_idx):
    m_tile = T.alloc_var("int32")
    n_tile = T.alloc_var("int32")
    if tile_idx >= 144:
        m_tile = tile_idx // 9
        n_tile = tile_idx % 9
    else:
        superrow = tile_idx // 36
        within = tile_idx % 36
        if within < 32:
            group = within // 32
            local = within % 32
            m_tile = superrow * 4 + local % 4
            n_group = T.Select(superrow % 2 == 0, group, 0 - group)
            n_tile = n_group * 8 + ((local // 4) ^ (local % 4))
        else:
            remainder = within - 32
            m_tile = superrow * 4 + remainder % 4
            n_tile = 8 + remainder // 4
    return m_tile, n_tile


@T.prim_func
def main(
    A: T.Tensor((4096, 7168), "bfloat16"),
    B: T.Tensor((2112, 7168), "bfloat16"),
    D: T.Tensor((4096, 2112), "bfloat16"),
):
    with T.Kernel(32) as bx:  # noqa: SIM117 - Keep the measured TileLang scope structure.
        with T.Cube():
            a_l0 = T.alloc_l0a((3, 128, 64), "bfloat16")
            a_l1 = T.alloc_l1((4, 256, 128), "bfloat16")
            b_l0 = T.alloc_l0b((2, 240, 64), "bfloat16")
            b_l1 = T.alloc_l1((4, 240, 128), "bfloat16")
            c_l0_0_0 = T.alloc_l0c((128, 240), "float32")
            c_l0_1_0 = T.alloc_l0c((128, 240), "float32")
            for tile_idx in T.Persistent([144], 32, bx):
                m_tile, n_tile = swizzle(tile_idx)
                T.assume(m_tile >= 0)
                T.assume(m_tile < 16)
                T.assume(n_tile >= 0)
                T.assume(n_tile < 9)
                m_base = m_tile * 256
                n_base = n_tile * 240
                # Eight full 240-column N tiles.
                if n_tile < 8:
                    T.copy(B[n_base : n_base + 240, 0:128], b_l1[0, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.copy(A[m_base : m_base + 256, 0:128], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.copy(b_l1[0, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    # Phases: prime (2, 2), steady (49), drain (2, 3).
                    for rotation_0 in range(2):
                        T.copy(
                            a_l1[
                                0,
                                128 * ((1 + rotation_0) % 2) : (128 * ((1 + rotation_0) % 2)) + 128,
                                64 * (rotation_0 % 2) : (64 * (rotation_0 % 2)) + 64,
                            ],
                            a_l0[1 + (1 + rotation_0) % 2, :128, :],
                        )
                        T.ascend_set_flag("MTE1_M", 1 + (1 + rotation_0) % 2)
                    T.copy(B[n_base : n_base + 240, 128:256], b_l1[1, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.copy(A[m_base : m_base + 256, 128:256], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    for rotation_1 in range(2):
                        T.ascend_set_flag("MTE1_MTE2", 5 + rotation_1 % 2)
                        T.ascend_wait_flag("M_MTE1", rotation_1 % 2)
                        T.copy(
                            a_l1[1 + rotation_1 % 2, 128:256, 0:64], a_l0[rotation_1 % 2, :128, :]
                        )
                        T.ascend_set_flag("MTE1_M", rotation_1 % 2)
                        T.ascend_wait_flag("M_MTE1", (2 + rotation_1) % 3)
                        T.copy(
                            a_l1[1 + rotation_1 % 2, 0:128, 64:128],
                            a_l0[(2 + rotation_1) % 3, :128, :],
                        )
                        T.ascend_set_flag("MTE1_M", (2 + rotation_1) % 3)
                        T.copy(
                            B[
                                n_base : n_base + 240,
                                256 + 128 * rotation_1 : (256 + 128 * rotation_1) + 128,
                            ],
                            b_l1[2 + rotation_1 % 2, :240, :],
                        )
                        T.ascend_set_flag("MTE2_MTE1", 6 + rotation_1 % 2)
                        T.ascend_wait_flag("MTE1_M", rotation_1 % 2)
                        T.gemm(
                            a_l0[rotation_1 % 2, :128, :],
                            b_l0[0, :240, :],
                            c_l0_1_0[:128, :240],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", rotation_1 % 2)
                        T.ascend_set_flag("M_MTE1", 3)
                        T.ascend_wait_flag("M_MTE1", 1 + rotation_1 % 2)
                        T.copy(
                            a_l1[1 + rotation_1 % 2, 128:256, 64:128],
                            a_l0[1 + rotation_1 % 2, :128, :],
                        )
                        T.ascend_set_flag("MTE1_M", 1 + rotation_1 % 2)
                        T.ascend_set_flag("MTE1_MTE2", 1 + rotation_1 % 2)
                        T.ascend_wait_flag("MTE1_M", (2 + rotation_1) % 3)
                        T.ascend_wait_flag("MTE1_M", 4)
                        T.gemm(
                            a_l0[(2 + rotation_1) % 3, :128, :],
                            b_l0[1, :240, :],
                            c_l0_0_0[:128, :240],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", (2 + rotation_1) % 3)
                        T.copy(
                            A[
                                m_base : m_base + 256,
                                256 + 128 * rotation_1 : (256 + 128 * rotation_1) + 128,
                            ],
                            a_l1[2 + rotation_1 % 2, :256, :],
                        )
                        T.ascend_set_flag("MTE2_MTE1", 2 + rotation_1 % 2)
                        T.ascend_wait_flag("M_MTE1", 3)
                        T.ascend_wait_flag("MTE2_MTE1", 6 + rotation_1 % 2)
                        T.copy(b_l1[2 + rotation_1 % 2, 0:240, 0:64], b_l0[0, :240, :])
                        T.ascend_set_flag("MTE1_M", 3)
                        T.ascend_wait_flag("MTE1_M", 1 + rotation_1 % 2)
                        T.gemm(
                            a_l0[1 + rotation_1 % 2, :128, :],
                            b_l0[1, :240, :],
                            c_l0_1_0[:128, :240],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", 1 + rotation_1 % 2)
                        T.ascend_set_flag("M_MTE1", 4)
                        T.ascend_wait_flag("M_MTE1", (2 + rotation_1) % 3)
                        T.ascend_wait_flag("MTE2_MTE1", 2 + rotation_1 % 2)
                        T.copy(
                            a_l1[2 + rotation_1 % 2, 0:128, 0:64],
                            a_l0[(2 + rotation_1) % 3, :128, :],
                        )
                        T.ascend_set_flag("MTE1_M", (2 + rotation_1) % 3)
                        T.ascend_wait_flag("MTE1_M", (2 + rotation_1) % 3)
                        T.ascend_wait_flag("MTE1_M", 3)
                        T.gemm(
                            a_l0[(2 + rotation_1) % 3, :128, :],
                            b_l0[0, :240, :],
                            c_l0_0_0[:128, :240],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", (2 + rotation_1) % 3)
                        T.ascend_wait_flag("M_MTE1", 4)
                        T.copy(b_l1[2 + rotation_1 % 2, 0:240, 64:128], b_l0[1, :240, :])
                        T.ascend_set_flag("MTE1_M", 4)
                    for rotation_2 in range(49):
                        T.ascend_set_flag("MTE1_MTE2", 4 + (3 + rotation_2) % 4)
                        T.ascend_wait_flag("M_MTE1", (2 + rotation_2) % 3)
                        T.copy(
                            a_l1[(3 + rotation_2) % 4, 128:256, 0:64],
                            a_l0[(2 + rotation_2) % 3, :128, :],
                        )
                        T.ascend_set_flag("MTE1_M", (2 + rotation_2) % 3)
                        T.ascend_wait_flag("M_MTE1", (1 + rotation_2) % 3)
                        T.copy(
                            a_l1[(3 + rotation_2) % 4, 0:128, 64:128],
                            a_l0[(1 + rotation_2) % 3, :128, :],
                        )
                        T.ascend_set_flag("MTE1_M", (1 + rotation_2) % 3)
                        T.ascend_wait_flag("MTE1_MTE2", 4 + rotation_2 % 4)
                        T.copy(
                            B[
                                n_base : n_base + 240,
                                512 + 128 * rotation_2 : (512 + 128 * rotation_2) + 128,
                            ],
                            b_l1[rotation_2 % 4, :240, :],
                        )
                        T.ascend_set_flag("MTE2_MTE1", 4 + rotation_2 % 4)
                        T.ascend_wait_flag("MTE1_M", (2 + rotation_2) % 3)
                        T.gemm(
                            a_l0[(2 + rotation_2) % 3, :128, :],
                            b_l0[0, :240, :],
                            c_l0_1_0[:128, :240],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", (2 + rotation_2) % 3)
                        T.ascend_set_flag("M_MTE1", 3)
                        T.ascend_wait_flag("M_MTE1", rotation_2 % 3)
                        T.copy(
                            a_l1[(3 + rotation_2) % 4, 128:256, 64:128],
                            a_l0[rotation_2 % 3, :128, :],
                        )
                        T.ascend_set_flag("MTE1_M", rotation_2 % 3)
                        T.ascend_set_flag("MTE1_MTE2", (3 + rotation_2) % 4)
                        T.ascend_wait_flag("MTE1_M", (1 + rotation_2) % 3)
                        T.ascend_wait_flag("MTE1_M", 4)
                        T.gemm(
                            a_l0[(1 + rotation_2) % 3, :128, :],
                            b_l0[1, :240, :],
                            c_l0_0_0[:128, :240],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", (1 + rotation_2) % 3)
                        T.ascend_wait_flag("MTE1_MTE2", rotation_2 % 4)
                        T.copy(
                            A[
                                m_base : m_base + 256,
                                512 + 128 * rotation_2 : (512 + 128 * rotation_2) + 128,
                            ],
                            a_l1[rotation_2 % 4, :256, :],
                        )
                        T.ascend_set_flag("MTE2_MTE1", rotation_2 % 4)
                        T.ascend_wait_flag("M_MTE1", 3)
                        T.ascend_wait_flag("MTE2_MTE1", 4 + rotation_2 % 4)
                        T.copy(b_l1[rotation_2 % 4, 0:240, 0:64], b_l0[0, :240, :])
                        T.ascend_set_flag("MTE1_M", 3)
                        T.ascend_wait_flag("MTE1_M", rotation_2 % 3)
                        T.gemm(
                            a_l0[rotation_2 % 3, :128, :],
                            b_l0[1, :240, :],
                            c_l0_1_0[:128, :240],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", rotation_2 % 3)
                        T.ascend_set_flag("M_MTE1", 4)
                        T.ascend_wait_flag("M_MTE1", (1 + rotation_2) % 3)
                        T.ascend_wait_flag("MTE2_MTE1", rotation_2 % 4)
                        T.copy(
                            a_l1[rotation_2 % 4, 0:128, 0:64], a_l0[(1 + rotation_2) % 3, :128, :]
                        )
                        T.ascend_set_flag("MTE1_M", (1 + rotation_2) % 3)
                        T.ascend_wait_flag("MTE1_M", (1 + rotation_2) % 3)
                        T.ascend_wait_flag("MTE1_M", 3)
                        T.gemm(
                            a_l0[(1 + rotation_2) % 3, :128, :],
                            b_l0[0, :240, :],
                            c_l0_0_0[:128, :240],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", (1 + rotation_2) % 3)
                        T.ascend_wait_flag("M_MTE1", 4)
                        T.copy(b_l1[rotation_2 % 4, 0:240, 64:128], b_l0[1, :240, :])
                        T.ascend_set_flag("MTE1_M", 4)
                    for rotation_3 in range(2):
                        T.ascend_wait_flag("M_MTE1", (2 * rotation_3) % 3)
                        T.copy(
                            a_l1[
                                0,
                                128 * ((1 + rotation_3) % 2) : (128 * ((1 + rotation_3) % 2)) + 128,
                                64 * (rotation_3 % 2) : (64 * (rotation_3 % 2)) + 64,
                            ],
                            a_l0[(2 * rotation_3) % 3, :128, :],
                        )
                        T.ascend_set_flag("MTE1_M", (2 * rotation_3) % 3)
                    for rotation_4 in range(3):
                        T.ascend_wait_flag("MTE1_MTE2", 5 + rotation_4 % 3)
                        T.copy(
                            B[
                                n_base : n_base + 240,
                                6784 + 128 * rotation_4 : (6784 + 128 * rotation_4) + 128,
                            ],
                            b_l1[1 + rotation_4 % 3, :240, :],
                        )
                        T.ascend_set_flag("MTE2_MTE1", 5 + rotation_4 % 3)
                        T.ascend_wait_flag("MTE1_M", rotation_4 % 3)
                        T.gemm(
                            a_l0[rotation_4 % 3, :128, :],
                            b_l0[0, :240, :],
                            c_l0_1_0[:128, :240],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", rotation_4 % 3)
                        T.ascend_set_flag("M_MTE1", 3)
                        T.ascend_wait_flag("M_MTE1", (1 + rotation_4) % 3)
                        T.copy(
                            a_l1[rotation_4 % 3, 128:256, 64:128],
                            a_l0[(1 + rotation_4) % 3, :128, :],
                        )
                        T.ascend_set_flag("MTE1_M", (1 + rotation_4) % 3)
                        T.ascend_wait_flag("MTE1_M", (2 + rotation_4) % 3)
                        T.ascend_wait_flag("MTE1_M", 4)
                        T.gemm(
                            a_l0[(2 + rotation_4) % 3, :128, :],
                            b_l0[1, :240, :],
                            c_l0_0_0[:128, :240],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", (2 + rotation_4) % 3)
                        T.ascend_wait_flag("MTE1_MTE2", 1 + rotation_4 % 3)
                        T.copy(
                            A[
                                m_base : m_base + 256,
                                6784 + 128 * rotation_4 : (6784 + 128 * rotation_4) + 128,
                            ],
                            a_l1[1 + rotation_4 % 3, :256, :],
                        )
                        T.ascend_set_flag("MTE2_MTE1", 1 + rotation_4 % 3)
                        T.ascend_wait_flag("M_MTE1", 3)
                        T.ascend_wait_flag("MTE2_MTE1", 5 + rotation_4 % 3)
                        T.copy(b_l1[1 + rotation_4 % 3, 0:240, 0:64], b_l0[0, :240, :])
                        T.ascend_set_flag("MTE1_M", 3)
                        T.ascend_wait_flag("MTE1_M", (1 + rotation_4) % 3)
                        T.gemm(
                            a_l0[(1 + rotation_4) % 3, :128, :],
                            b_l0[1, :240, :],
                            c_l0_1_0[:128, :240],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", (1 + rotation_4) % 3)
                        T.ascend_set_flag("M_MTE1", 4)
                        T.ascend_wait_flag("M_MTE1", (2 + rotation_4) % 3)
                        T.ascend_wait_flag("MTE2_MTE1", 1 + rotation_4 % 3)
                        T.copy(
                            a_l1[1 + rotation_4 % 3, 0:128, 0:64],
                            a_l0[(2 + rotation_4) % 3, :128, :],
                        )
                        T.ascend_set_flag("MTE1_M", (2 + rotation_4) % 3)
                        T.ascend_wait_flag("MTE1_M", (2 + rotation_4) % 3)
                        T.ascend_wait_flag("MTE1_M", 3)
                        T.gemm(
                            a_l0[(2 + rotation_4) % 3, :128, :],
                            b_l0[0, :240, :],
                            c_l0_0_0[:128, :240],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", (2 + rotation_4) % 3)
                        T.ascend_wait_flag("M_MTE1", 4)
                        T.copy(b_l1[1 + rotation_4 % 3, 0:240, 64:128], b_l0[1, :240, :])
                        T.ascend_set_flag("MTE1_M", 4)
                        T.ascend_wait_flag("M_MTE1", (1 + rotation_4) % 3)
                        T.copy(
                            a_l1[1 + rotation_4 % 3, 128:256, 0:64],
                            a_l0[(1 + rotation_4) % 3, :128, :],
                        )
                        T.ascend_set_flag("MTE1_M", (1 + rotation_4) % 3)
                        T.ascend_wait_flag("M_MTE1", rotation_4 % 3)
                        T.copy(
                            a_l1[1 + rotation_4 % 3, 0:128, 64:128], a_l0[rotation_4 % 3, :128, :]
                        )
                        T.ascend_set_flag("MTE1_M", rotation_4 % 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_FIX", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_FIX", 1)
                    T.ascend_wait_flag("M_FIX", 0)
                    T.copy(
                        c_l0_0_0[:128, :240],
                        D[m_base + 0 : m_base + 128, n_base + 0 : n_base + 240],
                    )
                    T.ascend_wait_flag("M_FIX", 1)
                    T.copy(
                        c_l0_1_0[:128, :240],
                        D[m_base + 128 : m_base + 256, n_base + 0 : n_base + 240],
                    )
                    T.ascend_pipe_barrier("PIPE_ALL")
                # 192-column N tail; narrower B and L0C slices.
                elif n_tile == 8:
                    T.copy(B[n_base : n_base + 192, 0:128], b_l1[0, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.copy(A[m_base : m_base + 256, 0:128], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.copy(b_l1[0, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.copy(B[n_base : n_base + 192, 128:256], b_l1[1, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 128:256], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    for rotation_0 in range(2):
                        T.ascend_set_flag("MTE1_MTE2", 5 + rotation_0 % 2)
                        T.ascend_wait_flag("M_MTE1", rotation_0 % 2)
                        T.copy(
                            a_l1[1 + rotation_0 % 2, 128:256, 0:64], a_l0[rotation_0 % 2, :128, :]
                        )
                        T.ascend_set_flag("MTE1_M", rotation_0 % 2)
                        T.copy(
                            B[
                                n_base : n_base + 192,
                                256 + 128 * rotation_0 : (256 + 128 * rotation_0) + 128,
                            ],
                            b_l1[2 + rotation_0 % 2, :192, :],
                        )
                        T.ascend_set_flag("MTE2_MTE1", 6 + rotation_0 % 2)
                        T.ascend_wait_flag("M_MTE1", (2 + rotation_0) % 3)
                        T.copy(
                            a_l1[1 + rotation_0 % 2, 0:128, 64:128],
                            a_l0[(2 + rotation_0) % 3, :128, :],
                        )
                        T.ascend_set_flag("MTE1_M", (2 + rotation_0) % 3)
                        T.ascend_wait_flag("MTE1_M", rotation_0 % 2)
                        T.gemm(
                            a_l0[rotation_0 % 2, :128, :],
                            b_l0[0, :192, :],
                            c_l0_1_0[:128, :192],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", rotation_0 % 2)
                        T.ascend_set_flag("M_MTE1", 3)
                        T.ascend_wait_flag("M_MTE1", 1 + rotation_0 % 2)
                        T.copy(
                            a_l1[1 + rotation_0 % 2, 128:256, 64:128],
                            a_l0[1 + rotation_0 % 2, :128, :],
                        )
                        T.ascend_set_flag("MTE1_M", 1 + rotation_0 % 2)
                        T.ascend_set_flag("MTE1_MTE2", 1 + rotation_0 % 2)
                        T.copy(
                            A[
                                m_base : m_base + 256,
                                256 + 128 * rotation_0 : (256 + 128 * rotation_0) + 128,
                            ],
                            a_l1[2 + rotation_0 % 2, :256, :],
                        )
                        T.ascend_set_flag("MTE2_MTE1", 2 + rotation_0 % 2)
                        T.ascend_wait_flag("MTE1_M", (2 + rotation_0) % 3)
                        T.ascend_wait_flag("MTE1_M", 4)
                        T.gemm(
                            a_l0[(2 + rotation_0) % 3, :128, :],
                            b_l0[1, :192, :],
                            c_l0_0_0[:128, :192],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", (2 + rotation_0) % 3)
                        T.ascend_wait_flag("MTE1_M", 1 + rotation_0 % 2)
                        T.gemm(
                            a_l0[1 + rotation_0 % 2, :128, :],
                            b_l0[1, :192, :],
                            c_l0_1_0[:128, :192],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", 1 + rotation_0 % 2)
                        T.ascend_set_flag("M_MTE1", 4)
                        T.ascend_wait_flag("M_MTE1", 3)
                        T.ascend_wait_flag("MTE2_MTE1", 6 + rotation_0 % 2)
                        T.copy(b_l1[2 + rotation_0 % 2, 0:192, 0:64], b_l0[0, :192, :])
                        T.ascend_set_flag("MTE1_M", 3)
                        T.ascend_wait_flag("M_MTE1", (2 + rotation_0) % 3)
                        T.ascend_wait_flag("MTE2_MTE1", 2 + rotation_0 % 2)
                        T.copy(
                            a_l1[2 + rotation_0 % 2, 0:128, 0:64],
                            a_l0[(2 + rotation_0) % 3, :128, :],
                        )
                        T.ascend_set_flag("MTE1_M", (2 + rotation_0) % 3)
                        T.ascend_wait_flag("MTE1_M", (2 + rotation_0) % 3)
                        T.ascend_wait_flag("MTE1_M", 3)
                        T.gemm(
                            a_l0[(2 + rotation_0) % 3, :128, :],
                            b_l0[0, :192, :],
                            c_l0_0_0[:128, :192],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", (2 + rotation_0) % 3)
                        T.ascend_wait_flag("M_MTE1", 4)
                        T.copy(b_l1[2 + rotation_0 % 2, 0:192, 64:128], b_l0[1, :192, :])
                        T.ascend_set_flag("MTE1_M", 4)
                    for rotation_1 in range(49):
                        T.ascend_set_flag("MTE1_MTE2", 4 + (3 + rotation_1) % 4)
                        T.ascend_wait_flag("M_MTE1", (2 + rotation_1) % 3)
                        T.copy(
                            a_l1[(3 + rotation_1) % 4, 128:256, 0:64],
                            a_l0[(2 + rotation_1) % 3, :128, :],
                        )
                        T.ascend_set_flag("MTE1_M", (2 + rotation_1) % 3)
                        T.ascend_wait_flag("MTE1_MTE2", 4 + rotation_1 % 4)
                        T.copy(
                            B[
                                n_base : n_base + 192,
                                512 + 128 * rotation_1 : (512 + 128 * rotation_1) + 128,
                            ],
                            b_l1[rotation_1 % 4, :192, :],
                        )
                        T.ascend_set_flag("MTE2_MTE1", 4 + rotation_1 % 4)
                        T.ascend_wait_flag("M_MTE1", (1 + rotation_1) % 3)
                        T.copy(
                            a_l1[(3 + rotation_1) % 4, 0:128, 64:128],
                            a_l0[(1 + rotation_1) % 3, :128, :],
                        )
                        T.ascend_set_flag("MTE1_M", (1 + rotation_1) % 3)
                        T.ascend_wait_flag("MTE1_M", (2 + rotation_1) % 3)
                        T.gemm(
                            a_l0[(2 + rotation_1) % 3, :128, :],
                            b_l0[0, :192, :],
                            c_l0_1_0[:128, :192],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", (2 + rotation_1) % 3)
                        T.ascend_set_flag("M_MTE1", 3)
                        T.ascend_wait_flag("M_MTE1", rotation_1 % 3)
                        T.copy(
                            a_l1[(3 + rotation_1) % 4, 128:256, 64:128],
                            a_l0[rotation_1 % 3, :128, :],
                        )
                        T.ascend_set_flag("MTE1_M", rotation_1 % 3)
                        T.ascend_set_flag("MTE1_MTE2", (3 + rotation_1) % 4)
                        T.ascend_wait_flag("MTE1_MTE2", rotation_1 % 4)
                        T.copy(
                            A[
                                m_base : m_base + 256,
                                512 + 128 * rotation_1 : (512 + 128 * rotation_1) + 128,
                            ],
                            a_l1[rotation_1 % 4, :256, :],
                        )
                        T.ascend_set_flag("MTE2_MTE1", rotation_1 % 4)
                        T.ascend_wait_flag("MTE1_M", (1 + rotation_1) % 3)
                        T.ascend_wait_flag("MTE1_M", 4)
                        T.gemm(
                            a_l0[(1 + rotation_1) % 3, :128, :],
                            b_l0[1, :192, :],
                            c_l0_0_0[:128, :192],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", (1 + rotation_1) % 3)
                        T.ascend_wait_flag("MTE1_M", rotation_1 % 3)
                        T.gemm(
                            a_l0[rotation_1 % 3, :128, :],
                            b_l0[1, :192, :],
                            c_l0_1_0[:128, :192],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", rotation_1 % 3)
                        T.ascend_set_flag("M_MTE1", 4)
                        T.ascend_wait_flag("M_MTE1", 3)
                        T.ascend_wait_flag("MTE2_MTE1", 4 + rotation_1 % 4)
                        T.copy(b_l1[rotation_1 % 4, 0:192, 0:64], b_l0[0, :192, :])
                        T.ascend_set_flag("MTE1_M", 3)
                        T.ascend_wait_flag("M_MTE1", (1 + rotation_1) % 3)
                        T.ascend_wait_flag("MTE2_MTE1", rotation_1 % 4)
                        T.copy(
                            a_l1[rotation_1 % 4, 0:128, 0:64], a_l0[(1 + rotation_1) % 3, :128, :]
                        )
                        T.ascend_set_flag("MTE1_M", (1 + rotation_1) % 3)
                        T.ascend_wait_flag("MTE1_M", (1 + rotation_1) % 3)
                        T.ascend_wait_flag("MTE1_M", 3)
                        T.gemm(
                            a_l0[(1 + rotation_1) % 3, :128, :],
                            b_l0[0, :192, :],
                            c_l0_0_0[:128, :192],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", (1 + rotation_1) % 3)
                        T.ascend_wait_flag("M_MTE1", 4)
                        T.copy(b_l1[rotation_1 % 4, 0:192, 64:128], b_l0[1, :192, :])
                        T.ascend_set_flag("MTE1_M", 4)
                    for rotation_2 in range(3):
                        T.ascend_wait_flag("M_MTE1", rotation_2 % 3)
                        T.copy(a_l1[rotation_2 % 3, 128:256, 0:64], a_l0[rotation_2 % 3, :128, :])
                        T.ascend_set_flag("MTE1_M", rotation_2 % 3)
                        T.ascend_wait_flag("MTE1_MTE2", 5 + rotation_2 % 3)
                        T.copy(
                            B[
                                n_base : n_base + 192,
                                6784 + 128 * rotation_2 : (6784 + 128 * rotation_2) + 128,
                            ],
                            b_l1[1 + rotation_2 % 3, :192, :],
                        )
                        T.ascend_set_flag("MTE2_MTE1", 5 + rotation_2 % 3)
                        T.ascend_wait_flag("M_MTE1", (2 + rotation_2) % 3)
                        T.copy(
                            a_l1[rotation_2 % 3, 0:128, 64:128], a_l0[(2 + rotation_2) % 3, :128, :]
                        )
                        T.ascend_set_flag("MTE1_M", (2 + rotation_2) % 3)
                        T.ascend_wait_flag("MTE1_M", rotation_2 % 3)
                        T.gemm(
                            a_l0[rotation_2 % 3, :128, :],
                            b_l0[0, :192, :],
                            c_l0_1_0[:128, :192],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", rotation_2 % 3)
                        T.ascend_set_flag("M_MTE1", 3)
                        T.ascend_wait_flag("M_MTE1", (1 + rotation_2) % 3)
                        T.copy(
                            a_l1[rotation_2 % 3, 128:256, 64:128],
                            a_l0[(1 + rotation_2) % 3, :128, :],
                        )
                        T.ascend_set_flag("MTE1_M", (1 + rotation_2) % 3)
                        T.ascend_wait_flag("MTE1_MTE2", 1 + rotation_2 % 3)
                        T.copy(
                            A[
                                m_base : m_base + 256,
                                6784 + 128 * rotation_2 : (6784 + 128 * rotation_2) + 128,
                            ],
                            a_l1[1 + rotation_2 % 3, :256, :],
                        )
                        T.ascend_set_flag("MTE2_MTE1", 1 + rotation_2 % 3)
                        T.ascend_wait_flag("MTE1_M", (2 + rotation_2) % 3)
                        T.ascend_wait_flag("MTE1_M", 4)
                        T.gemm(
                            a_l0[(2 + rotation_2) % 3, :128, :],
                            b_l0[1, :192, :],
                            c_l0_0_0[:128, :192],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", (2 + rotation_2) % 3)
                        T.ascend_wait_flag("MTE1_M", (1 + rotation_2) % 3)
                        T.gemm(
                            a_l0[(1 + rotation_2) % 3, :128, :],
                            b_l0[1, :192, :],
                            c_l0_1_0[:128, :192],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", (1 + rotation_2) % 3)
                        T.ascend_set_flag("M_MTE1", 4)
                        T.ascend_wait_flag("M_MTE1", 3)
                        T.ascend_wait_flag("MTE2_MTE1", 5 + rotation_2 % 3)
                        T.copy(b_l1[1 + rotation_2 % 3, 0:192, 0:64], b_l0[0, :192, :])
                        T.ascend_set_flag("MTE1_M", 3)
                        T.ascend_wait_flag("M_MTE1", (2 + rotation_2) % 3)
                        T.ascend_wait_flag("MTE2_MTE1", 1 + rotation_2 % 3)
                        T.copy(
                            a_l1[1 + rotation_2 % 3, 0:128, 0:64],
                            a_l0[(2 + rotation_2) % 3, :128, :],
                        )
                        T.ascend_set_flag("MTE1_M", (2 + rotation_2) % 3)
                        T.ascend_wait_flag("MTE1_M", (2 + rotation_2) % 3)
                        T.ascend_wait_flag("MTE1_M", 3)
                        T.gemm(
                            a_l0[(2 + rotation_2) % 3, :128, :],
                            b_l0[0, :192, :],
                            c_l0_0_0[:128, :192],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", (2 + rotation_2) % 3)
                        T.ascend_wait_flag("M_MTE1", 4)
                        T.copy(b_l1[1 + rotation_2 % 3, 0:192, 64:128], b_l0[1, :192, :])
                        T.ascend_set_flag("MTE1_M", 4)
                    for rotation_3 in range(2):
                        T.ascend_wait_flag("M_MTE1", (2 * rotation_3) % 3)
                        T.copy(
                            a_l1[
                                3,
                                128 * ((1 + rotation_3) % 2) : (128 * ((1 + rotation_3) % 2)) + 128,
                                64 * (rotation_3 % 2) : (64 * (rotation_3 % 2)) + 64,
                            ],
                            a_l0[(2 * rotation_3) % 3, :128, :],
                        )
                        T.ascend_set_flag("MTE1_M", (2 * rotation_3) % 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_FIX", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_FIX", 1)
                    T.ascend_wait_flag("M_FIX", 0)
                    T.copy(
                        c_l0_0_0[:128, :192],
                        D[m_base + 0 : m_base + 128, n_base + 0 : n_base + 192],
                    )
                    T.ascend_wait_flag("M_FIX", 1)
                    T.copy(
                        c_l0_1_0[:128, :192],
                        D[m_base + 128 : m_base + 256, n_base + 0 : n_base + 192],
                    )
                    T.ascend_pipe_barrier("PIPE_ALL")


def compile_kernel(m: int, n: int, k: int, trans_b: bool, dtype: str):
    """Compile this schedule; it serves one fixed BF16 NT shape."""
    if (m, n, k) != (4096, 2112, 7168) or not trans_b or dtype != "bfloat16":
        raise ValueError(
            f"this schedule serves BF16 NT (4096, 2112, 7168), got {(m, n, k)}, "
            f"trans_b={trans_b}, dtype={dtype}"
        )
    return tilelang.compile(
        main,
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_auto_schedule": False, "tl.disable_shared_memory_reuse": True},
    )
