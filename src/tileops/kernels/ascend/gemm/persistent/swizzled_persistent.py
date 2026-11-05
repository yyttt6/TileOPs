"""Persistent GEMM with row-window swizzling and L1-direct unit flags."""

import tilelang
import tilelang.ascend.language as T

# One entry per served shape; every field names a value the schedule fixes per case.
_CONFIGS = {
    (4096, 7168, 2048): {"tile_m": 256, "tile_n": 256},
    (4096, 4096, 7168): {"tile_m": 256, "tile_n": 256},
    (4096, 7168, 16384): {"tile_m": 256, "tile_n": 256},
    (4096, 24576, 1536): {"tile_m": 256, "tile_n": 256},
    (64, 7168, 2048): {"tile_m": 64, "tile_n": 256},
    (96, 2112, 7168): {"tile_m": 96, "tile_n": 96},
    (32, 24576, 4096): {"tile_m": 32, "tile_n": 256},
}


def build(m: int, n: int, k: int, trans_b: bool, dtype: str, tile_m: int, tile_n: int):
    """Build one ``D = A @ B.T`` for shapes that are whole multiples of their tiles.

    Tiles are walked in a window of up to four M rows; every other row window
    walks N backwards, so neighbouring cores share the same weight tiles in L2.
    """
    tile_k = 256
    num_blocks = 32
    num_stages = 2
    if not trans_b:
        raise ValueError("this schedule serves trans_b=True only")
    if dtype not in ("bfloat16", "float16"):
        raise ValueError(f"this schedule serves 16-bit floats, got {dtype}")
    if any(type(value) is not int or value <= 0 for value in (m, n, k, tile_m, tile_n)):
        raise ValueError("dimensions and physical tiles must be positive integers")
    if any(tile % 16 for tile in (tile_m, tile_n, tile_k)):
        raise ValueError("16-aligned physical tiles required")
    if m % tile_m or n % tile_n or k % tile_k:
        raise ValueError("this schedule requires M, N and K to be whole multiples of their tiles")
    if 2 * (tile_m + tile_n) * tile_k * num_stages > 512 * 1024:
        raise ValueError("L1 operand versions exceed the 512 KiB capacity")
    if 4 * tile_m * tile_n > 256 * 1024:
        raise ValueError("FP32 accumulator exceeds the 256 KiB L0C capacity")
    m_tiles = m // tile_m
    n_tiles = n // tile_n
    k_tiles = k // tile_k
    out_tiles = m_tiles * n_tiles
    window = min(4, m_tiles)
    main_row = m_tiles // window - 1
    tail_window = m_tiles - main_row * window
    unit_flag_2 = 2
    unit_flag_3 = 3

    # Serpentine N order per M-row window, so neighbouring cores share weight tiles.
    @T.macro
    def swizzle(tile_idx):
        m_tile = T.alloc_var("int32")
        n_tile = T.alloc_var("int32")
        row_idx = tile_idx // n_tiles // window
        if row_idx < main_row:
            m_tile = row_idx * window + tile_idx % window
            n_tile = tile_idx // window % n_tiles
        else:
            tail_idx = tile_idx - main_row * window * n_tiles
            m_tile = main_row * window + tail_idx % tail_window
            n_tile = tail_idx // tail_window % n_tiles
        if row_idx % 2 != 0:
            n_tile = n_tiles - 1 - n_tile
        return (m_tile, n_tile)

    @T.prim_func
    def main(
        A: T.Tensor((m, k), dtype),
        B: T.Tensor((n, k), dtype),
        D: T.Tensor((m, n), dtype),
    ):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(num_blocks) as bx:
            acc = T.alloc_l0c((tile_m, tile_n), "float32")
            a_l1 = T.alloc_l1((tile_m, tile_k), dtype)
            b_l1 = T.alloc_l1((tile_n, tile_k), dtype)
            # Persistent tiles in swizzled order; both operands come from L1.
            for tile_idx in T.Persistent([out_tiles], num_blocks, bx):
                m_tile, n_tile = swizzle(tile_idx)
                for kt in T.Pipelined(k_tiles, num_stages=num_stages):
                    T.copy(
                        A[
                            m_tile * tile_m : (m_tile + 1) * tile_m,
                            kt * tile_k : (kt + 1) * tile_k,
                        ],
                        a_l1,
                    )
                    T.copy(
                        B[
                            n_tile * tile_n : (n_tile + 1) * tile_n,
                            kt * tile_k : (kt + 1) * tile_k,
                        ],
                        b_l1,
                    )
                    T.gemm(
                        a_l1,
                        b_l1,
                        acc,
                        transpose_B=True,
                        clear_accum=kt == 0,
                        unit_flag_ctrl=T.Select(kt == k_tiles - 1, unit_flag_3, unit_flag_2),
                    )
                T.copy(acc, D[m_tile * tile_m, n_tile * tile_n], unit_flag_ctrl=unit_flag_3)

    return main


def compile_kernel(m: int, n: int, k: int, trans_b: bool, dtype: str):
    """Compile the schedule configured for ``(m, n, k)``, its layout and dtype."""
    return tilelang.compile(
        build(m, n, k, trans_b=trans_b, dtype=dtype, **_CONFIGS[m, n, k]),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )
