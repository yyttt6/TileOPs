"""W4A16 contraction whose entire K tile fits on chip, in original activation order."""

import tilelang
import tilelang.ascend.language as T


def fits(m: int, n: int, k: int, tile_m: int, tile_n: int) -> bool:
    """Whole-operand capacity, packing, and output-grid guards."""
    if min(m, n, k, tile_m, tile_n) <= 0:
        return False
    if k % 128 or n % tile_n or tile_m % 16 or tile_n % 16:
        return False
    return (
        2 * (tile_m + tile_n) * k <= 512 * 1024
        and 4 * tile_m * tile_n <= 256 * 1024
        and tile_n * (k // 2 + 6 * k + 3 * (k // 128)) + 512 <= 176 * 1024
    )


def build(m: int, n: int, k: int, tile_m: int, tile_n: int, blocks: int):
    """Load/dequantize whole K once per output tile, with one CV handoff."""
    if not fits(m, n, k, tile_m, tile_n):
        raise ValueError("whole contraction violates physical tile or on-chip capacity constraints")
    groups = k // 128
    paired = k >= 512 and k % 256 == 0

    @T.macro
    def dequantize_paired_group(stream, scale, zero, weight, row, group, part: int):
        mask_all = T.simd.pset(8, "PAT_ALL")
        mask16 = T.simd.pset(16, "PAT_VL128")
        q = T.simd.vcvt(stream, "float16", mask_all, part=part)
        z = T.simd.vcvt(T.simd.vld(zero[row, group], "BRC_B8"), "float16", mask_all, part=0)
        factor = T.simd.vld(scale[row, group], "BRC_B16")
        value = T.simd.vmul(T.simd.vsub(q, z, mask16), factor, mask16)
        T.simd.vsts(weight[row, group * 128], value, mask16, "NORM_B16")

    @T.prim_func
    def main(
        A: T.Tensor((m, k), "float16"),
        P: T.Tensor((n, k // 2), "uint8"),
        S: T.Tensor((n, groups), "float16"),
        Z: T.Tensor((n, groups), "uint8"),
        Order: T.Tensor((128,), "uint16"),
        D: T.Tensor((m, n), "float16"),
    ):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.MixedKernel(blocks) as (bx, sid):
            a_l1 = T.alloc_l1((tile_m, k), "float16")
            b_l1 = T.alloc_l1((tile_n, k), "float16")
            acc = T.alloc_l0c((tile_m, tile_n), "float32")
            packed = T.alloc_shared((tile_n, k // 2), "uint8")
            scale = T.alloc_shared((tile_n, groups), "float16")
            zero = T.alloc_shared((tile_n, groups), "uint8")
            weight = T.alloc_shared((tile_n, k), "float16")
            if not paired:
                natural = T.alloc_shared((tile_n, k), "float16")
                order = T.alloc_shared((128,), "uint16")
                if sid == 0:
                    T.copy(Order, order)
            for index in T.serial(bx, T.ceildiv(m, tile_m) * (n // tile_n), blocks):
                first_m = index // (n // tile_n) * tile_m
                first_n = index % (n // tile_n) * tile_n
                rows = T.min(tile_m, m - first_m)
                T.copy(A[first_m : first_m + rows, :], a_l1[:rows, :])
                if sid == 0:
                    T.copy(P[first_n : first_n + tile_n, :], packed)
                    T.copy(S[first_n : first_n + tile_n, :], scale)
                    T.copy(Z[first_n : first_n + tile_n, :], zero)
                    if paired:
                        with T.SimdVF():
                            mask8 = T.simd.pset(8, "PAT_VL128")
                            fifteen = T.simd.vdup(T.uint8(15), "uint8")
                            for row in T.serial(tile_n):
                                for pair in T.serial(k // 256):
                                    raw = T.simd.vld(packed[row, pair * 128], "NORM_B8")
                                    low = T.simd.vand(raw, fifteen, mask8)
                                    high = T.simd.vshrs(raw, 4, mask8)
                                    stream, _ = T.simd.vintlv(
                                        T.reinterpret(low, "uint8x256"),
                                        T.reinterpret(high, "uint8x256"),
                                    )
                                    dequantize_paired_group(
                                        stream, scale, zero, weight, row, 2 * pair, 0
                                    )
                                    dequantize_paired_group(
                                        stream, scale, zero, weight, row, 2 * pair + 1, 1
                                    )
                        T.copy(weight, b_l1)
                    else:
                        with T.SimdVF():
                            mask8 = T.simd.pset(8, "PAT_VL64")
                            mask16 = T.simd.pset(16, "PAT_VL128")
                            fifteen = T.simd.vdup(T.uint8(15), "uint8")
                            for row in T.serial(tile_n):
                                for group in T.serial(groups):
                                    raw = T.simd.vld(packed[row, group * 64], "NORM_B8")
                                    low = T.simd.vand(raw, fifteen, mask8)
                                    high = T.simd.vand(T.simd.vshrs(raw, 4, mask8), fifteen, mask8)
                                    stream, _ = T.simd.vintlv(
                                        T.reinterpret(low, "uint8x256"),
                                        T.reinterpret(high, "uint8x256"),
                                    )
                                    z = T.simd.vunpack(
                                        T.simd.vld(zero[row, group], "BRC_B8"), part=0
                                    )
                                    q = T.simd.vunpack(stream, part=0)
                                    centered = T.simd.vsub(q, z, mask16)
                                    wide = T.reinterpret(centered, "int16x128")
                                    factor = T.simd.vld(scale[row, group], "BRC_B16")
                                    value = T.simd.vmul(
                                        T.simd.vcvt(wide, "float16", mask16), factor, mask16
                                    )
                                    T.simd.vsts(weight[row, group * 128], value, mask16, "NORM_B16")
                        with T.SimdVF():
                            inverse = T.simd.vld(order[0], "NORM_B16")
                            for row in T.serial(tile_n):
                                for group in T.serial(groups):
                                    gathered = T.simd.vgather2(weight[row, group * 128], inverse)
                                    T.simd.vsts(
                                        natural[row, group * 128], gathered, dist="NORM_B16"
                                    )
                        T.copy(natural, b_l1)
                T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True, unit_flag_ctrl=3)
                T.copy(
                    acc[:rows, :],
                    D[first_m : first_m + rows, first_n : first_n + tile_n],
                    unit_flag_ctrl=3,
                )

    return main


def compile_kernel(m: int, n: int, k: int, tile_m: int, tile_n: int, blocks: int):
    return tilelang.compile(
        build(m, n, k, tile_m, tile_n, blocks),
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_fast_math": False},
    )
