"""Bounded-memory INT8 rows with separate maximum and quantization launches."""

import functools

import tilelang
import tilelang.ascend.language as T

from tileops.kernels.ascend.quantization.int8_scale import int8_divide, int8_scale


@functools.lru_cache(maxsize=64)
def build_int8_row_stream(rows: int, k: int, dtype: str, smoothed: bool):
    cols = 8192
    row_blocks = min(64, rows)
    chunks = (k + cols - 1) // cols
    total = rows * chunks
    tile_blocks = min(64, total)

    @T.macro
    def stage(X, Smooth, raw_input, smooth_values, magnitude, row, column, count):
        T.copy(X[row * k + column : row * k + column + count], raw_input[:count])
        if smoothed:
            T.copy(Smooth[column : column + count], smooth_values[:count])
        with T.SimdVF():
            full = T.simd.pset(32)
            for v in T.serial(cols // 64):
                valid = T.simd.vcmps(T.simd.vci(v * 64, "int32"), count, full, op="lt")
                if dtype == "float32":
                    wide_input = T.simd.vld(raw_input[v * 64])
                else:
                    wide_input = T.simd.vcvt(
                        T.simd.vld(raw_input[v * 64], dist="UNPK_B16"),
                        "float32",
                        part=0,
                    )
                staged_input = T.simd.vsel(wide_input, T.simd.vdup(T.float32(0), "float32"), valid)
                T.simd.vsts(magnitude[v * 64], staged_input, full)
                if smoothed:
                    staged_smooth = T.simd.vsel(
                        T.simd.vld(smooth_values[v * 64]),
                        T.simd.vdup(T.float32(1), "float32"),
                        valid,
                    )
                    T.simd.vsts(smooth_values[v * 64], staged_smooth, full)
        if smoothed:
            with T.SimdVF():
                full = T.simd.pset(32)
                for v in T.serial(cols // 64):
                    values = T.simd.vdiv(
                        T.simd.vld(magnitude[v * 64]),
                        T.simd.vld(smooth_values[v * 64]),
                        full,
                        precision="exact",
                    )
                    T.simd.vsts(magnitude[v * 64], values, full)
                T.simd.mem_bar("VST_VLD")

    @T.prim_func
    def maximum(
        X: T.Tensor((rows * k,), dtype),
        Smooth: T.Tensor((k,), "float32"),
        Peaks: T.Tensor((rows,), "float32"),
    ):
        with T.Kernel(row_blocks) as bx:
            magnitude = T.alloc_shared((cols,), "float32")
            raw_input = T.alloc_shared((cols,), dtype)
            smooth_values = T.alloc_shared((cols if smoothed else 1,), "float32")
            peak = T.alloc_shared((1,), "float32")
            for step in T.serial(T.ceildiv(rows, row_blocks)):
                row = step * row_blocks + bx
                if row < rows:
                    with T.SimdVF():
                        one = T.simd.pset(32, "PAT_VL1")
                        T.simd.vsts(peak[0], T.simd.vdup(T.float32(0), "float32"), one, "ONEPT_B32")
                    for chunk in T.serial(chunks):
                        column = chunk * cols
                        count = T.min(cols, k - column)
                        stage(X, Smooth, raw_input, smooth_values, magnitude, row, column, count)
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            one = T.simd.pset(32, "PAT_VL1")
                            acc = T.simd.alloc_local((1,), "int32")
                            acc[0] = T.simd.vcmax(
                                T.reinterpret(
                                    T.simd.vabs(T.simd.vld(magnitude[0]), full), "int32x64"
                                ),
                                full,
                            )
                            for v in T.serial(1, cols // 64):
                                acc[0] = T.simd.vmax(
                                    acc[0],
                                    T.simd.vcmax(
                                        T.reinterpret(
                                            T.simd.vabs(T.simd.vld(magnitude[v * 64]), full),
                                            "int32x64",
                                        ),
                                        full,
                                    ),
                                    full,
                                )
                            acc[0] = T.simd.vmax(
                                acc[0],
                                T.reinterpret(T.simd.vdup(peak[0], "float32"), "int32x64"),
                                full,
                            )
                            T.simd.vsts(
                                peak[0],
                                T.reinterpret(acc[0], "float32x64"),
                                one,
                                "ONEPT_B32",
                            )
                    T.copy(peak, Peaks[row : row + 1])

    @T.prim_func
    def quantize(
        X: T.Tensor((rows * k,), dtype),
        Smooth: T.Tensor((k,), "float32"),
        Peaks: T.Tensor((rows,), "float32"),
        Q: T.Tensor((rows * k,), "int8"),
        Scale: T.Tensor((rows,), "float32"),
    ):
        with T.Kernel(tile_blocks) as bx:
            magnitude = T.alloc_shared((cols,), "float32")
            raw_input = T.alloc_shared((cols,), dtype)
            smooth_values = T.alloc_shared((cols if smoothed else 1,), "float32")
            peak = T.alloc_shared((1,), "float32")
            factor = T.alloc_shared((1,), "float32")
            tiny = T.alloc_shared((1,), "int32")
            normalized_scale = T.alloc_shared((1,), "float32")
            codes = T.alloc_shared((cols,), "int8")
            for step in T.serial(T.ceildiv(total, tile_blocks)):
                tile = step * tile_blocks + bx
                if tile < total:
                    row = tile // chunks
                    column = tile % chunks * cols
                    count = T.min(cols, k - column)
                    T.copy(Peaks[row : row + 1], peak)
                    stage(X, Smooth, raw_input, smooth_values, magnitude, row, column, count)
                    int8_scale(peak, factor, tiny, normalized_scale)
                    int8_divide(magnitude, factor, tiny, normalized_scale, cols)
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        for v in T.serial(cols // 64):
                            quotient = T.simd.vld(magnitude[v * 64])
                            clipped = T.simd.vmin(
                                T.simd.vmax(
                                    quotient,
                                    T.simd.vdup(T.float32(-127), "float32"),
                                    full,
                                ),
                                T.simd.vdup(T.float32(127), "float32"),
                                full,
                            )
                            integers = T.simd.vcvt(clipped, "int32", full, round="ROUND_R")
                            packed = T.simd.vcvt(
                                T.simd.vand(integers, T.simd.vdup(T.int32(255), "int32"), full),
                                "uint8",
                                full,
                                sat=False,
                                part=0,
                            )
                            T.simd.vsts(
                                codes[v * 64],
                                T.reinterpret(packed, "int8x256"),
                                full,
                                "PK4_B32",
                            )
                    T.copy(codes[:count], Q[row * k + column : row * k + column + count])
                    if column == 0:
                        T.copy(factor, Scale[row : row + 1])

    maximum_kernel = tilelang.compile(maximum, target="ascend", out_idx=2)
    quantize_kernel = tilelang.compile(quantize, target="ascend", out_idx=[3, 4])

    def run(x, smooth):
        peaks = maximum_kernel(x, smooth)
        return quantize_kernel(x, smooth, peaks)

    return run
