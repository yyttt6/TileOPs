"""Full-row DMA/SIMD average-pool1d for regular temporal shapes.

The API comes from the parent's mirrored installed SIMD module. Dynamic counts
are converted with vcvt rather than a packed SIMT integer-to-float cast.
"""

import functools

import tilelang
import tilelang.ascend.language as T


def supports_rows(window, dtype):
    if len(window.size) != 1 or not window.rows:
        return False
    (width,) = window.size
    (outputs,) = window.out
    item = 4 if dtype == "float32" else 2
    # Includes both raw/wide buffers even though raw is dead for float32.
    declared_ub = width * (item + 4) + outputs * item
    return (
        width > 0
        and width % 64 == 0
        and outputs > 0
        and outputs % 64 == 0
        and window.kernel[0] <= 9
        and window.stride[0] <= 8
        and declared_ub <= 208 * 1024
        and window.rows * max(width, outputs) < 2**31
    )


@functools.lru_cache(maxsize=128)
def build_rows(window, dtype):
    """Pool all outputs from one input row in each persistent task.

    Interior vectors need neither masks nor count calculations. Only boundary
    vectors take the masked-gather and explicit divisor-count path. The input
    row is copied exactly once, with no negative GM origin or OOB DMA region.
    """
    assert supports_rows(window, dtype)
    (width,) = window.size
    (outputs,) = window.out
    (kw,) = window.kernel
    (sw,) = window.stride
    (pw,) = window.pad
    rows = window.rows
    blocks = min(rows, 64)
    steps = (rows + blocks - 1) // blocks
    lower, upper = window.overlap[0][0], window.overlap[1][0]
    fixed = window.divisor_override is not None or window.whole_window_divides
    divisor = window.divisor

    @T.prim_func
    def main(X: T.Tensor((rows, width), dtype), Y: T.Tensor((rows, outputs), dtype)):
        with T.Kernel(blocks) as core:
            raw = T.alloc_shared((width,), dtype)
            data = T.alloc_shared((width,), "float32")
            output = T.alloc_shared((outputs,), dtype)
            for step in T.serial(steps):
                row = step * blocks + core
                if row < rows:
                    if dtype == "float32":
                        T.copy(X[row, :], data)
                    else:
                        T.copy(X[row, :], raw)
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        if dtype != "float32":
                            for v in T.serial(width // 64):
                                raw_value = T.simd.vld(raw[v * 64], dist="UNPK_B16")
                                converted = T.simd.vcvt(raw_value, "float32", part=0)
                                T.simd.vsts(data[v * 64], converted, full)
                            T.simd.mem_bar("VST_VLD")
                        lane = T.simd.vci(0, "int32")
                        step_indices = T.simd.vmuls(lane, T.int32(sw), full)
                        zero = T.simd.vdup(T.float32(0), "float32", full)
                        izero = T.simd.vdup(T.int32(0), "int32", full)
                        acc = T.simd.alloc_local((1,), "float32")
                        average = T.simd.alloc_local((1,), "float32")
                        for v in T.serial(outputs // 64):
                            start = v * 64 * sw - pw
                            starts = T.simd.vadds(step_indices, start, full)
                            acc[0] = zero
                            if start >= 0 and start + 63 * sw + kw <= width:
                                for tap in T.unroll(kw):
                                    indices = T.simd.vadds(starts, T.int32(tap), full)
                                    value = T.simd.vgather2(
                                        data[0],
                                        T.reinterpret(indices, "uint32x64"),
                                        full,
                                    )
                                    acc[0] = T.simd.vadd(acc[0], value, full)
                                average[0] = T.simd.vmuls(acc[0], T.float32(1.0 / divisor), full)
                            else:
                                for tap in T.unroll(kw):
                                    indices = T.simd.vadds(starts, T.int32(tap), full)
                                    valid = T.simd.pand(
                                        T.simd.vcmps(indices, T.int32(0), full, op="ge"),
                                        T.simd.vcmps(indices, T.int32(width), full, op="lt"),
                                        full,
                                    )
                                    safe = T.simd.vsel(indices, izero, valid)
                                    value = T.simd.vgather2(
                                        data[0],
                                        T.reinterpret(safe, "uint32x64"),
                                        full,
                                    )
                                    acc[0] = T.simd.vadd(
                                        acc[0],
                                        T.simd.vsel(value, zero, valid),
                                        full,
                                    )
                                if fixed:
                                    average[0] = T.simd.vmuls(
                                        acc[0], T.float32(1.0 / divisor), full
                                    )
                                else:
                                    ends = T.simd.vadds(starts, T.int32(kw), full)
                                    count = T.simd.vmaxs(
                                        T.simd.vsub(
                                            T.simd.vmins(ends, T.int32(upper), full),
                                            T.simd.vmaxs(starts, T.int32(lower), full),
                                            full,
                                        ),
                                        T.int32(1),
                                        full,
                                    )
                                    denominator = T.simd.vcvt(count, "float32", full)
                                    average[0] = T.simd.vdiv(acc[0], denominator, full)
                            if dtype == "float32":
                                T.simd.vsts(output[v * 64], average[0], full)
                            else:
                                narrowed = T.simd.vcvt(average[0], dtype, sat=False)
                                T.simd.vsts(output[v * 64], narrowed, full, dist="PK_B32")
                    T.copy(output, Y[row, :])

    main = main.with_attr("tilelang_callee_allocated_outputs", 0)
    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=128)
def build_audio_native16(window, dtype):
    """Optional audio variant: gather/deinterleave in 16-bit UB, sum in FP32.

    Widen even/odd lanes separately, then interleave FP32 results before the
    final cast. This avoids the entire FP32 row staging pass. Kept outside
    default dispatch until the parent compares it against audio_dma_simd.
    """
    assert dtype in ("float16", "bfloat16")
    assert window.kernel == (3,) and window.stride == (2,) and window.pad == (1,)
    assert not window.ceil_mode and window.count_include_pad
    assert window.divisor_override is None
    (width,) = window.size
    (outputs,) = window.out
    assert width % 256 == 0 and width <= 8192
    rows = window.rows
    blocks = min(64, rows)

    @T.prim_func
    def main(X: T.Tensor((rows, width), dtype), Y: T.Tensor((rows, outputs), dtype)):
        with T.Kernel(blocks) as core:
            raw = T.alloc_shared((width,), dtype)
            output = T.alloc_shared((outputs,), dtype)
            for step in T.serial((rows + blocks - 1) // blocks):
                row = step * blocks + core
                if row < rows:
                    T.copy(X[row, :], raw)
                    with T.SimdVF():
                        full16 = T.simd.pset(16)
                        full32 = T.simd.pset(32)
                        lane_offsets = T.simd.vmuls(
                            T.simd.vci(0, "int16"),
                            T.int16(2),
                            full16,
                        )
                        zero16 = T.simd.vdup(T.cast(0, dtype), dtype, full16)
                        zero32 = T.simd.vdup(T.float32(0), "float32", full32)
                        for v in T.serial(outputs // 128):
                            center, right = T.simd.vld2(raw[v * 256], dist="DINTLV_B16")
                            left_indices = T.simd.vadds(
                                lane_offsets,
                                T.cast(v * 256 - 1, "int16"),
                                full16,
                            )
                            valid_left = T.simd.vcmps(left_indices, T.int16(0), full16, "ge")
                            safe = T.simd.vmaxs(left_indices, T.int16(0), full16)
                            gathered = T.simd.vgather2(
                                raw[0],
                                T.reinterpret(safe, "uint16x128"),
                                full16,
                            )
                            left = T.simd.vsel(gathered, zero16, valid_left)
                            left_even = T.simd.vcvt(left, "float32", full16, part=0)
                            center_even = T.simd.vcvt(center, "float32", full16, part=0)
                            right_even = T.simd.vcvt(right, "float32", full16, part=0)
                            left_odd = T.simd.vcvt(left, "float32", full16, part=1)
                            center_odd = T.simd.vcvt(center, "float32", full16, part=1)
                            right_odd = T.simd.vcvt(right, "float32", full16, part=1)
                            even01 = T.simd.vadd(
                                T.simd.vadd(zero32, left_even, full32),
                                center_even,
                                full32,
                            )
                            odd01 = T.simd.vadd(
                                T.simd.vadd(zero32, left_odd, full32),
                                center_odd,
                                full32,
                            )
                            even_sum = T.simd.vadd(even01, right_even, full32)
                            odd_sum = T.simd.vadd(odd01, right_odd, full32)
                            even_avg = T.simd.vmuls(even_sum, T.float32(1.0 / 3.0), full32)
                            odd_avg = T.simd.vmuls(odd_sum, T.float32(1.0 / 3.0), full32)
                            low, high = T.simd.vintlv(even_avg, odd_avg)
                            low16 = T.simd.vcvt(low, dtype, sat=False)
                            high16 = T.simd.vcvt(high, dtype, sat=False)
                            T.simd.vsts(output[v * 128], low16, full32, "PK_B32")
                            T.simd.vsts(output[v * 128 + 64], high16, full32, "PK_B32")
                    T.copy(output, Y[row, :])

    main = main.with_attr("tilelang_callee_allocated_outputs", 0)
    return tilelang.compile(main, target="ascend", out_idx=-1)
