"""Four output rows per SIMD vector for narrow BF16 volume pooling."""

import functools

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S


def supports(window, dtype):
    return (
        window.rows == 48
        and window.size == (10, 20, 22)
        and window.kernel == (2, 2, 3)
        and window.stride == (2, 2, 2)
        and window.pad == (0, 1, 1)
        and window.out == (5, 11, 11)
        and not window.ceil_mode
        and window.count_include_pad
        and window.divisor_override == 7
        and dtype == "bfloat16"
    )


@functools.lru_cache(maxsize=1)
def build(window, dtype, stages=2):
    assert window.rows == 48 and window.size == (10, 20, 22) and window.kernel == (2, 2, 3)
    assert window.stride == (2, 2, 2) and window.pad == (0, 1, 1) and window.out == (5, 11, 11)
    assert not window.ceil_mode and window.count_include_pad and window.divisor_override == 7
    assert dtype == "bfloat16" and stages == 2

    @T.prim_func
    def main(X: T.Tensor((211200,), "bfloat16"), Y: T.Tensor((240, 11, 11), "bfloat16")):
        with T.Kernel(64) as core:
            raw = T.alloc_shared((896,), "bfloat16")
            data = T.alloc_shared((896,), "float32")
            output = T.alloc_shared((12, 16), "bfloat16")
            T.annotate_buffer_versions({raw: 2, data: 1, output: 2})
            for step in T.Pipelined(4, num_stages=2):
                job = step * 64 + core
                if job < 240:
                    begin = job * 880
                    T.copy(X[begin : begin + 880], raw[:880])
                    with T.SimdVF():
                        full = S.pset(32)
                        zero = S.vdup(T.float32(0), "float32", full)
                        izero = S.vdup(T.int32(0), "int32", full)
                        for v in T.serial(14):
                            val = S.vcvt(
                                S.vld(raw[v * 64], dist="UNPK_B16"), "float32", full, part=0
                            )
                            S.vsts(data[v * 64], val, full)
                        S.mem_bar("VST_VLD")
                        lane = S.vci(0, "int32")
                        local_y = S.vshrs(lane, T.int32(4), full)
                        ox = S.vsub(lane, S.vmuls(local_y, T.int32(16), full), full)
                        x0 = S.vadds(S.vmuls(ox, T.int32(2), full), T.int32(-1), full)
                        offsets = S.alloc_local((6,), "uint32")
                        masks = S.alloc_local((6,), "bool")
                        acc = S.alloc_local((1,), "float32")
                        for group in T.serial(3):
                            oy = S.vadds(local_y, T.cast(group * 4, "int32"), full)
                            y0 = S.vadds(S.vmuls(oy, T.int32(2), full), T.int32(-1), full)
                            for ky in T.Unroll(2, explicit=True):
                                iy = S.vadds(y0, T.int32(ky), full)
                                my = S.pand(
                                    S.vcmps(iy, T.int32(0), full, "ge"),
                                    S.vcmps(iy, T.int32(20), full, "lt"),
                                    full,
                                )
                                for kx in T.Unroll(3, explicit=True):
                                    ix = S.vadds(x0, T.int32(kx), full)
                                    mx = S.pand(
                                        S.vcmps(ix, T.int32(0), full, "ge"),
                                        S.vcmps(ix, T.int32(22), full, "lt"),
                                        full,
                                    )
                                    masks[ky * 3 + kx] = S.pand(my, mx, full)
                                    offset = S.vadd(S.vmuls(iy, T.int32(22), full), ix, full)
                                    offsets[ky * 3 + kx] = T.reinterpret(
                                        S.vsel(offset, izero, masks[ky * 3 + kx]), "uint32x64"
                                    )
                            acc[0] = zero
                            for dz in T.Unroll(2, explicit=True):
                                for tap in T.Unroll(6, explicit=True):
                                    offset = S.vadds(offsets[tap], T.uint32(dz * 440), full)
                                    val = S.vgather2(data[0], offset, full)
                                    acc[0] = S.vadd(acc[0], S.vsel(val, zero, masks[tap]), full)
                            result = S.vmuls(acc[0], T.float32(1.0 / 7), full)
                            S.vsts(
                                output[group * 4, 0],
                                S.vcvt(result, "bfloat16", full, sat=False),
                                full,
                                "PK_B32",
                            )
                    T.copy(output[:11, :11], Y[job, :, :])

    return tilelang.compile(
        main.with_attr("tilelang_callee_allocated_outputs", 0), target="ascend", out_idx=-1
    )
