"""Exact ECG-CNN BF16 max1d specialization selected by the MaxPool constructor.

N16,C128,L2048,k5,s2,p2,d2,ceil=True,out1023. Values and flat int64
indices retain the validated ordered selection: first finite ties, last NaN.
"""

import functools


def supports_call(call):
    return (
        call.n == 16
        and call.c_in == 128
        and tuple(call.size) == (2048,)
        and tuple(call.window) == (5,)
        and tuple(call.stride) == (2,)
        and tuple(call.pad) == (2,)
        and tuple(call.dilation) == (2,)
        and call.ceil_mode is True
        and str(call.dtype).split(".")[-1] == "bfloat16"
    )


def program(indices=False):
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    rows, width, outputs, padded_outputs, row_tile = 2048, 2048, 1023, 1024, 4
    tiles = rows // row_tile
    index_size = rows * outputs if indices else 1
    pick_width = padded_outputs if indices else 1

    @T.prim_func
    def main(
        X: T.Tensor((rows * width,), "bfloat16"),
        Y: T.Tensor((rows * outputs,), "bfloat16"),
        I: T.Tensor((index_size,), "int64"),
    ):
        with T.Kernel(64) as core:
            raw = T.alloc_shared((row_tile * width,), "bfloat16")
            data = T.alloc_shared((row_tile * width,), "float32")
            output = T.alloc_shared((row_tile, padded_outputs), "bfloat16")
            picks = T.alloc_shared((row_tile, pick_width), "int32")
            for step in T.serial(tiles // 64):
                tile = step * 64 + core
                begin = tile * row_tile * width
                T.copy(X[begin : begin + row_tile * width], raw)
                with T.SimdVF():
                    full = S.pset(32)
                    for part in T.serial(row_tile * width // 64):
                        converted = S.vcvt(
                            S.vld(raw[part * 64], "UNPK_B16"), "float32", full, part=0
                        )
                        S.vsts(data[part * 64], converted, full)
                    S.mem_bar("VST_VLD")
                    zero = S.vdup(T.int32(0), "int32", full)
                    value = S.alloc_local((1,), "float32")
                    pick = S.alloc_local((1,), "int32")
                    for r in T.serial(row_tile):
                        for v in T.serial(padded_outputs // 64):
                            lane = S.vci(T.cast(v * 64, "int32"), "int32")
                            active = S.vcmps(lane, T.int32(outputs), full, "lt")
                            starts = S.vadds(S.vmuls(lane, T.int32(2), full), T.int32(-2), full)
                            value[0] = S.vdup(T.float32(float("-inf")), "float32", full)
                            if indices:
                                pick[0] = S.vdup(T.int32(-1), "int32", full)
                            for k in T.Unroll(5, explicit=True):
                                position = S.vadds(starts, T.int32(k * 2), full)
                                valid = S.pand(
                                    active,
                                    S.pand(
                                        S.vcmps(position, T.int32(0), full, "ge"),
                                        S.vcmps(position, T.int32(width), full, "lt"),
                                        full,
                                    ),
                                    full,
                                )
                                offset = T.reinterpret(S.vsel(position, zero, valid), "uint32x64")
                                incoming = S.vgather2(data[r * width], offset, full)
                                better = S.por(
                                    S.vcmp(incoming, value[0], full, "gt"),
                                    S.vcmp(incoming, incoming, full, "ne"),
                                    full,
                                )
                                if indices:
                                    take = S.pand(
                                        valid,
                                        S.por(
                                            better,
                                            S.vcmps(pick[0], T.int32(0), full, "lt"),
                                            full,
                                        ),
                                        full,
                                    )
                                else:
                                    take = S.pand(valid, better, full)
                                value[0] = S.vsel(incoming, value[0], take)
                                if indices:
                                    pick[0] = S.vsel(position, pick[0], take)
                            narrowed = S.vcvt(value[0], "bfloat16", full, sat=False)
                            S.vsts(output[r, v * 64], narrowed, full, "PK_B32")
                            if indices:
                                S.vsts(picks[r, v * 64], pick[0], full)
                for r in T.serial(row_tile):
                    out_begin = (tile * row_tile + r) * outputs
                    T.copy(output[r, :outputs], Y[out_begin : out_begin + outputs])
                if indices:
                    with T.SimtVF(threads=256):
                        for r, col in T.Parallel(row_tile, outputs):
                            I[(tile * row_tile + r) * outputs + col] = T.cast(
                                picks[r, col], "int64"
                            )

    return main.with_attr("tilelang_callee_allocated_outputs", 0)


@functools.lru_cache(maxsize=2)
def build(indices=False):
    import tilelang

    return tilelang.compile(program(indices), target="ascend", out_idx=[1, 2])
