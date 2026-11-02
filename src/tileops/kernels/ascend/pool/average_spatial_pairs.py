"""Two exact spatial average-pooling specializations with paired output rows.

Full-plane DMA and FP32 widening are shared across fourteen output-row pairs.
The public call gate includes N/C. No registration or dispatch mutation here.
"""

import functools

CONTRACTS = {
    "fp16-5x5": dict(
        n=2,
        c=128,
        h=56,
        w=56,
        kh=5,
        ph=2,
        ow=28,
        dtype="float16",
        divisor=25,
        ceil=False,
        include=True,
        override=None,
    ),
    "bf16-ceil": dict(
        n=3,
        c=96,
        h=55,
        w=57,
        kh=3,
        ph=1,
        ow=29,
        dtype="bfloat16",
        divisor=7,
        ceil=True,
        include=False,
        override=7,
    ),
}


def contract(window, dtype):
    for name, c in CONTRACTS.items():
        if (
            dtype == c["dtype"]
            and window.rows == c["n"] * c["c"]
            and window.size == (c["h"], c["w"])
            and window.out == (28, c["ow"])
            and window.kernel == (c["kh"], 5)
            and window.stride == (2, 2)
            and window.pad == (c["ph"], 2)
            and window.ceil_mode == c["ceil"]
            and window.count_include_pad == c["include"]
            and window.divisor_override == c["override"]
        ):
            return name
    return None


def call_contract(call):
    """Additionally gate actual N,C at the constructor boundary, not just N*C."""
    for name, c in CONTRACTS.items():
        if (
            call.n == c["n"]
            and call.c_in == c["c"]
            and tuple(call.size) == (c["h"], c["w"])
            and tuple(call.window) == (c["kh"], 5)
            and tuple(call.stride) == (2, 2)
            and tuple(call.pad) == (c["ph"], 2)
            and call.ceil_mode == c["ceil"]
            and call.count_include_pad == c["include"]
            and call.divisor_override == c["override"]
            and str(call.dtype).split(".")[-1] == c["dtype"]
        ):
            return name
    return None


def program(name="fp16-5x5"):
    c = CONTRACTS[name]
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    h, w, kh, ph, ow = c["h"], c["w"], c["kh"], c["ph"], c["ow"]
    dtype, divisor = c["dtype"], c["divisor"]
    rows = c["n"] * c["c"]
    plane = h * w
    padded_plane = (plane + 63) // 64 * 64
    paired_width = 2 * ow
    aligned_rows = w % 8 == 0

    @T.prim_func
    def main(X: T.Tensor((rows * plane,), dtype), Y: T.Tensor((rows, 14, paired_width), dtype)):
        with T.Kernel(64) as core:
            raw = T.alloc_shared((padded_plane,), dtype)
            data = T.alloc_shared((padded_plane,), "float32")
            output = T.alloc_shared((14, 64), dtype)
            # Only raw and output cross pipeline iterations. FP32 scratch is
            # produced and consumed wholly within one SIMD stage.
            T.annotate_buffer_versions({raw: 2, data: 1, output: 2})
            for step in T.Pipelined((rows + 63) // 64, num_stages=2):
                row = step * 64 + core
                if row < rows:
                    T.copy(X[row * plane : (row + 1) * plane], raw[:plane])
                    with T.SimdVF():
                        full = S.pset(32)
                        zero = S.vdup(T.float32(0), "float32", full)
                        izero = S.vdup(T.int32(0), "int32", full)
                        for part in T.serial(padded_plane // 64):
                            converted = S.vcvt(
                                S.vld(raw[part * 64], "UNPK_B16"), "float32", full, part=0
                            )
                            S.vsts(data[part * 64], converted, full)
                        S.mem_bar("VST_VLD")
                        lane = S.vci(0, "int32")
                        high = S.vcmps(lane, T.int32(ow), full, "ge")
                        high_columns = S.vsel(S.vdup(T.int32(ow), "int32", full), izero, high)
                        high_rows = S.vsel(S.vdup(T.int32(2), "int32", full), izero, high)
                        xstart = S.vadds(
                            S.vmuls(S.vsub(lane, high_columns, full), T.int32(2), full),
                            T.int32(-2),
                            full,
                        )
                        active = S.vcmps(lane, T.int32(paired_width), full, "lt")
                        xpos = S.alloc_local((5,), "int32")
                        masks = S.alloc_local((5,), "bool")
                        offsets = S.alloc_local((5,), "uint32")
                        for kx in T.Unroll(5, explicit=True):
                            xpos[kx] = S.vadds(xstart, T.int32(kx), full)
                            masks[kx] = S.pand(
                                active,
                                S.pand(
                                    S.vcmps(xpos[kx], T.int32(0), full, "ge"),
                                    S.vcmps(xpos[kx], T.int32(w), full, "lt"),
                                    full,
                                ),
                                full,
                            )
                            relative = S.vadd(S.vmuls(high_rows, T.int32(w), full), xpos[kx], full)
                            offsets[kx] = T.reinterpret(
                                S.vsel(relative, izero, masks[kx]), "uint32x64"
                            )
                        acc = S.alloc_local((1,), "float32")
                        for pair in T.serial(14):
                            acc[0] = zero
                            if pair > 0 and pair < 13:
                                for ky in T.serial(kh):
                                    base_y = pair * 4 - ph + ky
                                    for kx in T.Unroll(5, explicit=True):
                                        if aligned_rows:
                                            incoming = S.vgather2(
                                                data[base_y * w], offsets[kx], full
                                            )
                                        else:
                                            whole_offset = S.vadds(
                                                offsets[kx], T.cast(base_y * w, "uint32"), full
                                            )
                                            incoming = S.vgather2(data[0], whole_offset, full)
                                        acc[0] = S.vadd(
                                            acc[0], S.vsel(incoming, zero, masks[kx]), full
                                        )
                            else:
                                for ky in T.serial(kh):
                                    y = S.vadds(
                                        high_rows, T.cast(pair * 4 - ph + ky, "int32"), full
                                    )
                                    y_valid = S.pand(
                                        S.vcmps(y, T.int32(0), full, "ge"),
                                        S.vcmps(y, T.int32(h), full, "lt"),
                                        full,
                                    )
                                    for kx in T.Unroll(5, explicit=True):
                                        valid = S.pand(masks[kx], y_valid, full)
                                        address = S.vadd(
                                            S.vmuls(y, T.int32(w), full), xpos[kx], full
                                        )
                                        safe = T.reinterpret(
                                            S.vsel(address, izero, valid), "uint32x64"
                                        )
                                        incoming = S.vgather2(data[0], safe, full)
                                        acc[0] = S.vadd(acc[0], S.vsel(incoming, zero, valid), full)
                            average = S.vmuls(acc[0], T.float32(1.0 / divisor), full)
                            narrowed = S.vcvt(average, dtype, full, sat=False)
                            S.vsts(output[pair, 0], narrowed, active, "PK_B32")
                    T.copy(output[:, :paired_width], Y[row, :, :])

    return main.with_attr("tilelang_callee_allocated_outputs", 0)


@functools.lru_cache(maxsize=2)
def build(name="fp16-5x5"):
    import tilelang

    return tilelang.compile(program(name), target="ascend", out_idx=-1)
