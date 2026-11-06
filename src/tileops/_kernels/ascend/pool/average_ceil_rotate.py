"""Avg ceil BF16 round06: inject source edge lanes before circular permutation.

Fixed frozen05 lookahead dataflow; keep every ordered FP32 arithmetic operation.
The only candidate removes broadcasts, not divisor correction or +0 additions.
"""

import functools

CASES = {
    "ceil1d": dict(
        n=2,
        c=128,
        size=(2048,),
        out=(1024,),
        kernel=(4,),
        stride=(2,),
        pad=(1,),
        ceil=True,
        include=False,
        override=None,
        dtype="bfloat16",
    )
}


def call_contract(call):
    for name, c in CASES.items():
        if (
            call.n == c["n"]
            and call.c_in == c["c"]
            and (tuple(call.size) == c["size"])
            and (tuple(call.window) == c["kernel"])
            and (tuple(call.stride) == c["stride"])
            and (tuple(call.pad) == c["pad"])
            and (call.ceil_mode == c["ceil"])
            and (call.count_include_pad == c["include"])
            and (call.divisor_override == c["override"])
            and (str(call.dtype).split(".")[-1] == c["dtype"])
        ):
            return name
    return None


def supports(window, dtype, name):
    c = CASES[name]
    return (
        window.rows == c["n"] * c["c"]
        and window.size == c["size"]
        and (window.out == c["out"])
        and (window.kernel == c["kernel"])
        and (window.stride == c["stride"])
        and (window.pad == c["pad"])
        and (window.ceil_mode == c["ceil"])
        and (window.count_include_pad == c["include"])
        and (window.divisor_override == c["override"])
        and (dtype == c["dtype"])
    )


VARIANTS = {"rotate_edges": dict(batch=4, stages=1)}


def temporal_program(variant):
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    assert variant in VARIANTS
    batch = VARIANTS[variant]["batch"]
    stages = VARIANTS[variant]["stages"]

    @T.prim_func
    def main(X: T.Tensor((256, 2048), "bfloat16"), Y: T.Tensor((256, 1024), "bfloat16")):
        with T.Kernel(64) as core:
            raw = T.alloc_shared((batch, 2048), "bfloat16")
            output = T.alloc_shared((batch, 1024), "bfloat16")
            T.annotate_buffer_versions({raw: stages, output: stages})
            for step in T.Pipelined(4 // batch, num_stages=stages):
                first_row = (step * 64 + core) * batch
                T.copy(X[first_row : first_row + batch, :], raw)
                with T.SimdVF():
                    full16 = S.pset(16)
                    full = S.pset(32)
                    zero = S.vdup(T.float32(0), "float32", full)
                    lane = S.vci(0, "int32")
                    first = S.vcmps(lane, T.int32(0), full, "eq")
                    last = S.vcmps(lane, T.int32(63), full, "eq")
                    index_mask = S.vdup(T.int32(63), "int32", full)
                    prev = S.vand(S.vadds(lane, T.int32(-1), full), index_mask, full)
                    nxt = S.vand(S.vadds(lane, T.int32(1), full), index_mask, full)
                    carry = S.alloc_local((1,), "float32")
                    following = S.alloc_local((1,), "float32")
                    ae = S.alloc_local((1,), "float32")
                    ao = S.alloc_local((1,), "float32")
                    a = S.alloc_local((1,), "float32")
                    b = S.alloc_local((1,), "float32")
                    c = S.alloc_local((1,), "float32")
                    d = S.alloc_local((1,), "float32")
                    na = S.alloc_local((1,), "float32")
                    nb = S.alloc_local((1,), "float32")
                    nc = S.alloc_local((1,), "float32")
                    nd = S.alloc_local((1,), "float32")
                    for r in T.Unroll(batch, explicit=True):
                        carry[0] = zero
                        (e0, o0) = S.vld2(raw[r, 0], "DINTLV_B16")
                        a[0] = S.vcvt(e0, "float32", full16, part=0)
                        b[0] = S.vcvt(o0, "float32", full16, part=0)
                        c[0] = S.vcvt(e0, "float32", full16, part=1)
                        d[0] = S.vcvt(o0, "float32", full16, part=1)
                        for v in T.Unroll(8, explicit=True):
                            # Source lane63 receives previous d[63], then rotate right.
                            left = S.vselr(S.vsel(carry[0], d[0], last), prev)
                            following[0] = zero
                            if v < 7:
                                (en, on) = S.vld2(raw[r, (v + 1) * 256], "DINTLV_B16")
                                na[0] = S.vcvt(en, "float32", full16, part=0)
                                nb[0] = S.vcvt(on, "float32", full16, part=0)
                                nc[0] = S.vcvt(en, "float32", full16, part=1)
                                nd[0] = S.vcvt(on, "float32", full16, part=1)
                                following[0] = na[0]
                            # Source lane0 receives next a[0], then rotate left.
                            right = S.vselr(S.vsel(following[0], a[0], first), nxt)
                            se0 = S.vadd(zero, left, full)
                            se1 = S.vadd(se0, a[0], full)
                            se2 = S.vadd(se1, b[0], full)
                            se3 = S.vadd(se2, c[0], full)
                            so0 = S.vadd(zero, b[0], full)
                            so1 = S.vadd(so0, c[0], full)
                            so2 = S.vadd(so1, d[0], full)
                            so3 = S.vadd(so2, right, full)
                            (lo_sum, hi_sum) = S.vintlv(se3, so3)
                            if v == 0:
                                denom_lo = S.vsel(
                                    S.vdup(T.float32(3), "float32", full),
                                    S.vdup(T.float32(4), "float32", full),
                                    first,
                                )
                                ae[0] = S.vdiv(lo_sum, denom_lo, full)
                            else:
                                ae[0] = S.vmuls(lo_sum, T.float32(0.25), full)
                            if v == 7:
                                denom_hi = S.vsel(
                                    S.vdup(T.float32(3), "float32", full),
                                    S.vdup(T.float32(4), "float32", full),
                                    last,
                                )
                                ao[0] = S.vdiv(hi_sum, denom_hi, full)
                            else:
                                ao[0] = S.vmuls(hi_sum, T.float32(0.25), full)
                            S.vsts(
                                output[r, v * 128],
                                S.vcvt(ae[0], "bfloat16", sat=False),
                                full,
                                "PK_B32",
                            )
                            S.vsts(
                                output[r, v * 128 + 64],
                                S.vcvt(ao[0], "bfloat16", sat=False),
                                full,
                                "PK_B32",
                            )
                            carry[0] = d[0]
                            if v < 7:
                                a[0] = na[0]
                                b[0] = nb[0]
                                c[0] = nc[0]
                                d[0] = nd[0]
                T.copy(output, Y[first_row : first_row + batch, :])

    return main.with_attr("tilelang_callee_allocated_outputs", 0)


def program(variant="rotate_edges"):
    return temporal_program(variant)


@functools.lru_cache(maxsize=2)
def build(variant="rotate_edges"):
    import tilelang

    return tilelang.compile(program(variant), target="ascend", out_idx=-1)
