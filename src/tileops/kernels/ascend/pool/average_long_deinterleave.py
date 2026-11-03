"""Exact long-temporal FP16 pooling with contiguous deinterleave and FP32 sums.

The validated program retains two-stage raw/output buffering and the original
five-tap accumulation order. The public constructor gate fixes N=2,C=256.
"""

import functools


def supports(window, dtype):
    return (
        dtype == "float16"
        and window.rows == 512
        and window.size == (32000,)
        and window.out == (8000,)
        and window.kernel == (5,)
        and window.stride == (4,)
        and window.pad == (2,)
        and not window.ceil_mode
        and window.count_include_pad
        and window.divisor_override is None
    )


def program():
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    @T.prim_func
    def main(X: T.Tensor((512, 32000), "float16"), Y: T.Tensor((512, 8000), "float16")):
        with T.Kernel(64) as core:
            raw = T.alloc_shared((32000,), "float16")
            output = T.alloc_shared((8000,), "float16")
            T.annotate_buffer_versions({raw: 2, output: 2})
            for step in T.Pipelined(8, num_stages=2):
                row = step * 64 + core
                T.copy(X[row, :], raw)
                with T.SimdVF():
                    full16 = S.pset(16)
                    full32 = S.pset(32)
                    zero = S.vdup(T.float32(0), "float32", full32)
                    lane = S.vci(0, "int32")
                    first = S.vcmps(lane, T.int32(0), full32, "eq")
                    predecessor = T.reinterpret(
                        S.vmaxs(S.vadds(lane, T.int32(-1), full32), T.int32(0), full32), "uint32x64"
                    )
                    # VF-local mutable carry: previous block's x[4*j+2/3].
                    # Reset for every input row, so the two left padding taps
                    # of the first output are explicit +0 (never GM/UB OOB).
                    carry = S.alloc_local((2,), "float32")
                    carry[0] = zero
                    carry[1] = zero
                    for v in T.serial(125):
                        # Exactly 256 half elements per load; all 125 loads fit
                        # the 32000-element row without a guard allocation.
                        evens, odds = S.vld2(raw[v * 256], dist="DINTLV_B16")
                        center = S.vcvt(evens, "float32", full16, part=0)  # x[4*j]
                        right1 = S.vcvt(odds, "float32", full16, part=0)  # x[4*j+1]
                        right2 = S.vcvt(evens, "float32", full16, part=1)  # x[4*j+2]
                        next_left1 = S.vcvt(odds, "float32", full16, part=1)  # x[4*j+3]
                        left2 = S.vsel(carry[0], S.vselr(right2, predecessor), first)
                        left1 = S.vsel(carry[1], S.vselr(next_left1, predecessor), first)
                        # Sequential +0,left2,left1,center,right1,right2, exactly
                        # as the original tap loop; no subtraction/window sums.
                        sum0 = S.vadd(zero, left2, full32)
                        sum1 = S.vadd(sum0, left1, full32)
                        sum2 = S.vadd(sum1, center, full32)
                        sum3 = S.vadd(sum2, right1, full32)
                        total = S.vadd(sum3, right2, full32)
                        average = S.vmuls(total, T.float32(1.0 / 5.0), full32)
                        narrowed = S.vcvt(average, "float16", sat=False)
                        S.vsts(output[v * 64], narrowed, full32, "PK_B32")
                        carry[0] = S.vdupv(right2, full32, pos="POS_HIGHEST")
                        carry[1] = S.vdupv(next_left1, full32, pos="POS_HIGHEST")
                T.copy(output, Y[row, :])

    return main.with_attr("tilelang_callee_allocated_outputs", 0)


@functools.lru_cache(maxsize=1)
def build(window, dtype):
    import tilelang

    assert supports(window, dtype)
    return tilelang.compile(program(), target="ascend", out_idx=-1)


def supports_call(call):
    return (
        call.n == 2
        and call.c_in == 256
        and tuple(call.size) == (32000,)
        and tuple(call.window) == (5,)
        and tuple(call.stride) == (4,)
        and tuple(call.pad) == (2,)
        and not call.ceil_mode
        and call.count_include_pad
        and call.divisor_override is None
        and str(call.dtype).split(".")[-1] == "float16"
    )
