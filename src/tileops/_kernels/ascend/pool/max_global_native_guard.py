"""Native f16 values + INPUT bitwise NaN guard; whole-tile G07 strict fallback.
No reliance on native NaN propagation. Fixed row256/core64/stages2, one iteration; values only.
Fast admission: input magnitude max<=0x7c00 AND every output absbits>=0x0400.
"""

import functools

CONFIGS = {"native_guard": dict(row_tile=256, ilp=1, stages=2)}


def program(variant="native_guard", indices=False):
    if variant != "native_guard" or indices:
        raise ValueError("native_guard is exact global128 values only")
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    @T.prim_func
    def main(
        X: T.Tensor((16384 * 128,), "float16"),
        Y: T.Tensor((16384,), "float16"),
        I: T.Tensor((1,), "int64"),
    ):
        with T.Kernel(64) as core:
            raw = T.alloc_shared((32768,), "float16")
            values = T.alloc_shared((256,), "float16")
            flag = T.alloc_shared((16,), "uint16")
            repair = T.alloc_var("uint16", init=0)
            T.annotate_buffer_versions({raw: 2, values: 2, flag: 2})
            for step in T.Pipelined(1, num_stages=2):
                job = step * 64 + core
                T.copy(X[job * 32768 : (job + 1) * 32768], raw, l2_cache_ctrl="NOTALLOC_KEEP")
                with T.SimdVF():
                    full = S.pset(16)
                    one = S.pset(16, "PAT_VL1")
                    abs_mask = S.vdup(T.uint16(32767), "uint16", full)
                    zeros = S.vdup(T.uint16(0), "uint16", full)
                    ones = S.vdup(T.uint16(1), "uint16", full)
                    magnitude = S.alloc_local((1,), "uint16")
                    magnitude[0] = zeros
                    for r in T.serial(256):
                        original = S.vld(raw[r * 128])
                        abs_bits = S.vand(T.reinterpret(original, "uint16x128"), abs_mask, full)
                        magnitude[0] = S.vmax(magnitude[0], abs_bits, full)
                        native = S.vcmax(original, full)
                        S.vsts(values[r], native, one, "ONEPT_B16")
                    maximum_bits = S.vcmax(magnitude[0], full)
                    has_nan = S.vcmps(maximum_bits, T.uint16(31744), one, "gt")
                    # Same-VF output store/read needs an explicit memory ordering point.
                    S.mem_bar("VST_VLD")
                    native_values = S.vld(values[0])
                    native_abs = S.vand(T.reinterpret(native_values, "uint16x128"), abs_mask, full)
                    native_values_hi = S.vld(values[128])
                    native_abs_hi = S.vand(
                        T.reinterpret(native_values_hi, "uint16x128"), abs_mask, full
                    )
                    # Fold both128-lane output vectors: any tiny output in all256
                    # rows must trigger the SAME whole-core strict fallback.
                    output_min_abs = S.vmin(native_abs, native_abs_hi, full)
                    small = S.vcmps(output_min_abs, T.uint16(1024), full, "lt")
                    unsafe = S.por(has_nan, small, full)
                    flags = S.vsel(ones, zeros, unsafe)
                    any_unsafe = S.vcmax(flags, full)
                    S.vsts(flag[0], any_unsafe, one, "ONEPT_B16")
                # A visible scalar assignment must lower to V->S completion before
                # the UB load/branch. Audit generated CPP and ELF, not just source.
                repair = flag[0]
                if repair != 0:
                    with T.SimdVF():
                        full = S.pset(16)
                        one = S.pset(16, "PAT_VL1")
                        sign_bit = S.vdup(T.uint16(32768), "uint16", full)
                        slot_one = S.vdup(T.uint16(1), "uint16", full)
                        # Signed ramp -128..-1 has uint16 bits ff80..ffff.
                        nan_keys = T.reinterpret(S.vci(-128, "int16"), "uint16x128")
                        for r in T.serial(256):
                            original = S.vld(raw[r * 128])
                            bits = T.reinterpret(original, "uint16x128")
                            sign = S.vshrs(T.reinterpret(bits, "int16x128"), T.int16(15), full)
                            # lint: codespell allow: Ascend vector bitwise-OR intrinsic.
                            flip = S.vor(  # codespell:ignore vor
                                T.reinterpret(sign, "uint16x128"), sign_bit, full
                            )
                            key = S.vxor(bits, flip, full)
                            # Subtracting sign=-1 adds one to negative keys: both zeros
                            # naturally become 0x8000. All integer arithmetic is u16.
                            zero_folded = S.vsub(key, T.reinterpret(sign, "uint16x128"), full)
                            nans = S.vcmp(original, original, full, "ne")
                            ordered = S.vsel(nan_keys, zero_folded, nans)
                            reduced = S.vcmax(ordered, full)
                            # uint16 max emits value in lane0 and raw integer index in lane1.
                            winner = S.vselr(reduced, slot_one)
                            chosen = S.vselr(original, winner)
                            S.vsts(values[r], chosen, one, "ONEPT_B16")
                T.copy(values[0:256], Y[job * 256 : (job + 1) * 256])

    return main.with_attr("tilelang_callee_allocated_outputs", 0)


@functools.lru_cache(None)
def build(variant="native_guard", indices=False):
    import tilelang

    return tilelang.compile(program(variant, indices), target="ascend", out_idx=[1, 2])


def supports_call(call, indices=False):
    return (
        not indices
        and call.n == 64
        and call.c_in == 256
        and tuple(call.size) == (128,)
        and tuple(call.window) == (128,)
        and tuple(call.stride) == (1,)
        and tuple(call.pad) == (0,)
        and tuple(call.dilation) == (1,)
        and call.ceil_mode is False
        and str(call.dtype).split(".")[-1] == "float16"
    )
