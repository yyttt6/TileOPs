"""SIMD row-tile reductions for radix counts and softmax mass."""

import tilelang.ascend.language as T


@T.macro
def _probabilities_tile(x, probabilities, boundary_out, count, maximum, mass, boundary, dtype: str):
    with T.SimdVF():
        mask = T.simd.pset(32)
        one = T.simd.pset(32, "PAT_VL1")
        largest = T.simd.vdup(maximum, "float32")
        denominator = T.simd.vdup(mass, "float32")
        threshold = T.simd.vexp(T.simd.vsub(T.simd.vdup(boundary, "float32"), largest, mask), mask)
        threshold = T.simd.vdiv(threshold, denominator, mask, precision="exact")
        T.simd.vsts(boundary_out[0], threshold, one, extent=1)
        for j in T.serial((count + 63) // 64):
            valid = T.simd.update_mask(T.min(64, count - j * 64), 32)
            value = (
                T.simd.vld(x[j * 64])
                if dtype == "float32"
                else T.simd.vcvt(T.simd.vld(x[j * 64], dist="UNPK_B16"), "float32", part=0)
            )
            weight = T.simd.vexp(T.simd.vsub(value, largest, mask), mask)
            probability = T.simd.vdiv(weight, denominator, mask, precision="exact")
            T.simd.vsts(probabilities[j * 64], probability, valid)


@T.macro
def _keys(value, key, mask):
    canonical = T.simd.vsel(
        T.simd.vdup(T.float32(0), "float32"),
        value,
        T.simd.vcmps(value, T.float32(0), mask, op="eq"),
    )
    raw = T.reinterpret(canonical, "uint32x64")
    signed = T.reinterpret(canonical, "int32x64")
    negative = T.simd.vcmps(signed, T.int32(0), mask, op="lt")
    flip = T.simd.vsel(
        T.simd.vdup(T.uint32(0xFFFFFFFF), "uint32"),
        T.simd.vdup(T.uint32(0x80000000), "uint32"),
        negative,
    )
    absolute = T.simd.vand(raw, T.simd.vdup(T.uint32(0x7FFFFFFF), "uint32"), mask)
    key[0] = T.simd.vsel(
        T.simd.vdup(T.uint32(0xFFFFFFFF), "uint32"),
        T.simd.vxor(raw, flip, mask),
        T.simd.vcmps(absolute, T.uint32(0x7F800000), mask, op="gt"),
    )


@T.macro
def _rank_tile(x, out, count, pivot, shift: int, dtype: str):
    with T.SimdVF():
        mask = T.simd.pset(32)
        one = T.simd.pset(32, "PAT_VL1")
        key = T.simd.alloc_local(1, "uint32")
        total = T.simd.alloc_local(1, "int32")
        total[0] = T.simd.vdup(T.int32(0), "int32")
        for j in T.serial((count + 63) // 64):
            valid = T.simd.update_mask(T.min(64, count - j * 64), 32)
            value = (
                T.simd.vld(x[j * 64])
                if dtype == "float32"
                else T.simd.vcvt(T.simd.vld(x[j * 64], dist="UNPK_B16"), "float32", part=0)
            )
            _keys(value, key, mask)
            above = T.simd.vcmps(T.simd.vshrs(key[0], shift, mask), pivot, mask, op="ge")
            selected = T.simd.pand(valid, above, mask)
            increment = T.simd.vsel(
                T.simd.vdup(T.int32(1), "int32"),
                T.simd.vdup(T.int32(0), "int32"),
                selected,
            )
            total[0] = T.simd.vadd(total[0], increment, mask)
        result = T.simd.vcadd(total[0], mask)
        T.simd.vsts(out[0], result, one, extent=1)


@T.macro
def _mass_tile(x, out, count, pivot, maximum, shift: int, dtype: str):
    with T.SimdVF():
        mask = T.simd.pset(32)
        one = T.simd.pset(32, "PAT_VL1")
        key = T.simd.alloc_local(1, "uint32")
        total = T.simd.alloc_local(1, "float32")
        total[0] = T.simd.vdup(T.float32(0), "float32")
        largest = T.simd.vdup(maximum, "float32")
        for j in T.serial((count + 63) // 64):
            valid = T.simd.update_mask(T.min(64, count - j * 64), 32)
            value = (
                T.simd.vld(x[j * 64])
                if dtype == "float32"
                else T.simd.vcvt(T.simd.vld(x[j * 64], dist="UNPK_B16"), "float32", part=0)
            )
            _keys(value, key, mask)
            above = T.simd.vcmps(T.simd.vshrs(key[0], shift, mask), pivot, mask, op="ge")
            selected = T.simd.pand(valid, above, mask)
            weight = T.simd.vexp(T.simd.vsub(value, largest, mask), mask)
            selected = T.simd.pand(selected, T.simd.vcmp(value, value, mask, op="eq"), mask)
            weight = T.simd.vsel(weight, T.simd.vdup(T.float32(0), "float32"), selected)
            total[0] = T.simd.vadd(total[0], weight, mask)
        result = T.simd.vcadd(total[0], mask)
        T.simd.vsts(out[0], result, one, extent=1)
