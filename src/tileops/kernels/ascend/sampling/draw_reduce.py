"""SIMD residual weights and chunk totals for categorical draws."""

import tilelang.ascend.language as T


@T.macro
def _weights_tile(x, d, out, count, residual: bool):
    with T.SimdVF():
        mask = T.simd.pset(32)
        one = T.simd.pset(32, "PAT_VL1")
        total = T.simd.alloc_local(1, "float32")
        total[0] = T.simd.vdup(T.float32(0), "float32")
        zero = T.simd.vdup(T.float32(0), "float32")
        for j in T.serial((count + 63) // 64):
            valid = T.simd.update_mask(T.min(64, count - j * 64), 32)
            weight = T.simd.vld(x[j * 64])
            if residual:
                weight = T.simd.vmax(T.simd.vsub(weight, T.simd.vld(d[j * 64]), mask), zero, mask)
            weight = T.simd.vsel(weight, zero, valid)
            total[0] = T.simd.vadd(total[0], weight, mask)
        result = T.simd.vcadd(total[0], mask)
        T.simd.vsts(out[0], result, one, extent=1)


@T.macro
def _draw_tile(x, d, out, count, point, begin, residual: bool):
    with T.SimdVF():
        mask = T.simd.pset(32)
        one = T.simd.pset(32, "PAT_VL1")
        lanes = T.simd.vci(T.int32(0), "int32")
        end_lane = T.simd.vdup(T.uint32(63), "uint32")
        zero = T.simd.vdup(T.float32(0), "float32")
        missing = T.simd.vdup(T.int32(2147483647), "int32")
        negative = T.simd.vdup(T.int32(-1), "int32")
        carry = T.simd.alloc_local(1, "float32")
        first = T.simd.alloc_local(1, "int32")
        last = T.simd.alloc_local(1, "int32")
        carry[0] = zero
        first[0] = missing
        last[0] = negative
        indices = T.simd.alloc_local(6, "int32")
        for k in T.unroll(6, explicit=True):
            indices[k] = T.simd.vmaxs(
                T.simd.vadds(lanes, T.int32(-(1 << k)), mask), T.int32(0), mask
            )
        for j in T.serial((count + 63) // 64):
            valid = T.simd.update_mask(T.min(64, count - j * 64), 32)
            value = T.simd.vld(x[0, j * 64])
            if residual:
                value = T.simd.vmax(T.simd.vsub(value, T.simd.vld(d[j * 64]), mask), zero, mask)
            value = T.simd.vsel(value, zero, valid)
            prefix = T.simd.alloc_local(1, "float32")
            prefix[0] = value
            for k in T.unroll(6, explicit=True):
                shifted = T.simd.vselr(prefix[0], indices[k])
                active = T.simd.vcmps(lanes, T.int32(1 << k), mask, op="ge")
                prefix[0] = T.simd.vadd(prefix[0], T.simd.vsel(shifted, zero, active), mask)
            cdf = T.simd.vadd(prefix[0], carry[0], mask)
            carry[0] = T.simd.vselr(cdf, end_lane)
            positive = T.simd.pand(T.simd.vcmps(value, T.float32(0), mask, op="gt"), valid, mask)
            above = T.simd.pand(T.simd.vcmps(cdf, point, mask, op="gt"), positive, mask)
            index = T.simd.vadds(lanes, j * 64, mask)
            first[0] = T.simd.vmin(first[0], T.simd.vsel(index, missing, above), mask)
            last[0] = T.simd.vmax(last[0], T.simd.vsel(index, negative, positive), mask)
        f = T.simd.vcmin(first[0], mask)
        l = T.simd.vcmax(last[0], mask)
        selected = T.simd.vsel(f, l, T.simd.vcmps(f, count, mask, op="lt"))
        selected = T.simd.vadds(selected, begin, mask)
        T.simd.vsts(out[0], selected, one, extent=1)
