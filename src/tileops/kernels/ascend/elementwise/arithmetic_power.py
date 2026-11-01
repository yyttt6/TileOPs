"""Ascend power with negative-base, IEEE special-value and broadcast support."""
import functools

import torch

from .arithmetic import ArithmeticKernelBase
from tileops.kernels.elementwise.call_spec import BinaryElementwiseFwdInterface

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None


if T is not None:
    @T.macro
    def _power(av, bv, mask, algorithm: int):
        # All inputs are widened exactly to FP32 before classification and SFU math.
        zero = T.simd.vdup(T.float32(0), "float32", mask)
        one = T.simd.vdup(T.float32(1), "float32", mask)
        infinity = T.simd.vdup(T.float32(float("inf")), "float32", mask)
        nan = T.simd.vdup(T.float32(float("nan")), "float32", mask)
        aa = T.simd.vabs(av)
        ab = T.simd.vabs(bv)
        bi = T.simd.vcvt(bv, "int32", round="ROUND_Z")
        whole = T.simd.vcvt(bi, "float32")
        # Every representable FP32 number at/above 2**24 is an even integer.
        small = T.simd.vcmps(ab, T.float32(16777216), mask, op="lt")
        finite_b = T.simd.vcmp(ab, infinity, mask, op="lt")
        integer = T.simd.pand(T.simd.por(T.simd.vcmp(whole, bv, mask, op="eq"), T.simd.pnot(small, mask), mask), finite_b, mask)
        lowbit = T.simd.vand(bi, T.simd.vdup(T.int32(1), "int32", mask), mask)
        odd = T.simd.pand(T.simd.pand(integer, small, mask), T.simd.vcmps(lowbit, T.int32(1), mask, op="eq"), mask)
        if algorithm == 0 or algorithm == 2:
            # Retained paired control: the generic non-FTZ wrappers use two SFUs each.
            magnitude = T.simd.vexp(T.simd.vmul(T.simd.vln(aa, mask, precision="ftz_false"), bv, mask), mask, precision="ftz_false")
        else:
            # Select the scaled input before the SFU: one ln and one exp per lane.
            # Scaling is exact for positive subnormal inputs. The exponential is
            # split slightly above the normal/subnormal boundary to cover rounding.
            subnormal = T.simd.pand(T.simd.vcmps(aa, T.float32(1.1754943508222875e-38), mask, op="lt"), T.simd.vcmp(aa, zero, mask, op="gt"), mask)
            log_input = T.simd.vsel(T.simd.vmuls(aa, T.float32(8388608), mask), aa, subnormal)
            log_value = T.simd.vln(log_input, mask, precision="ftz_true")
            correction = T.simd.vsel(T.simd.vdup(T.float32(15.942385152878742), "float32", mask), zero, subnormal)
            exponent = T.simd.vmul(T.simd.vsub(log_value, correction, mask), bv, mask)
            split = T.simd.vcmps(exponent, T.float32(-87), mask, op="lt")
            exp_input = T.simd.vsel(T.simd.vmuls(exponent, T.float32(0.5), mask), exponent, split)
            exp_value = T.simd.vexp(exp_input, mask, precision="ftz_true")
            magnitude = T.simd.vsel(T.simd.vmul(exp_value, exp_value, mask), exp_value, split)
        # SFU inf*0 and ln(0) corner cases are resolved explicitly.
        is_zero = T.simd.vcmp(aa, zero, mask, op="eq")
        is_inf = T.simd.vcmp(aa, infinity, mask, op="eq")
        positive_exp = T.simd.vcmp(bv, zero, mask, op="gt")
        magnitude = T.simd.vsel(T.simd.vsel(zero, infinity, positive_exp), magnitude, is_zero)
        magnitude = T.simd.vsel(T.simd.vsel(infinity, zero, positive_exp), magnitude, is_inf)
        # Reciprocal distinguishes -0; direct comparison covers -infinity.
        if algorithm >= 2:
            negative = T.simd.vcmps(T.reinterpret(av, "int32x64"), T.int32(0), mask, op="lt")
        else:
            negative = T.simd.por(T.simd.vcmp(av, zero, mask, op="lt"), T.simd.vcmp(T.simd.vdiv(one, av, mask, precision="ftz_true"), zero, mask, op="lt"), mask)
        result = T.simd.vsel(T.simd.vneg(magnitude), magnitude, T.simd.pand(negative, odd, mask))
        negative_finite_nonzero = T.simd.pand(T.simd.vcmp(av, zero, mask, op="lt"), T.simd.vcmp(aa, infinity, mask, op="lt"), mask)
        invalid = T.simd.pand(negative_finite_nonzero, T.simd.pand(T.simd.pnot(integer, mask), finite_b, mask), mask)
        result = T.simd.vsel(nan, result, invalid)
        # |x| == 1 with infinite exponent is 1, including (-1)**(+/-inf).
        inf_exponent_one = T.simd.pand(T.simd.vcmp(ab, infinity, mask, op="eq"), T.simd.vcmp(aa, one, mask, op="eq"), mask)
        result = T.simd.vsel(one, result, inf_exponent_one)
        operand_nan = T.simd.por(T.simd.vcmp(av, av, mask, op="ne"), T.simd.vcmp(bv, bv, mask, op="ne"), mask)
        result = T.simd.vsel(nan, result, operand_nan)
        result = T.simd.vsel(av, result, T.simd.vcmp(bv, one, mask, op="eq"))
        # These identities override NaN according to torch.pow/C pow semantics.
        identity = T.simd.por(T.simd.vcmp(av, one, mask, op="eq"), T.simd.vcmp(bv, zero, mask, op="eq"), mask)
        return T.simd.vsel(one, result, identity)

    @T.macro
    def _body(a, b, c, width: int, begin, period: int, repeat: int,
              dtype: str, alpha: float, algorithm: int):
        with T.SimdVF():
            mask = T.simd.pset(32)
            for i in range(width // 64):
                av = T.simd.vld(a[i * 64]) if dtype == 'float32' else T.simd.vcvt(T.simd.vld(a[i * 64], dist='UNPK_B16'), 'float32', part=0)
                if period and repeat > 1:
                    index = ((begin + i * 64) // repeat) % period
                    bv = T.simd.vld(b[index], dist='BRC_B32') if dtype == 'float32' else T.simd.vcvt(T.simd.vld(b[index], dist='BRC_B16'), 'float32', part=0)
                elif period:
                    index = (begin + i * 64) % period
                    bv = T.simd.vld(b[index]) if dtype == 'float32' else T.simd.vcvt(T.simd.vld(b[index], dist='UNPK_B16'), 'float32', part=0)
                else:
                    bv = T.simd.vld(b[i * 64]) if dtype == 'float32' else T.simd.vcvt(T.simd.vld(b[i * 64], dist='UNPK_B16'), 'float32', part=0)
                cv = _power(av, bv, mask, algorithm)
                if dtype == 'float32':
                    T.simd.vsts(c[i * 64], cv, mask)
                else:
                    cv16 = T.simd.vcvt(cv, dtype)
                    T.simd.vsts(c[i * 64], cv16, mask, dist='PK_B32')

    @functools.lru_cache(maxsize=512)
    def build(n, dtype, period=0, repeat=1, alpha=1, tile=8192, blocks=72, stages=2, algorithm=2):
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1
        b_width = period or tile
        b_tail = period or tail

        @T.prim_func
        def main(A: T.Tensor((n,), dtype), B: T.Tensor((period or n,), dtype), C: T.Tensor((n,), dtype)):
            with T.Kernel(grid) as bx:
                if full:
                    a = T.alloc_shared((tile,), dtype)
                    b = T.alloc_shared((b_width,), dtype)
                    c = T.alloc_shared((tile,), dtype)
                    T.annotate_buffer_versions({a: stages, c: stages})
                    if period:
                        T.copy(B, b)
                    else:
                        T.annotate_buffer_versions({b: stages})
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(A[begin:begin + tile], a)
                            if not period:
                                T.copy(B[begin:begin + tile], b)
                            _body(a, b, c, tile, begin, period, repeat, dtype, alpha, algorithm)
                            T.copy(c, C[begin:begin + tile], l2_cache_ctrl=0)
                if tail:
                    if bx == 0:
                        at = T.alloc_shared((tail,), dtype)
                        bt = T.alloc_shared((b_tail,), dtype)
                        ct = T.alloc_shared((tail,), dtype)
                        T.copy(A[full * tile:n], at)
                        if period:
                            T.copy(B, bt)
                        else:
                            T.copy(B[full * tile:n], bt)
                        _body(at, bt, ct, tail, full * tile, period, repeat, dtype, alpha, algorithm)
                        T.copy(ct, C[full * tile:n], l2_cache_ctrl=0)
        return tilelang.compile(main, target='ascend', out_idx=-1)

    @T.macro
    def _channel_body(a, b, c, channels: int, repeat: int,
                      dtype: str, alpha: float, b_begin, algorithm: int):
        with T.SimdVF():
            mask = T.simd.pset(32)
            for ch in range(channels):
                bv = T.simd.vld(b[b_begin + ch], dist='BRC_B32') if dtype == 'float32' else T.simd.vcvt(T.simd.vld(b[b_begin + ch], dist='BRC_B16'), 'float32', part=0)
                for i in range(repeat // 64):
                    index = ch * repeat + i * 64
                    av = T.simd.vld(a[index]) if dtype == 'float32' else T.simd.vcvt(T.simd.vld(a[index], dist='UNPK_B16'), 'float32', part=0)
                    cv = _power(av, bv, mask, algorithm)
                    if dtype == 'float32':
                        T.simd.vsts(c[index], cv, mask)
                    else:
                        cv16 = T.simd.vcvt(cv, dtype)
                        T.simd.vsts(c[index], cv16, mask, dist='PK_B32')

    @functools.lru_cache(maxsize=512)
    def build_channel(n, dtype, period, repeat, alpha=1, tile=4, blocks=72, stages=2, algorithm=2):
        width = tile * repeat
        count = n // width
        grid = min(blocks, count)
        assert repeat % 64 == 0 and period % tile == 0 and n % width == 0

        @T.prim_func
        def main(A: T.Tensor((n,), dtype), B: T.Tensor((period,), dtype), C: T.Tensor((n,), dtype)):
            with T.Kernel(grid) as bx:
                a = T.alloc_shared((width,), dtype)
                b = T.alloc_shared((period,), dtype)
                c = T.alloc_shared((width,), dtype)
                T.annotate_buffer_versions({a: stages, c: stages})
                T.copy(B, b)
                for step in T.Pipelined((count + grid - 1) // grid, num_stages=stages):
                    block = step * grid + bx
                    if block < count:
                        begin = block * width
                        T.copy(A[begin:begin + width], a)
                        _channel_body(a, b, c, tile, repeat, dtype, alpha, (block * tile) % period, algorithm)
                        T.copy(c, C[begin:begin + width], l2_cache_ctrl=0)
        return tilelang.compile(main, target='ascend', out_idx=-1)

    @T.macro
    def _bias_body(a, b, c, rows: int, period: int, dtype: str, alpha: float, algorithm: int):
        with T.SimdVF():
            mask = T.simd.pset(32)
            for row in range(rows):
                for i in range(period // 64):
                    index = row * period + i * 64
                    av = T.simd.vld(a[index]) if dtype == "float32" else T.simd.vcvt(T.simd.vld(a[index], dist="UNPK_B16"), "float32", part=0)
                    bv = T.simd.vld(b[i * 64]) if dtype == "float32" else T.simd.vcvt(T.simd.vld(b[i * 64], dist="UNPK_B16"), "float32", part=0)
                    cv = _power(av, bv, mask, algorithm)
                    if dtype == "float32":
                        T.simd.vsts(c[index], cv, mask)
                    else:
                        cv16 = T.simd.vcvt(cv, dtype)
                        T.simd.vsts(c[index], cv16, mask, dist="PK_B32")

    @functools.lru_cache(maxsize=256)
    def build_bias(n, dtype, period, alpha=1, rows=2, blocks=64, stages=2, algorithm=2):
        width = rows * period
        count = n // width
        grid = min(blocks, count)
        assert n % width == 0 and period % 64 == 0

        @T.prim_func
        def main(A: T.Tensor((n,), dtype), B: T.Tensor((period,), dtype), C: T.Tensor((n,), dtype)):
            with T.Kernel(grid) as bx:
                a = T.alloc_shared((width,), dtype)
                b = T.alloc_shared((period,), dtype)
                c = T.alloc_shared((width,), dtype)
                T.annotate_buffer_versions({a: stages, c: stages})
                T.copy(B, b)
                for step in T.Pipelined((count + grid - 1) // grid, num_stages=stages):
                    block = step * grid + bx
                    if block < count:
                        begin = block * width
                        T.copy(A[begin:begin + width], a)
                        _bias_body(a, b, c, rows, period, dtype, alpha, algorithm)
                        T.copy(c, C[begin:begin + width], l2_cache_ctrl=0)
        return tilelang.compile(main, target="ascend", out_idx=-1)

    @T.macro
    def _widen_inputs(a, b, af, bf, width: int, dtype: str):
        with T.SimdVF():
            mask = T.simd.pset(32)
            for i in range(width // 64):
                av = T.simd.vcvt(T.simd.vld(a[i * 64], dist="UNPK_B16"), "float32", part=0)
                bv = T.simd.vcvt(T.simd.vld(b[i * 64], dist="UNPK_B16"), "float32", part=0)
                T.simd.vsts(af[i * 64], av, mask)
                T.simd.vsts(bf[i * 64], bv, mask)

    @T.macro
    def _narrow_output(cf, c, width: int, dtype: str):
        with T.SimdVF():
            mask = T.simd.pset(32)
            for i in range(width // 64):
                cv = T.simd.vcvt(T.simd.vld(cf[i * 64]), dtype)
                T.simd.vsts(c[i * 64], cv, mask, dist="PK_B32")

    @functools.lru_cache(maxsize=256)
    def build_widened(n, dtype, tile=4096, blocks=64, stages=2, algorithm=2):
        """Separate exact conversion VFs from the homogeneous FP32 arithmetic VF."""
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1
        assert dtype in ("float16", "bfloat16") and n % 64 == 0
        assert (full > 0) * (6 * tile * stages + 12 * tile) + 18 * tail <= 248 * 1024

        @T.prim_func
        def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
            with T.Kernel(grid) as bx:
                if full:
                    a = T.alloc_shared((tile,), dtype)
                    b = T.alloc_shared((tile,), dtype)
                    c = T.alloc_shared((tile,), dtype)
                    af = T.alloc_shared((tile,), "float32")
                    bf = T.alloc_shared((tile,), "float32")
                    cf = T.alloc_shared((tile,), "float32")
                    T.annotate_buffer_versions({a: stages, b: stages, c: stages, af: 1, bf: 1, cf: 1})
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(A[begin:begin + tile], a)
                            T.copy(B[begin:begin + tile], b)
                            _widen_inputs(a, b, af, bf, tile, dtype)
                            _body(af, bf, cf, tile, 0, 0, 1, "float32", 1, algorithm)
                            _narrow_output(cf, c, tile, dtype)
                            T.copy(c, C[begin:begin + tile], l2_cache_ctrl=0)
                if tail:
                    if bx == 0:
                        at = T.alloc_shared((tail,), dtype)
                        bt = T.alloc_shared((tail,), dtype)
                        ct = T.alloc_shared((tail,), dtype)
                        aft = T.alloc_shared((tail,), "float32")
                        bft = T.alloc_shared((tail,), "float32")
                        cft = T.alloc_shared((tail,), "float32")
                        T.copy(A[full * tile:n], at)
                        T.copy(B[full * tile:n], bt)
                        _widen_inputs(at, bt, aft, bft, tail, dtype)
                        _body(aft, bft, cft, tail, 0, 0, 1, "float32", 1, algorithm)
                        _narrow_output(cft, ct, tail, dtype)
                        T.copy(ct, C[full * tile:n], l2_cache_ctrl=0)
        return tilelang.compile(main, target="ascend", out_idx=-1)


    @T.macro
    def _widen_input(a, af, width: int):
        with T.SimdVF():
            mask = T.simd.pset(32)
            for i in range(width // 64):
                av = T.simd.vcvt(T.simd.vld(a[i * 64], dist="UNPK_B16"), "float32", part=0)
                T.simd.vsts(af[i * 64], av, mask)

    @functools.lru_cache(maxsize=256)
    def build_widened_bias(n, dtype, period, tile=4096, blocks=64, stages=2, algorithm=2):
        """Retain the widened bias on chip, and convert each input/output tile once."""
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1
        assert dtype in ("float16", "bfloat16") and n % 64 == 0 and period % 64 == 0
        assert (full > 0) * (4 * tile * stages + 8 * tile) + 6 * period + 12 * tail <= 248 * 1024

        @T.prim_func
        def main(A: T.Tensor((n,), dtype), B: T.Tensor((period,), dtype), C: T.Tensor((n,), dtype)):
            with T.Kernel(grid) as bx:
                b = T.alloc_shared((period,), dtype)
                bf = T.alloc_shared((period,), "float32")
                T.copy(B, b)
                _widen_input(b, bf, period)
                if full:
                    a = T.alloc_shared((tile,), dtype)
                    c = T.alloc_shared((tile,), dtype)
                    af = T.alloc_shared((tile,), "float32")
                    cf = T.alloc_shared((tile,), "float32")
                    T.annotate_buffer_versions({a: stages, c: stages, af: 1, cf: 1})
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(A[begin:begin + tile], a)
                            _widen_input(a, af, tile)
                            _body(af, bf, cf, tile, begin, period, 1, "float32", 1, algorithm)
                            _narrow_output(cf, c, tile, dtype)
                            T.copy(c, C[begin:begin + tile], l2_cache_ctrl=0)
                if tail:
                    if bx == 0:
                        at = T.alloc_shared((tail,), dtype)
                        ct = T.alloc_shared((tail,), dtype)
                        aft = T.alloc_shared((tail,), "float32")
                        cft = T.alloc_shared((tail,), "float32")
                        T.copy(A[full * tile:n], at)
                        _widen_input(at, aft, tail)
                        _body(aft, bf, cft, tail, full * tile, period, 1, "float32", 1, algorithm)
                        _narrow_output(cft, ct, tail, dtype)
                        T.copy(ct, C[full * tile:n], l2_cache_ctrl=0)
        return tilelang.compile(main, target="ascend", out_idx=-1)



class AscendPowKernel(ArithmeticKernelBase, BinaryElementwiseFwdInterface):
    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.a_shape, call.b_shape, call.dtype)

    def _compile(self, dtype_name, tile, stages):
        # Dedicated conversion VFs avoid the measured mixed-dtype arithmetic cost.
        # The 4K/two-stage layout won the controlled 16-bit search.
        if self.dtype.itemsize == 2:
            if not self.period:
                return build_widened(self.padded, dtype_name, 4096, 64, 2)
            if self.repeat == 1 and not (self.period & (self.period - 1)):
                return build_widened_bias(self.padded, dtype_name, self.period, 4096, 64, 2)
        if dtype_name == "float32" and not self.period:
            tile, stages = 4096, 3
        if self.period and self.repeat == 1 and self.period & (self.period - 1):
            rows = 2 if self.dtype.itemsize == 2 and self.n % (2 * self.period) == 0 else 1
            pipeline = 2
            while (rows * self.period * 2 * pipeline + self.period) * self.dtype.itemsize > 248 * 1024:
                if rows > 1:
                    rows //= 2
                else:
                    pipeline = 1
            return build_bias(self.n, dtype_name, self.period, 1, rows, 64, pipeline)
        channels = 4
        # Keep the channel index outside the vector loop, including odd channel
        # counts. Smaller groups also fit wide broadcasts within the UB budget.
        if self.repeat > 1:
            while channels > 1 and (self.period % channels != 0 or
                    (channels * self.repeat * 4 + self.period) * self.dtype.itemsize > 248 * 1024):
                channels //= 2
            if (channels * self.repeat * 4 + self.period) * self.dtype.itemsize <= 248 * 1024:
                return build_channel(self.padded, dtype_name, self.period, self.repeat, 1, channels, 64, 2)
        return build(self.padded, dtype_name, self.period, self.repeat, 1, tile, 64, stages)



if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("PowFwdOp", "ascend", AscendPowKernel)
