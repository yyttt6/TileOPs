"""Ascend true division with FP32 precise arithmetic and on-chip broadcasts."""

import functools

import torch

from tileops.kernels.ascend.elementwise.arithmetic import ArithmeticKernelBase
from tileops.kernels.elementwise.call_spec import BinaryElementwiseFwdInterface

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None


if T is not None:

    @T.macro
    def _repair_floor(
        a, b, c, width: int, begin, period: int, repeat: int, dtype: str, channel: bool = False
    ):
        # Scalar libm supplies the remainder; staged vector division keeps
        # the FP32 quotient exact, including tiny divisors.
        limit = 8192 if dtype == "float16" else 65536 if dtype == "bfloat16" else 4194304
        numerator = T.alloc_shared((width,), "float32")
        adjustment = T.alloc_shared((width,), "float32")
        T.annotate_buffer_versions({numerator: 1, adjustment: 1})
        repair_threads = min(width, 256)
        with T.SimtVF(threads=repair_threads):
            repair_thread = T.get_thread_binding()
            for repair_step in T.serial((width + repair_threads - 1) // repair_threads):
                j = repair_step * repair_threads + repair_thread
                if j < width:
                    scalar_a = T.cast(a[j], "float32")
                    if channel:
                        scalar_b = T.cast(b[begin + j // repeat], "float32")
                    elif period:
                        scalar_b = T.cast(b[((begin + j) // repeat) % period], "float32")
                    else:
                        scalar_b = T.cast(b[j], "float32")
                    numerator[j] = scalar_a
                    adjustment[j] = T.float32(0)
                    if T.abs(scalar_a) >= T.abs(scalar_b) * T.float32(limit):
                        rem = T.call_extern("float32", "fmodf", scalar_a, scalar_b)
                        numerator[j] = scalar_a - rem
                        adjustment[j] = T.if_then_else(
                            T.And(rem != 0, (scalar_b < 0) != (rem < 0)), T.float32(1), T.float32(0)
                        )
        with T.SimdVF():
            mask = T.simd.pset(32)
            zero = T.simd.vdup(T.float32(0), "float32", mask)
            for i in range(width // 64):
                repair_a = (
                    T.simd.vld(a[i * 64])
                    if dtype == "float32"
                    else T.simd.vcvt(T.simd.vld(a[i * 64], dist="UNPK_B16"), "float32", part=0)
                )
                if channel:
                    index = begin + i * 64 // repeat
                    repair_b = (
                        T.simd.vld(b[index], dist="BRC_B32")
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(b[index], dist="BRC_B16"), "float32", part=0)
                    )
                elif period and repeat > 1:
                    index = ((begin + i * 64) // repeat) % period
                    repair_b = (
                        T.simd.vld(b[index], dist="BRC_B32")
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(b[index], dist="BRC_B16"), "float32", part=0)
                    )
                elif period:
                    index = ((begin // 64 + i) % (period // 64)) * 64
                    repair_b = (
                        T.simd.vld(b[index])
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(b[index], dist="UNPK_B16"), "float32", part=0)
                    )
                else:
                    repair_b = (
                        T.simd.vld(b[i * 64])
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(b[i * 64], dist="UNPK_B16"), "float32", part=0)
                    )
                q = _divide(T.simd.vld(numerator[i * 64]), repair_b, "float32", "true", mask)
                div = T.simd.vsub(q, T.simd.vld(adjustment[i * 64]), mask)
                whole = T.simd.vcvt(T.simd.vcvt(div, "int32", round="ROUND_F"), "float32")
                small = T.simd.vcmps(T.simd.vabs(div), T.float32(8388608), mask, op="lt")
                whole = T.simd.vsel(whole, div, small)
                if dtype != "float32":
                    whole = T.simd.vcvt(T.simd.vcvt(whole, dtype), "float32", part=0)
                bumped = T.simd.vadds(whole, T.float32(1), mask)
                if dtype != "float32":
                    bumped = T.simd.vcvt(T.simd.vcvt(bumped, dtype), "float32", part=0)
                up = T.simd.vcmps(T.simd.vsub(div, whole, mask), T.float32(0.5), mask, op="gt")
                result = T.simd.vsel(bumped, whole, up)
                eligible = T.simd.vcmp(
                    T.simd.vabs(repair_a),
                    T.simd.vmuls(T.simd.vabs(repair_b), T.float32(limit), mask),
                    mask,
                    op="ge",
                )
                eligible = T.simd.pand(eligible, T.simd.vcmp(repair_b, zero, mask, op="ne"), mask)
                eligible = T.simd.pand(eligible, T.simd.vcmp(div, zero, mask, op="ne"), mask)
                old = (
                    T.simd.vld(c[i * 64])
                    if dtype == "float32"
                    else T.simd.vcvt(T.simd.vld(c[i * 64], dist="UNPK_B16"), "float32", part=0)
                )
                result = T.simd.vsel(result, old, eligible)
                if dtype == "float32":
                    T.simd.vsts(c[i * 64], result, mask)
                else:
                    T.simd.vsts(c[i * 64], T.simd.vcvt(result, dtype), mask, dist="PK_B32")

    @T.macro
    def _divide(av, bv, dtype: str, rounding_mode: str, mask):
        # Normalize tiny operands before division: the vector divider flushes
        # subnormal inputs even in exact mode. Scaling both operands preserves
        # the quotient; overflow of the other operand implies a saturated result.
        if dtype != "float16":
            tiny_a = T.simd.vcmps(T.simd.vabs(av), T.float32(2.350988701644575e-38), mask, op="lt")
            tiny_b = T.simd.vcmps(T.simd.vabs(bv), T.float32(2.350988701644575e-38), mask, op="lt")
            tiny = T.simd.por(tiny_a, tiny_b, mask)
            scale = T.simd.vsel(
                T.simd.vdup(T.float32(16777216), "float32", mask),
                T.simd.vdup(T.float32(1), "float32", mask),
                tiny,
            )
            qa = T.simd.vmul(av, scale, mask)
            qb = T.simd.vmul(bv, scale, mask)
        else:
            qa, qb = av, bv
        if dtype == "float32" or rounding_mode != "true":
            q = T.simd.vdiv(qa, qb, mask, precision="exact")
        else:
            q = T.simd.vdiv(qa, qb, mask, precision="ftz_true")
        if dtype == "bfloat16" and rounding_mode == "true":
            # BF16 subnormals are integral multiples of 2**-133. Compute that
            # integer in the normal FP32 range, then construct its bits so the
            # divider's output FTZ cannot erase a representable BF16 value.
            sub_num = T.simd.vmuls(T.simd.vabs(av), T.float32(18446744073709551616.0), mask)
            sub_den = T.simd.vmuls(T.simd.vabs(bv), T.float32(1.6940658945086007e-21), mask)
            units = T.simd.vdiv(sub_num, sub_den, mask, precision="exact")
            mantissa = T.simd.vcvt(units, "int32", round="ROUND_R")
            bits = T.simd.vshls(mantissa, T.int32(16), mask)
            sign = T.simd.vand(
                T.reinterpret(q, "int32x64"), T.simd.vdup(T.int32(-2147483648), "int32", mask), mask
            )
            subnormal = T.reinterpret(T.simd.vor(bits, sign, mask), "float32x64")
            use_sub = T.simd.vcmps(T.simd.vabs(q), T.float32(1.1754943508222875e-38), mask, op="lt")
            use_sub = T.simd.pand(
                use_sub, T.simd.vcmps(T.simd.vabs(av), T.float32(0), mask, op="gt"), mask
            )
            use_sub = T.simd.pand(
                use_sub, T.simd.vcmps(T.simd.vabs(bv), T.float32(float("inf")), mask, op="lt"), mask
            )
            q = T.simd.vsel(subnormal, q, use_sub)
        if rounding_mode == "true":
            result = q
        else:
            zero = T.simd.vdup(T.float32(0), "float32", mask)
            one = T.simd.vdup(T.float32(1), "float32", mask)
            negative_one = T.simd.vdup(T.float32(-1), "float32", mask)
            nan = T.simd.vdup(T.float32(float("nan")), "float32", mask)
            if rounding_mode == "trunc":
                if dtype != "float32":
                    narrow = T.simd.vcvt(q, dtype)
                    rounded_q = T.simd.vcvt(narrow, "float32", part=0)
                else:
                    rounded_q = q
                whole = T.simd.vcvt(T.simd.vcvt(rounded_q, "int32", round="ROUND_Z"), "float32")
                small = T.simd.vcmps(T.simd.vabs(rounded_q), T.float32(8388608), mask, op="lt")
                result = T.simd.vsel(whole, rounded_q, small)
            else:
                whole = T.simd.vcvt(T.simd.vcvt(q, "int32", round="ROUND_F"), "float32")
                small = T.simd.vcmps(T.simd.vabs(q), T.float32(8388608), mask, op="lt")
                whole = T.simd.vsel(whole, q, small)
                residual = T.simd.alloc_local(1, "float32")
                residual[0] = av
                T.simd.vmula(residual[0], T.simd.vneg(whole), bv, mask)
                opposite = T.simd.pxor(
                    T.simd.vcmp(residual[0], zero, mask, op="lt"),
                    T.simd.vcmp(bv, zero, mask, op="lt"),
                    mask,
                )
                nonzero = T.simd.vcmp(residual[0], zero, mask, op="ne")
                finite_b = T.simd.vcmps(T.simd.vabs(bv), T.float32(float("inf")), mask, op="lt")
                correct = T.simd.pand(T.simd.pand(opposite, nonzero, mask), finite_b, mask)
                result = T.simd.vsub(whole, T.simd.vsel(one, zero, correct), mask)
                infinite_b = T.simd.vcmps(T.simd.vabs(bv), T.float32(float("inf")), mask, op="eq")
                opposite_ab = T.simd.pxor(
                    T.simd.vcmp(av, zero, mask, op="lt"), T.simd.vcmp(bv, zero, mask, op="lt"), mask
                )
                nonzero_a = T.simd.vcmp(av, zero, mask, op="ne")
                infinite_result = T.simd.vsel(
                    negative_one, q, T.simd.pand(opposite_ab, nonzero_a, mask)
                )
                result = T.simd.vsel(infinite_result, result, infinite_b)
                infinite_a = T.simd.vcmps(T.simd.vabs(av), T.float32(float("inf")), mask, op="eq")
                result = T.simd.vsel(nan, result, infinite_a)
                result = T.simd.vsel(q, result, T.simd.vcmp(bv, zero, mask, op="eq"))
            operand_nan = T.simd.por(
                T.simd.vcmp(av, av, mask, op="ne"), T.simd.vcmp(bv, bv, mask, op="ne"), mask
            )
            result = T.simd.vsel(q, result, operand_nan)
            # Conversion to integers loses signed zero. Recover only its sign.
            result = T.simd.vsel(
                T.simd.vmul(q, zero, mask), result, T.simd.vcmp(result, zero, mask, op="eq")
            )
        return result

    @T.macro
    def _body(
        a,
        b,
        c,
        width: int,
        begin,
        period: int,
        repeat: int,
        dtype: str,
        alpha: float,
        rounding_mode: str,
    ):
        with T.SimdVF():
            mask = T.simd.pset(32)
            for i in range(width // 64):
                av = (
                    T.simd.vld(a[i * 64])
                    if dtype == "float32"
                    else T.simd.vcvt(T.simd.vld(a[i * 64], dist="UNPK_B16"), "float32", part=0)
                )
                if period and repeat > 1:
                    index = ((begin + i * 64) // repeat) % period
                    bv = (
                        T.simd.vld(b[index], dist="BRC_B32")
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(b[index], dist="BRC_B16"), "float32", part=0)
                    )
                elif period:
                    index = ((begin // 64 + i) % (period // 64)) * 64
                    bv = (
                        T.simd.vld(b[index])
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(b[index], dist="UNPK_B16"), "float32", part=0)
                    )
                else:
                    bv = (
                        T.simd.vld(b[i * 64])
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(b[i * 64], dist="UNPK_B16"), "float32", part=0)
                    )
                cv = _divide(av, bv, dtype, rounding_mode, mask)
                if dtype == "float32":
                    T.simd.vsts(c[i * 64], cv, mask)
                else:
                    cv16 = T.simd.vcvt(cv, dtype)
                    T.simd.vsts(c[i * 64], cv16, mask, dist="PK_B32")

        if rounding_mode == "floor":
            _repair_floor(a, b, c, width, begin, period, repeat, dtype)

    @functools.lru_cache(maxsize=512)
    def build(
        n, dtype, period=0, repeat=1, alpha=1, tile=8192, blocks=72, stages=2, rounding_mode="true"
    ):
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1
        b_width = period or tile
        b_tail = period or tail

        @T.prim_func
        def main(
            A: T.Tensor((n,), dtype), B: T.Tensor((period or n,), dtype), C: T.Tensor((n,), dtype)
        ):
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
                            T.copy(A[begin : begin + tile], a)
                            if not period:
                                T.copy(B[begin : begin + tile], b)
                            _body(a, b, c, tile, begin, period, repeat, dtype, alpha, rounding_mode)
                            T.copy(c, C[begin : begin + tile], l2_cache_ctrl=0)
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        at = T.alloc_shared((tail,), dtype)
                        bt = T.alloc_shared((b_tail,), dtype)
                        ct = T.alloc_shared((tail,), dtype)
                        T.copy(A[full * tile : n], at)
                        if period:
                            T.copy(B, bt)
                        else:
                            T.copy(B[full * tile : n], bt)
                        _body(
                            at,
                            bt,
                            ct,
                            tail,
                            full * tile,
                            period,
                            repeat,
                            dtype,
                            alpha,
                            rounding_mode,
                        )
                        T.copy(ct, C[full * tile : n], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @T.macro
    def _channel_body(
        a, b, c, channels: int, repeat: int, dtype: str, alpha: float, b_begin, rounding_mode: str
    ):
        with T.SimdVF():
            mask = T.simd.pset(32)
            for ch in range(channels):
                bv = (
                    T.simd.vld(b[b_begin + ch], dist="BRC_B32")
                    if dtype == "float32"
                    else T.simd.vcvt(T.simd.vld(b[b_begin + ch], dist="BRC_B16"), "float32", part=0)
                )
                for i in range(repeat // 64):
                    index = ch * repeat + i * 64
                    av = (
                        T.simd.vld(a[index])
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(a[index], dist="UNPK_B16"), "float32", part=0)
                    )
                    cv = _divide(av, bv, dtype, rounding_mode, mask)
                    if dtype == "float32":
                        T.simd.vsts(c[index], cv, mask)
                    else:
                        cv16 = T.simd.vcvt(cv, dtype)
                        T.simd.vsts(c[index], cv16, mask, dist="PK_B32")

        if rounding_mode == "floor":
            _repair_floor(a, b, c, channels * repeat, b_begin, 0, repeat, dtype, True)

    @functools.lru_cache(maxsize=512)
    def build_channel(
        n, dtype, period, repeat, alpha=1, tile=4, blocks=72, stages=2, rounding_mode="true"
    ):
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
                        T.copy(A[begin : begin + width], a)
                        _channel_body(
                            a,
                            b,
                            c,
                            tile,
                            repeat,
                            dtype,
                            alpha,
                            (block * tile) % period,
                            rounding_mode,
                        )
                        T.copy(c, C[begin : begin + width], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @T.macro
    def _bias_body(a, b, c, rows: int, period: int, dtype: str, alpha: float, rounding_mode: str):
        with T.SimdVF():
            mask = T.simd.pset(32)
            for row in range(rows):
                for i in range(period // 64):
                    index = row * period + i * 64
                    av = (
                        T.simd.vld(a[index])
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(a[index], dist="UNPK_B16"), "float32", part=0)
                    )
                    bv = (
                        T.simd.vld(b[i * 64])
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(b[i * 64], dist="UNPK_B16"), "float32", part=0)
                    )
                    cv = _divide(av, bv, dtype, rounding_mode, mask)
                    if dtype == "float32":
                        T.simd.vsts(c[index], cv, mask)
                    else:
                        cv16 = T.simd.vcvt(cv, dtype)
                        T.simd.vsts(c[index], cv16, mask, dist="PK_B32")

        if rounding_mode == "floor":
            _repair_floor(a, b, c, rows * period, 0, period, 1, dtype)

    @functools.lru_cache(maxsize=256)
    def build_bias(n, dtype, period, alpha=1, rows=2, blocks=64, stages=2, rounding_mode="true"):
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
                        T.copy(A[begin : begin + width], a)
                        _bias_body(a, b, c, rows, period, dtype, alpha, rounding_mode)
                        T.copy(c, C[begin : begin + width], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)


class AscendDivKernel(ArithmeticKernelBase, BinaryElementwiseFwdInterface):
    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(
            call.a_shape, call.b_shape, call.dtype, rounding_mode=call.rounding_mode
        )

    def __init__(self, a_shape, b_shape, dtype, *, rounding_mode=None):
        if rounding_mode not in (None, "floor", "trunc"):
            raise ValueError("rounding_mode must be None, floor, or trunc")
        self.rounding_mode = rounding_mode
        super().__init__(a_shape, b_shape, dtype)

    def _compile(self, dtype_name, tile, stages):
        if dtype_name == "float32" and not self.period:
            tile, stages = 4096, 3
        scratch = 8 if self.rounding_mode == "floor" else 0
        # Mixed SIMD/SIMT scheduling reserves additional intermediate buffers.
        # Keep floor configurations below 192 KiB to leave that headroom.
        budget = (192 if scratch else 248) * 1024
        size = self.dtype.itemsize
        while (
            (
                (2 * tile + (0 if self.period else tile)) * stages
                + self.period
                + 2 * (self.padded % tile)
                + (self.period or self.padded % tile)
            )
            * size
            + scratch * (tile + self.padded % tile)
        ) > budget:
            tile //= 2
        if (
            self.period
            and self.repeat == 1
            and self.period & (self.period - 1)
            and (3 * size + scratch) * self.period <= budget
        ):
            rows = 2 if self.dtype.itemsize == 2 and self.n % (2 * self.period) == 0 else 1
            pipeline = 2
            while (
                rows * self.period * 2 * pipeline + self.period
            ) * size + scratch * rows * self.period > budget:
                if rows > 1:
                    rows //= 2
                else:
                    pipeline = 1
            return build_bias(
                self.n, dtype_name, self.period, 1, rows, 64, pipeline, self.rounding_mode or "true"
            )
        channels = 4
        if (
            self.repeat > 1
            and self.period % channels == 0
            and (channels * self.repeat * 4 + self.period) * size + scratch * channels * self.repeat
            <= budget
        ):
            return build_channel(
                self.padded,
                dtype_name,
                self.period,
                self.repeat,
                1,
                channels,
                64,
                2,
                self.rounding_mode or "true",
            )
        return build(
            self.padded,
            dtype_name,
            self.period,
            self.repeat,
            1,
            tile,
            64,
            stages,
            self.rounding_mode or "true",
        )

    def forward(self, input, other):
        # Integer promotion remains outside the floating performance manifest.
        if not self.floating:
            return torch.div(input, other, rounding_mode=self.rounding_mode)
        return super().forward(input, other)


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("DivFwdOp", "ascend", AscendDivKernel)
