"""FloorDivide with a tile predicate gating the exact large-quotient repair."""
import functools

import torch

from tileops.kernels.ascend.elementwise.arithmetic import ArithmeticKernelBase
from tileops.kernels.elementwise.call_spec import BinaryElementwiseFwdInterface

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None


_PRELUDE = r"""#include <type_traits>
#include <c_api/asc_simd.h>
__simd_callee__ inline void ew_floor_fast_inplace(vector_f32 &dst, vector_bool mask) {
    vtrc(dst, dst, std::integral_constant<::ROUND, ROUND_F>(), mask, MODE_ZEROING);
}
"""

if T is not None:
    from tileops.kernels.ascend.elementwise.arithmetic_div import _divide
    from tileops.kernels.ascend.elementwise.arithmetic_modulo import _widen_input, _narrow_output

    @T.macro
    def _quick_floor(av, bv, q, dtype: str, mask):
        whole = T.simd.alloc_local(1, "float32")
        whole[0] = q
        T.evaluate(T.call_extern("void", "ew_floor_fast_inplace", whole[0], mask))
        if dtype == "float32":
            zero = T.simd.vdup(T.float32(0), "float32", mask)
            residual = T.simd.alloc_local(1, "float32")
            residual[0] = av
            T.simd.vmula(residual[0], T.simd.vneg(whole[0]), bv, mask)
            opposite = T.simd.pxor(T.simd.vcmp(residual[0], zero, mask, op="lt"), T.simd.vcmp(bv, zero, mask, op="lt"), mask)
            correct = T.simd.pand(opposite, T.simd.vcmp(residual[0], zero, mask, op="ne"), mask)
            return T.simd.vsub(whole[0], T.simd.vsel(T.simd.vdup(T.float32(1), "float32", mask), zero, correct), mask)
        else:
            return whole[0]

    @T.macro
    def _repair_floor(a, b, c, width: int, begin, period: int, repeat: int,
                      dtype: str, repair_guard, channel: bool = False, math_dtype: str = ""):
        # Scalar libm supplies the remainder; staged vector division keeps
        # the FP32 quotient exact, including tiny divisors.
        compute_dtype = dtype if math_dtype == "" else math_dtype
        limit = 8192 if compute_dtype == "float16" else 65536 if compute_dtype == "bfloat16" else 4194304
        numerator = T.alloc_shared((width,), "float32")
        adjustment = T.alloc_shared((width,), "float32")
        T.annotate_buffer_versions({numerator: 1, adjustment: 1})
        repair_threads = 256 if width % 256 == 0 else 128 if width % 128 == 0 else 64
        with T.SimtVF(threads=repair_threads):
            if repair_guard[0] != T.uint32(0):
                repair_thread = T.get_thread_binding()
                for repair_step in T.serial(width // repair_threads):
                    j = repair_step * repair_threads + repair_thread
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
                        adjustment[j] = T.if_then_else(T.And(rem != 0, (scalar_b < 0) != (rem < 0)), T.float32(1), T.float32(0))
        with T.SimdVF():
            if repair_guard[0] != T.uint32(0):
                mask = T.simd.pset(32)
                zero = T.simd.vdup(T.float32(0), "float32", mask)
                for i in range(width // 64):
                    repair_a = T.simd.vld(a[i * 64]) if dtype == "float32" else T.simd.vcvt(T.simd.vld(a[i * 64], dist="UNPK_B16"), "float32", part=0)
                    if channel:
                        index = begin + i * 64 // repeat
                        repair_b = T.simd.vld(b[index], dist="BRC_B32") if dtype == "float32" else T.simd.vcvt(T.simd.vld(b[index], dist="BRC_B16"), "float32", part=0)
                    elif period and repeat > 1:
                        index = ((begin + i * 64) // repeat) % period
                        repair_b = T.simd.vld(b[index], dist="BRC_B32") if dtype == "float32" else T.simd.vcvt(T.simd.vld(b[index], dist="BRC_B16"), "float32", part=0)
                    elif period:
                        index = ((begin // 64 + i) % (period // 64)) * 64
                        repair_b = T.simd.vld(b[index]) if dtype == "float32" else T.simd.vcvt(T.simd.vld(b[index], dist="UNPK_B16"), "float32", part=0)
                    else:
                        repair_b = T.simd.vld(b[i * 64]) if dtype == "float32" else T.simd.vcvt(T.simd.vld(b[i * 64], dist="UNPK_B16"), "float32", part=0)
                    q = _divide(T.simd.vld(numerator[i * 64]), repair_b, "float32", "true", mask)
                    div = T.simd.vsub(q, T.simd.vld(adjustment[i * 64]), mask)
                    whole = T.simd.vcvt(T.simd.vcvt(div, "int32", round="ROUND_F"), "float32")
                    small = T.simd.vcmps(T.simd.vabs(div), T.float32(8388608), mask, op="lt")
                    whole = T.simd.vsel(whole, div, small)
                    if compute_dtype != "float32":
                        whole = T.simd.vcvt(T.simd.vcvt(whole, compute_dtype), "float32", part=0)
                    bumped = T.simd.vadds(whole, T.float32(1), mask)
                    if compute_dtype != "float32":
                        bumped = T.simd.vcvt(T.simd.vcvt(bumped, compute_dtype), "float32", part=0)
                    up = T.simd.vcmps(T.simd.vsub(div, whole, mask), T.float32(0.5), mask, op="gt")
                    result = T.simd.vsel(bumped, whole, up)
                    eligible = T.simd.vcmp(T.simd.vabs(repair_a), T.simd.vmuls(T.simd.vabs(repair_b), T.float32(limit), mask), mask, op="ge")
                    eligible = T.simd.pand(eligible, T.simd.vcmp(repair_b, zero, mask, op="ne"), mask)
                    eligible = T.simd.pand(eligible, T.simd.vcmp(div, zero, mask, op="ne"), mask)
                    old = T.simd.vld(c[i * 64]) if dtype == "float32" else T.simd.vcvt(T.simd.vld(c[i * 64], dist="UNPK_B16"), "float32", part=0)
                    result = T.simd.vsel(result, old, eligible)
                    if dtype == "float32":
                        T.simd.vsts(c[i * 64], result, mask)
                    else:
                        T.simd.vsts(c[i * 64], T.simd.vcvt(result, dtype), mask, dist="PK_B32")

    @T.macro
    def _body(a, b, c, width: int, begin, period: int, repeat: int,
              dtype: str, alpha: float, rounding_mode: str):
        fast_guard = T.alloc_shared((192,), "uint32")
        T.annotate_buffer_versions({fast_guard: 1})
        fast_limit_bits = 0x46000000 if dtype == "float16" else 0x47800000 if dtype == "bfloat16" else 0x4a800000
        repair_guard = T.alloc_shared((64,), "uint32")
        T.annotate_buffer_versions({repair_guard: 1})
        repair_limit = 8192 if dtype == "float16" else 65536 if dtype == "bfloat16" else 4194304
        with T.SimdVF():
            mask = T.simd.pset(32)
            repair_any = T.simd.alloc_local(1, "uint32")
            repair_any[0] = T.simd.vdup(T.uint32(0), "uint32", mask)
            fast_min_q = T.simd.alloc_local(1, "uint32")
            fast_max_q = T.simd.alloc_local(1, "uint32")
            fast_min_b = T.simd.alloc_local(1, "uint32")
            fast_min_q[0] = T.simd.vdup(T.uint32(0x7f800000), "uint32", mask)
            fast_max_q[0] = T.simd.vdup(T.uint32(0), "uint32", mask)
            fast_min_b[0] = T.simd.vdup(T.uint32(0x7f800000), "uint32", mask)
            for fast_i in range(width // 64):
                fast_av = T.simd.vld(a[fast_i * 64]) if dtype == 'float32' else T.simd.vcvt(T.simd.vld(a[fast_i * 64], dist='UNPK_B16'), 'float32', part=0)
                if period and repeat > 1:
                    fast_index = ((begin + fast_i * 64) // repeat) % period
                    fast_bv = T.simd.vld(b[fast_index], dist='BRC_B32') if dtype == 'float32' else T.simd.vcvt(T.simd.vld(b[fast_index], dist='BRC_B16'), 'float32', part=0)
                elif period:
                    fast_index = ((begin // 64 + fast_i) % (period // 64)) * 64
                    fast_bv = T.simd.vld(b[fast_index]) if dtype == 'float32' else T.simd.vcvt(T.simd.vld(b[fast_index], dist='UNPK_B16'), 'float32', part=0)
                else:
                    fast_bv = T.simd.vld(b[fast_i * 64]) if dtype == 'float32' else T.simd.vcvt(T.simd.vld(b[fast_i * 64], dist='UNPK_B16'), 'float32', part=0)
                fast_q = T.simd.vdiv(fast_av, fast_bv, mask, precision="exact")
                fast_q_bits = T.reinterpret(fast_q, "uint32x64")
                fast_min_q[0] = T.simd.vmin(fast_min_q[0], fast_q_bits, mask)
                fast_max_q[0] = T.simd.vmax(fast_max_q[0], fast_q_bits, mask)
                fast_min_b[0] = T.simd.vmin(fast_min_b[0], T.reinterpret(T.simd.vabs(fast_bv), "uint32x64"), mask)
                fast_cv = _quick_floor(fast_av, fast_bv, fast_q, dtype, mask)
                if dtype == 'float32':
                    T.simd.vsts(c[fast_i * 64], fast_cv, mask)
                else:
                    fast_cv16 = T.simd.vcvt(fast_cv, dtype)
                    T.simd.vsts(c[fast_i * 64], fast_cv16, mask, dist='PK_B32')
            T.simd.vsts(fast_guard[0], T.simd.vcmin(fast_min_q[0], mask), mask)
            T.simd.vsts(fast_guard[64], T.simd.vcmax(fast_max_q[0], mask), mask)
            T.simd.vsts(fast_guard[128], T.simd.vcmin(fast_min_b[0], mask), mask)
            T.simd.mem_bar("VST_LD")
            # Positive bounded quotients have an exact floor tier. A divisor
            # >= 2**-100 keeps a nonzero FP32 FMA residual above the FTZ range.
            # Every other tile retains the full original arithmetic and repair.
            if T.Or(T.Or(fast_guard[0] == T.uint32(0), fast_guard[64] >= T.uint32(fast_limit_bits)), fast_guard[128] < T.uint32(0x0d800000)):
                for i in range(width // 64):
                    av = T.simd.vld(a[i * 64]) if dtype == 'float32' else T.simd.vcvt(T.simd.vld(a[i * 64], dist='UNPK_B16'), 'float32', part=0)
                    if period and repeat > 1:
                        index = ((begin + i * 64) // repeat) % period
                        bv = T.simd.vld(b[index], dist='BRC_B32') if dtype == 'float32' else T.simd.vcvt(T.simd.vld(b[index], dist='BRC_B16'), 'float32', part=0)
                    elif period:
                        index = ((begin // 64 + i) % (period // 64)) * 64
                        bv = T.simd.vld(b[index]) if dtype == 'float32' else T.simd.vcvt(T.simd.vld(b[index], dist='UNPK_B16'), 'float32', part=0)
                    else:
                        bv = T.simd.vld(b[i * 64]) if dtype == 'float32' else T.simd.vcvt(T.simd.vld(b[i * 64], dist='UNPK_B16'), 'float32', part=0)
                    needs_repair = T.simd.vcmp(T.simd.vabs(av), T.simd.vmuls(T.simd.vabs(bv), T.float32(repair_limit), mask), mask, op="ge")
                    repair_bits = T.simd.vsel(T.simd.vdup(T.uint32(1), "uint32", mask), T.simd.vdup(T.uint32(0), "uint32", mask), needs_repair)
                    repair_any[0] = T.simd.vmax(repair_any[0], repair_bits, mask)
                    cv = _divide(av, bv, dtype, rounding_mode, mask)
                    if dtype == 'float32':
                        T.simd.vsts(c[i * 64], cv, mask)
                    else:
                        cv16 = T.simd.vcvt(cv, dtype)
                        T.simd.vsts(c[i * 64], cv16, mask, dist='PK_B32')
            T.simd.vsts(repair_guard[0], T.simd.vcmax(repair_any[0], mask), mask)

        if rounding_mode == "floor":
            _repair_floor(a, b, c, width, begin, period, repeat, dtype, repair_guard)


    @functools.lru_cache(maxsize=512)
    def build(n, dtype, period=0, repeat=1, alpha=1, tile=8192, blocks=72, stages=2, rounding_mode="true"):
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1
        b_width = period or tile
        b_tail = period or tail

        @T.prim_func
        def main(A: T.Tensor((n,), dtype), B: T.Tensor((period or n,), dtype), C: T.Tensor((n,), dtype)):
            with T.Kernel(grid, prelude=_PRELUDE) as bx:
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
                            _body(a, b, c, tile, begin, period, repeat, dtype, alpha, rounding_mode)
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
                        _body(at, bt, ct, tail, full * tile, period, repeat, dtype, alpha, rounding_mode)
                        T.copy(ct, C[full * tile:n], l2_cache_ctrl=0)
        return tilelang.compile(main, target='ascend', out_idx=-1)

    @T.macro
    def _channel_body(a, b, c, channels: int, repeat: int,
                      dtype: str, alpha: float, b_begin, rounding_mode: str, math_dtype: str = ""):
        compute_dtype = dtype if math_dtype == "" else math_dtype
        fast_guard = T.alloc_shared((192,), "uint32")
        T.annotate_buffer_versions({fast_guard: 1})
        fast_limit_bits = 0x46000000 if compute_dtype == "float16" else 0x47800000 if compute_dtype == "bfloat16" else 0x4a800000
        repair_guard = T.alloc_shared((64,), "uint32")
        T.annotate_buffer_versions({repair_guard: 1})
        repair_limit = 8192 if compute_dtype == "float16" else 65536 if compute_dtype == "bfloat16" else 4194304
        with T.SimdVF():
            mask = T.simd.pset(32)
            repair_any = T.simd.alloc_local(1, "uint32")
            repair_any[0] = T.simd.vdup(T.uint32(0), "uint32", mask)
            fast_min_q = T.simd.alloc_local(1, "uint32")
            fast_max_q = T.simd.alloc_local(1, "uint32")
            fast_min_b = T.simd.alloc_local(1, "uint32")
            fast_min_q[0] = T.simd.vdup(T.uint32(0x7f800000), "uint32", mask)
            fast_max_q[0] = T.simd.vdup(T.uint32(0), "uint32", mask)
            fast_min_b[0] = T.simd.vdup(T.uint32(0x7f800000), "uint32", mask)
            for fast_ch in range(channels):
                fast_bv = T.simd.vld(b[b_begin + fast_ch], dist='BRC_B32') if dtype == 'float32' else T.simd.vcvt(T.simd.vld(b[b_begin + fast_ch], dist='BRC_B16'), 'float32', part=0)
                for fast_i in range(repeat // 64):
                    fast_index = fast_ch * repeat + fast_i * 64
                    fast_av = T.simd.vld(a[fast_index]) if dtype == 'float32' else T.simd.vcvt(T.simd.vld(a[fast_index], dist='UNPK_B16'), 'float32', part=0)
                    fast_q = T.simd.vdiv(fast_av, fast_bv, mask, precision="exact")
                    fast_q_bits = T.reinterpret(fast_q, "uint32x64")
                    fast_min_q[0] = T.simd.vmin(fast_min_q[0], fast_q_bits, mask)
                    fast_max_q[0] = T.simd.vmax(fast_max_q[0], fast_q_bits, mask)
                    fast_min_b[0] = T.simd.vmin(fast_min_b[0], T.reinterpret(T.simd.vabs(fast_bv), "uint32x64"), mask)
                    fast_cv = _quick_floor(fast_av, fast_bv, fast_q, compute_dtype, mask)
                    if dtype == 'float32':
                        T.simd.vsts(c[fast_index], fast_cv, mask)
                    else:
                        fast_cv16 = T.simd.vcvt(fast_cv, dtype)
                        T.simd.vsts(c[fast_index], fast_cv16, mask, dist='PK_B32')
            T.simd.vsts(fast_guard[0], T.simd.vcmin(fast_min_q[0], mask), mask)
            T.simd.vsts(fast_guard[64], T.simd.vcmax(fast_max_q[0], mask), mask)
            T.simd.vsts(fast_guard[128], T.simd.vcmin(fast_min_b[0], mask), mask)
            T.simd.mem_bar("VST_LD")
            # Positive bounded quotients have an exact floor tier. A divisor
            # >= 2**-100 keeps a nonzero FP32 FMA residual above the FTZ range.
            # Every other tile retains the full original arithmetic and repair.
            if T.Or(T.Or(fast_guard[0] == T.uint32(0), fast_guard[64] >= T.uint32(fast_limit_bits)), fast_guard[128] < T.uint32(0x0d800000)):
                for ch in range(channels):
                    bv = T.simd.vld(b[b_begin + ch], dist='BRC_B32') if dtype == 'float32' else T.simd.vcvt(T.simd.vld(b[b_begin + ch], dist='BRC_B16'), 'float32', part=0)
                    for i in range(repeat // 64):
                        index = ch * repeat + i * 64
                        av = T.simd.vld(a[index]) if dtype == 'float32' else T.simd.vcvt(T.simd.vld(a[index], dist='UNPK_B16'), 'float32', part=0)
                        needs_repair = T.simd.vcmp(T.simd.vabs(av), T.simd.vmuls(T.simd.vabs(bv), T.float32(repair_limit), mask), mask, op="ge")
                        repair_bits = T.simd.vsel(T.simd.vdup(T.uint32(1), "uint32", mask), T.simd.vdup(T.uint32(0), "uint32", mask), needs_repair)
                        repair_any[0] = T.simd.vmax(repair_any[0], repair_bits, mask)
                        cv = _divide(av, bv, compute_dtype, rounding_mode, mask)
                        if dtype == 'float32':
                            T.simd.vsts(c[index], cv, mask)
                        else:
                            cv16 = T.simd.vcvt(cv, dtype)
                            T.simd.vsts(c[index], cv16, mask, dist='PK_B32')
            T.simd.vsts(repair_guard[0], T.simd.vcmax(repair_any[0], mask), mask)

        if rounding_mode == "floor":
            _repair_floor(a, b, c, channels * repeat, b_begin, 0, repeat, dtype, repair_guard, True, compute_dtype)


    @functools.lru_cache(maxsize=512)
    def build_channel(n, dtype, period, repeat, alpha=1, tile=4, blocks=72, stages=2, rounding_mode="true"):
        width = tile * repeat
        count = n // width
        grid = min(blocks, count)
        assert repeat % 64 == 0 and period % tile == 0 and n % width == 0

        @T.prim_func
        def main(A: T.Tensor((n,), dtype), B: T.Tensor((period,), dtype), C: T.Tensor((n,), dtype)):
            with T.Kernel(grid, prelude=_PRELUDE) as bx:
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
                        _channel_body(a, b, c, tile, repeat, dtype, alpha, (block * tile) % period, rounding_mode)
                        T.copy(c, C[begin:begin + width], l2_cache_ctrl=0)
        return tilelang.compile(main, target='ascend', out_idx=-1)

    @T.macro
    def _bias_body(a, b, c, rows: int, period: int, dtype: str, alpha: float, rounding_mode: str):
        fast_guard = T.alloc_shared((192,), "uint32")
        T.annotate_buffer_versions({fast_guard: 1})
        fast_limit_bits = 0x46000000 if dtype == "float16" else 0x47800000 if dtype == "bfloat16" else 0x4a800000
        repair_guard = T.alloc_shared((64,), "uint32")
        T.annotate_buffer_versions({repair_guard: 1})
        repair_limit = 8192 if dtype == "float16" else 65536 if dtype == "bfloat16" else 4194304
        with T.SimdVF():
            mask = T.simd.pset(32)
            repair_any = T.simd.alloc_local(1, "uint32")
            repair_any[0] = T.simd.vdup(T.uint32(0), "uint32", mask)
            fast_min_q = T.simd.alloc_local(1, "uint32")
            fast_max_q = T.simd.alloc_local(1, "uint32")
            fast_min_b = T.simd.alloc_local(1, "uint32")
            fast_min_q[0] = T.simd.vdup(T.uint32(0x7f800000), "uint32", mask)
            fast_max_q[0] = T.simd.vdup(T.uint32(0), "uint32", mask)
            fast_min_b[0] = T.simd.vdup(T.uint32(0x7f800000), "uint32", mask)
            for fast_row in range(rows):
                for fast_i in range(period // 64):
                    fast_index = fast_row * period + fast_i * 64
                    fast_av = T.simd.vld(a[fast_index]) if dtype == "float32" else T.simd.vcvt(T.simd.vld(a[fast_index], dist="UNPK_B16"), "float32", part=0)
                    fast_bv = T.simd.vld(b[fast_i * 64]) if dtype == "float32" else T.simd.vcvt(T.simd.vld(b[fast_i * 64], dist="UNPK_B16"), "float32", part=0)
                    fast_q = T.simd.vdiv(fast_av, fast_bv, mask, precision="exact")
                    fast_q_bits = T.reinterpret(fast_q, "uint32x64")
                    fast_min_q[0] = T.simd.vmin(fast_min_q[0], fast_q_bits, mask)
                    fast_max_q[0] = T.simd.vmax(fast_max_q[0], fast_q_bits, mask)
                    fast_min_b[0] = T.simd.vmin(fast_min_b[0], T.reinterpret(T.simd.vabs(fast_bv), "uint32x64"), mask)
                    fast_cv = _quick_floor(fast_av, fast_bv, fast_q, dtype, mask)
                    if dtype == "float32":
                        T.simd.vsts(c[fast_index], fast_cv, mask)
                    else:
                        fast_cv16 = T.simd.vcvt(fast_cv, dtype)
                        T.simd.vsts(c[fast_index], fast_cv16, mask, dist="PK_B32")
            T.simd.vsts(fast_guard[0], T.simd.vcmin(fast_min_q[0], mask), mask)
            T.simd.vsts(fast_guard[64], T.simd.vcmax(fast_max_q[0], mask), mask)
            T.simd.vsts(fast_guard[128], T.simd.vcmin(fast_min_b[0], mask), mask)
            T.simd.mem_bar("VST_LD")
            # Positive bounded quotients have an exact floor tier. A divisor
            # >= 2**-100 keeps a nonzero FP32 FMA residual above the FTZ range.
            # Every other tile retains the full original arithmetic and repair.
            if T.Or(T.Or(fast_guard[0] == T.uint32(0), fast_guard[64] >= T.uint32(fast_limit_bits)), fast_guard[128] < T.uint32(0x0d800000)):
                for row in range(rows):
                    for i in range(period // 64):
                        index = row * period + i * 64
                        av = T.simd.vld(a[index]) if dtype == "float32" else T.simd.vcvt(T.simd.vld(a[index], dist="UNPK_B16"), "float32", part=0)
                        bv = T.simd.vld(b[i * 64]) if dtype == "float32" else T.simd.vcvt(T.simd.vld(b[i * 64], dist="UNPK_B16"), "float32", part=0)
                        needs_repair = T.simd.vcmp(T.simd.vabs(av), T.simd.vmuls(T.simd.vabs(bv), T.float32(repair_limit), mask), mask, op="ge")
                        repair_bits = T.simd.vsel(T.simd.vdup(T.uint32(1), "uint32", mask), T.simd.vdup(T.uint32(0), "uint32", mask), needs_repair)
                        repair_any[0] = T.simd.vmax(repair_any[0], repair_bits, mask)
                        cv = _divide(av, bv, dtype, rounding_mode, mask)
                        if dtype == "float32":
                            T.simd.vsts(c[index], cv, mask)
                        else:
                            cv16 = T.simd.vcvt(cv, dtype)
                            T.simd.vsts(c[index], cv16, mask, dist="PK_B32")
            T.simd.vsts(repair_guard[0], T.simd.vcmax(repair_any[0], mask), mask)

        if rounding_mode == "floor":
            _repair_floor(a, b, c, rows * period, 0, period, 1, dtype, repair_guard)


    @functools.lru_cache(maxsize=256)
    def build_bias(n, dtype, period, alpha=1, rows=2, blocks=64, stages=2, rounding_mode="true"):
        width = rows * period
        count = n // width
        grid = min(blocks, count)
        assert n % width == 0 and period % 64 == 0

        @T.prim_func
        def main(A: T.Tensor((n,), dtype), B: T.Tensor((period,), dtype), C: T.Tensor((n,), dtype)):
            with T.Kernel(grid, prelude=_PRELUDE) as bx:
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
                        _bias_body(a, b, c, rows, period, dtype, alpha, rounding_mode)
                        T.copy(c, C[begin:begin + width], l2_cache_ctrl=0)
        return tilelang.compile(main, target="ascend", out_idx=-1)


    @functools.lru_cache(maxsize=256)
    def build_widened_channel(n, dtype, period, repeat, tile=4, blocks=64, stages=2, rounding_mode="floor_refined", repair_threads=256):
        width = tile * repeat
        count = n // width
        grid = min(blocks, count)
        assert dtype in ("float16", "bfloat16") and period % 64 == 0
        assert repeat % 64 == 0 and period % tile == 0 and n % width == 0
        assert 4 * width * stages + 16 * width + 6 * period + 2048 <= 192 * 1024

        @T.prim_func
        def main(A: T.Tensor((n,), dtype), B: T.Tensor((period,), dtype), C: T.Tensor((n,), dtype)):
            with T.Kernel(grid, prelude=_PRELUDE) as bx:
                a = T.alloc_shared((width,), dtype)
                b = T.alloc_shared((period,), dtype)
                c = T.alloc_shared((width,), dtype)
                af = T.alloc_shared((width,), "float32")
                bf = T.alloc_shared((period,), "float32")
                cf = T.alloc_shared((width,), "float32")
                T.annotate_buffer_versions({a: stages, c: stages, af: 1, bf: 1, cf: 1})
                T.copy(B, b)
                _widen_input(b, bf, period)
                for step in T.Pipelined((count + grid - 1) // grid, num_stages=stages):
                    block = step * grid + bx
                    if block < count:
                        begin = block * width
                        T.copy(A[begin:begin + width], a)
                        _widen_input(a, af, width)
                        _channel_body(af, bf, cf, tile, repeat, "float32", 1, (block * tile) % period, rounding_mode, dtype)
                        _narrow_output(cf, c, width, dtype)
                        T.copy(c, C[begin:begin + width], l2_cache_ctrl=0)
        return tilelang.compile(main, target="ascend", out_idx=-1)


class FloorDivideImplementation(ArithmeticKernelBase, BinaryElementwiseFwdInterface):
    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.a_shape, call.b_shape, call.dtype, rounding_mode=call.rounding_mode)

    def __init__(self, a_shape, b_shape, dtype, *, rounding_mode=None):
        if rounding_mode not in (None, "floor", "trunc"):
            raise ValueError("rounding_mode must be None, floor, or trunc")
        self.rounding_mode = rounding_mode
        super().__init__(a_shape, b_shape, dtype)

    def _compile(self, dtype_name, tile, stages):
        # Larger full tiles amortize the guarded VF dispatches. Keep enough
        # room for the exact fallback even when the tile needs repair.
        tile, stages = (4096, 3) if dtype_name == "float32" else (8192, 2)
        scratch = 8 if self.rounding_mode == "floor" else 0
        # Mixed SIMD/SIMT scheduling reserves additional intermediate buffers.
        # Keep floor configurations below 192 KiB to leave that headroom.
        budget = (192 if scratch else 248) * 1024
        size = self.dtype.itemsize
        while (((2 * tile + (0 if self.period else tile)) * stages
                + self.period + 2 * (self.padded % tile) + (self.period or self.padded % tile)) * size
               + scratch * (tile + self.padded % tile)) > budget:
            tile //= 2
        if (self.period and self.repeat == 1 and self.period & (self.period - 1)
                and (3 * size + scratch) * self.period <= budget):
            rows = 2 if self.dtype.itemsize == 2 and self.n % (2 * self.period) == 0 else 1
            pipeline = 2
            while (rows * self.period * 2 * pipeline + self.period) * size + scratch * rows * self.period > budget:
                if rows > 1:
                    rows //= 2
                else:
                    pipeline = 1
            return build_bias(self.n, dtype_name, self.period, 1, rows, 64, pipeline, self.rounding_mode or "true")
        if self.dtype.itemsize == 2 and self.repeat > 1 and self.period % 64 == 0:
            widened_channels = 4
            while widened_channels > 1 and (self.period % widened_channels != 0 or
                    24 * widened_channels * self.repeat + 6 * self.period + 2048 > budget):
                widened_channels //= 2
            if 24 * widened_channels * self.repeat + 6 * self.period + 2048 <= budget:
                return build_widened_channel(self.padded, dtype_name, self.period, self.repeat, widened_channels, 64, 2, self.rounding_mode or "true")
        channels = 4
        if self.repeat > 1:
            while channels > 1 and (self.period % channels != 0 or
                    (channels * self.repeat * 4 + self.period) * size + scratch * channels * self.repeat > budget):
                channels //= 2
            if (channels * self.repeat * 4 + self.period) * size + scratch * channels * self.repeat <= budget:
                return build_channel(self.padded, dtype_name, self.period, self.repeat, 1, channels, 64, 2, self.rounding_mode or "true")
        return build(self.padded, dtype_name, self.period, self.repeat, 1, tile, 64, stages, self.rounding_mode or "true")

    def forward(self, input, other):
        # Integer promotion remains outside the floating performance manifest.
        if not self.floating:
            return torch.div(input, other, rounding_mode=self.rounding_mode)
        return super().forward(input, other)

