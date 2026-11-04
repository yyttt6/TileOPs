"""Row-warp InstanceNorm with common-scale compensated FP32 statistics."""
import functools, struct
import tilelang
import tilelang.ascend.language as T
from . import _instance_untracked as original

def supported(batch, channels, n, dtype, weighted, biased):
    size = 4 if dtype == 'float32' else 2
    pitch = max(16, (n + 15) // 16 * 16)
    return 1 < n <= 8192 and 4 * pitch * size + 4 * pitch + 2 * channels * size + 4096 < 192 * 1024

@functools.lru_cache(maxsize=128)
def build_fp32(batch, channels, spatial, groups, dtype, eps, weighted, biased):
    assert channels == groups and supported(batch, channels, spatial, dtype, weighted, biased)
    m = batch * channels
    n = spatial
    size = 4
    padded = max(16, (n + 15) // 16 * 16)
    rows = 8 if m % 8 == 0 and 32 * padded * size + 32 * padded + 2 * channels * size + 4096 < 192 * 1024 else 1
    if (weighted or biased) and m >= 1024 and (m % 16 == 0) and (n >= 512) and (48 * padded * size + 2 * channels * size + 4096 < 192 * 1024):
        rows = 16
    stages = 1 if rows == 16 else 2
    count = m // rows
    blocks = min(64, count)
    width = max(32, tilelang.next_power_of_2(n))
    pitch = max(16, (n + 15) // 16 * 16)
    zero_inverse = 1 / eps ** 0.5
    eps_hi = struct.unpack('f', struct.pack('f', eps))[0]
    eps_lo = eps - eps_hi
    raw_n = struct.unpack('I', struct.pack('f', float(n)))[0]
    rounded_n = raw_n + 32767 + (raw_n >> 16 & 1) & 4294901760
    bf_n = struct.unpack('f', struct.pack('I', rounded_n))[0]
    lanes = 32 if rows >= 8 else min(256, width)
    row_layout = T.Fragment((rows, width), forward_thread_fn=lambda r, i: r * lanes + i % lanes, forward_index_fn=lambda r, i: i // lanes)
    stat_layout = T.Fragment((rows, lanes), forward_thread_fn=lambda r, i: r * lanes + i, forward_index_fn=lambda r, i: 0)

    @T.prim_func
    def main(X: T.Tensor((m, n), dtype), W: T.Tensor((channels,), dtype), B: T.Tensor((channels,), dtype), Y: T.Tensor((m, n), dtype)):
        with T.Kernel(blocks) as bx:
            x = T.alloc_shared((rows, pitch), dtype)
            y = T.alloc_shared((rows, pitch), dtype)
            w = T.alloc_shared((channels,), dtype)
            b = T.alloc_shared((channels,), dtype)
            products = T.alloc_shared((rows, pitch), 'float32')
            means = T.alloc_shared((rows,), 'float32')
            maxima = T.alloc_shared((rows,), 'float32')
            offsets = T.alloc_shared((rows,), 'float32')
            alphas = T.alloc_shared((rows,), 'float32')
            if rows != 16:
                T.annotate_buffer_versions({x: 2, y: 2})
            if weighted:
                T.copy(W[:channels], w[:channels])
            if biased:
                T.copy(B[:channels], b[:channels])
            for step in T.Pipelined((count + blocks - 1) // blocks, num_stages=stages):
                first = (step * blocks + bx) * rows
                if first < m:
                    T.copy(X[first:first + rows, :n], x[:rows, :n])
                    with T.SimtVF(threads=rows * lanes):
                        total = T.alloc_reducer((rows,), 'float32', op='sum')
                        T.reducer_init(total)
                        lower = T.alloc_reducer((rows,), 'float32', op='sum')
                        T.reducer_init(lower)
                        for r, i in T.Parallel(rows, width, loop_layout=row_layout):
                            if i < n:
                                hi = T.reinterpret(T.reinterpret(x[r, i], 'uint32') & T.uint32(4294959104), 'float32')
                                T.reducer_update(total[r], hi)
                                T.reducer_update(lower[r], x[r, i] - hi)
                        sums = T.alloc_fragment((rows,), 'float32')
                        T.finalize_reducer(total, sums)
                        lows = T.alloc_fragment((rows,), 'float32')
                        T.finalize_reducer(lower, lows)
                        maximum = T.alloc_reducer((rows,), 'float32', op='max')
                        T.reducer_init(maximum)
                        squares = T.alloc_fragment((rows, width), 'float32')
                        rowmeans = T.alloc_fragment((rows, lanes), 'float32')
                        for r, i in T.Parallel(rows, lanes, loop_layout=stat_layout):
                            full_sum = T.call_extern('float32', 'fmaf', T.float32(1), sums[r], lows[r])
                            lost_sum = T.call_extern('float32', 'fmaf', T.float32(-1), full_sum, sums[r]) + lows[r]
                            mean_q = full_sum / n
                            mean_r = T.call_extern('float32', 'fmaf', -mean_q, T.float32(n), full_sum) + lost_sum
                            mean_corrected = T.call_extern('float32', 'fmaf', T.float32(1.0 / n), mean_r, mean_q)
                            mean_bits = T.reinterpret(mean_q, 'uint32')
                            neighbor_bits = T.if_then_else(mean_q == 0, T.if_then_else(mean_r < 0, T.uint32(2147483649), T.uint32(1)), T.if_then_else((mean_q > 0) == (mean_r > 0), mean_bits + T.uint32(1), mean_bits - T.uint32(1)))
                            neighbor = T.reinterpret(neighbor_bits, 'float32')
                            midpoint = (neighbor - mean_q) * T.float32(n * 0.5)
                            mean = T.if_then_else(mean_r == midpoint, T.if_then_else(mean_bits & T.uint32(1) == 0, mean_q, neighbor), mean_corrected)
                            rowmeans[r, i] = mean
                        for r, i in T.Parallel(rows, width, loop_layout=row_layout):
                            if i < n:
                                centered = T.cast(x[r, i], 'float32') - rowmeans[r, i % lanes]
                                square = centered * centered
                                squares[r, i] = square
                                T.reducer_update(maximum[r], square)
                        maxs = T.alloc_fragment((rows,), 'float32')
                        T.finalize_reducer(maximum, maxs)
                        for r, i in T.Parallel(rows, lanes, loop_layout=stat_layout):
                            if i == 0:
                                means[r] = rowmeans[r, i]
                                maxima[r] = maxs[r]
                        for r, i in T.Parallel(rows, width, loop_layout=row_layout):
                            if i < n:
                                products[r, i] = squares[r, i]
                    with T.SimtVF(threads=rows * lanes):
                        variance = T.alloc_reducer((rows,), 'float32', op='sum')
                        T.reducer_init(variance)
                        remainder = T.alloc_reducer((rows,), 'float32', op='sum')
                        T.reducer_init(remainder)
                        for r, i in T.Parallel(rows, width, loop_layout=row_layout):
                            if i < n:
                                square = products[r, i]
                                big = maxima[r] * 4096
                                shifted = T.call_extern('float32', 'fmaf', T.float32(1), square, big)
                                upper = T.call_extern('float32', 'fmaf', T.float32(-1), big, shifted)
                                T.reducer_update(variance[r], upper)
                                T.reducer_update(remainder[r], square - upper)
                        vars = T.alloc_fragment((rows,), 'float32')
                        T.finalize_reducer(variance, vars)
                        remainders = T.alloc_fragment((rows,), 'float32')
                        T.finalize_reducer(remainder, remainders)
                        values = T.alloc_fragment((rows, width), 'float32')
                        offset_values = T.alloc_fragment((rows, width), 'float32')
                        alpha_values = T.alloc_fragment((rows, width), 'float32')
                        for r, i in T.Parallel(rows, width, loop_layout=row_layout):
                            if i < n:
                                var_sum = vars[r] + remainders[r]
                                quotient = var_sum / n
                                residual = T.call_extern('float32', 'fmaf', -quotient, T.float32(n), var_sum)
                                v = T.call_extern('float32', 'fmaf', T.float32(1.0 / n), residual, quotient)
                                argument = v + T.float32(eps_hi)
                                displaced = T.call_extern('float32', 'fmaf', T.float32(-1), v, argument)
                                low = T.call_extern('float32', 'fmaf', T.float32(-1), T.call_extern('float32', 'fmaf', T.float32(-1), displaced, argument), v) + T.call_extern('float32', 'fmaf', T.float32(-1), displaced, T.float32(eps_hi)) + T.float32(eps_lo)
                                estimate = T.rsqrt(argument)
                                sq = T.call_extern('float32', 'fmaf', estimate, estimate, T.float32(0))
                                sq_error = T.call_extern('float32', 'fmaf', estimate, estimate, -sq)
                                err = T.call_extern('float32', 'fmaf', -argument, sq, T.float32(1)) - argument * sq_error - low * sq
                                corrected = T.call_extern('float32', 'fmaf', T.float32(0.5) * estimate, err, estimate)
                                inverse = T.if_then_else(v == 0, T.float32(zero_inverse), corrected)
                                rounded_mean = T.cast(T.cast(means[r], dtype), 'float32')
                                rounded_inverse = T.cast(T.cast(inverse, dtype), 'float32')
                                weight = T.cast(w[(first + r) % channels], 'float32') if weighted else T.float32(1)
                                bias = T.cast(b[(first + r) % channels], 'float32') if biased else T.float32(0)
                                alpha = rounded_inverse * weight
                                beta = T.call_extern('float32', 'fmaf', -rounded_mean, alpha, bias)
                                values[r, i] = T.cast(x[r, i], 'float32') * alpha
                                offset_values[r, i] = beta
                                alpha_values[r, i] = alpha
                        for r, i in T.Parallel(rows, lanes, loop_layout=stat_layout):
                            if i == 0:
                                offsets[r] = offset_values[r, i]
                        for r, i in T.Parallel(rows, width, loop_layout=row_layout):
                            if i < n:
                                products[r, i] = values[r, i]
                    with T.SimtVF(threads=rows * lanes):
                        output_values = T.alloc_fragment((rows, width), 'float32')
                        for r, i in T.Parallel(rows, width, loop_layout=row_layout):
                            if i < n:
                                output_values[r, i] = products[r, i] + offsets[r]
                        for r, i in T.Parallel(rows, width, loop_layout=row_layout):
                            if i < n:
                                y[r, i] = output_values[r, i]
                    T.copy(y[:rows, :n], Y[first:first + rows, :n])
    return tilelang.compile(main, target='ascend', out_idx=[3], pass_configs={'tl.device_compile_flags': ['-ffp-contract=off']})

@functools.lru_cache(maxsize=128)
def build_lowprecision(batch, channels, spatial, groups, dtype, eps, weighted, biased):
    assert channels == groups and supported(batch, channels, spatial, dtype, weighted, biased)
    m = batch * channels
    n = spatial
    size = 2
    padded = max(16, (n + 15) // 16 * 16)
    rows = 8 if m % 8 == 0 and 32 * padded * size + 32 * padded + 2 * channels * size + 4096 < 192 * 1024 else 1
    count = m // rows
    blocks = min(64, count)
    width = max(32, tilelang.next_power_of_2(n))
    pitch = max(16, (n + 15) // 16 * 16)
    zero_inverse = 1 / eps ** 0.5
    eps_hi = struct.unpack('f', struct.pack('f', eps))[0]
    eps_lo = eps - eps_hi
    raw_n = struct.unpack('I', struct.pack('f', float(n)))[0]
    rounded_n = raw_n + 32767 + (raw_n >> 16 & 1) & 4294901760
    bf_n = struct.unpack('f', struct.pack('I', rounded_n))[0]
    lanes = 32 if rows == 8 else min(256, width)
    row_layout = T.Fragment((rows, width), forward_thread_fn=lambda r, i: r * lanes + i % lanes, forward_index_fn=lambda r, i: i // lanes)
    stat_layout = T.Fragment((rows, lanes), forward_thread_fn=lambda r, i: r * lanes + i, forward_index_fn=lambda r, i: 0)

    @T.prim_func
    def main(X: T.Tensor((m, n), dtype), W: T.Tensor((channels,), dtype), B: T.Tensor((channels,), dtype), Y: T.Tensor((m, n), dtype)):
        with T.Kernel(blocks) as bx:
            x = T.alloc_shared((rows, pitch), dtype)
            y = T.alloc_shared((rows, pitch), dtype)
            w = T.alloc_shared((channels,), dtype)
            b = T.alloc_shared((channels,), dtype)
            products = T.alloc_shared((rows, pitch), 'float32')
            means = T.alloc_shared((rows,), 'float32')
            offsets = T.alloc_shared((rows,), 'float32')
            alphas = T.alloc_shared((rows,), 'float32')
            T.annotate_buffer_versions({x: 2, y: 2})
            if weighted:
                T.copy(W[:channels], w[:channels])
            if biased:
                T.copy(B[:channels], b[:channels])
            for step in T.Pipelined((count + blocks - 1) // blocks, num_stages=2):
                first = (step * blocks + bx) * rows
                if first < m:
                    T.copy(X[first:first + rows, :n], x[:rows, :n])
                    with T.SimtVF(threads=rows * lanes):
                        total = T.alloc_reducer((rows,), 'float32', op='sum')
                        T.reducer_init(total)
                        for r, i in T.Parallel(rows, width, loop_layout=row_layout):
                            if i < n:
                                T.reducer_update(total[r], T.cast(x[r, i], 'float32'))
                        sums = T.alloc_fragment((rows,), 'float32')
                        T.finalize_reducer(total, sums)
                        squares = T.alloc_fragment((rows, width), 'float32')
                        rowmeans = T.alloc_fragment((rows, width), 'float32')
                        for r, i in T.Parallel(rows, width, loop_layout=row_layout):
                            if i < n:
                                mean = sums[r] / n
                                centered = T.cast(x[r, i], 'float32') - mean
                                square = centered * centered
                                squares[r, i] = square
                                rowmeans[r, i] = mean
                        for r, i in T.Parallel(rows, lanes, loop_layout=stat_layout):
                            if i == 0:
                                means[r] = rowmeans[r, i]
                        for r, i in T.Parallel(rows, width, loop_layout=row_layout):
                            if i < n:
                                products[r, i] = squares[r, i]
                    with T.SimtVF(threads=rows * lanes):
                        variance = T.alloc_reducer((rows,), 'float32', op='sum')
                        T.reducer_init(variance)
                        for r, i in T.Parallel(rows, width, loop_layout=row_layout):
                            if i < n:
                                square = products[r, i]
                                T.reducer_update(variance[r], square)
                        vars = T.alloc_fragment((rows,), 'float32')
                        T.finalize_reducer(variance, vars)
                        values = T.alloc_fragment((rows, width), 'float32')
                        offset_values = T.alloc_fragment((rows, width), 'float32')
                        alpha_values = T.alloc_fragment((rows, width), 'float32')
                        for r, i in T.Parallel(rows, width, loop_layout=row_layout):
                            if i < n:
                                if dtype == 'bfloat16':
                                    sum_bf = T.cast(T.cast(vars[r], dtype), 'float32')
                                    var = T.cast(T.cast(sum_bf / T.float32(bf_n), dtype), 'float32')
                                    inverse = T.if_then_else(var == 0, T.float32(zero_inverse), T.rsqrt(var + eps))
                                else:
                                    inverse = T.if_then_else(vars[r] == 0, T.float32(zero_inverse), T.rsqrt(vars[r] / n + eps))
                                rounded_mean = T.cast(T.cast(means[r], dtype), 'float32')
                                rounded_inverse = T.cast(T.cast(inverse, dtype), 'float32')
                                weight = T.cast(w[(first + r) % channels], 'float32') if weighted else T.float32(1)
                                bias = T.cast(b[(first + r) % channels], 'float32') if biased else T.float32(0)
                                alpha = rounded_inverse * weight
                                beta = T.call_extern('float32', 'fmaf', -rounded_mean, alpha, bias)
                                values[r, i] = T.cast(x[r, i], 'float32') * alpha
                                offset_values[r, i] = beta
                                alpha_values[r, i] = alpha
                        for r, i in T.Parallel(rows, lanes, loop_layout=stat_layout):
                            if i == 0:
                                offsets[r] = offset_values[r, i]
                                if n % 16 != 0:
                                    alphas[r] = alpha_values[r, i]
                        for r, i in T.Parallel(rows, width, loop_layout=row_layout):
                            if i < n:
                                products[r, i] = values[r, i]
                    with T.SimtVF(threads=rows * lanes):
                        output_values = T.alloc_fragment((rows, width), 'float32')
                        for r, i in T.Parallel(rows, width, loop_layout=row_layout):
                            if i < n:
                                if n % 16 != 0 and i >= n // 16 * 16:
                                    output_values[r, i] = T.call_extern('float32', 'fmaf', T.cast(x[r, i], 'float32'), alphas[r], offsets[r])
                                else:
                                    output_values[r, i] = products[r, i] + offsets[r]
                        for r, i in T.Parallel(rows, width, loop_layout=row_layout):
                            if i < n:
                                y[r, i] = output_values[r, i]
                    T.copy(y[:rows, :n], Y[first:first + rows, :n])
    return tilelang.compile(main, target='ascend', out_idx=[3], pass_configs={'tl.device_compile_flags': ['-ffp-contract=off']})

def build_batched(batch, channels, spatial, groups, dtype, eps, weighted, biased):
    if dtype == 'float32':
        return build_fp32(batch, channels, spatial, groups, dtype, eps, weighted, biased)
    return build_lowprecision(batch, channels, spatial, groups, dtype, eps, weighted, biased)

def build(batch, channels, spatial, groups, dtype, eps, weighted, biased):
    if supported(batch, channels, spatial, dtype, weighted, biased):
        return build_batched(batch, channels, spatial, groups, dtype, eps, weighted, biased)
    return original.build(batch, channels, spatial, groups, dtype, eps, weighted, biased)
