"""Ascend LayerNorm with bounded UB storage and centered FP32 variance."""
import functools
from typing import ClassVar
import torch
from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm.call_spec import LayerNormFwdInterface
import tilelang
import tilelang.ascend.language as T


@functools.lru_cache(maxsize=256)
def _build_fallback(m, n, dtype, eps, weighted, biased):
    width = min(4096, tilelang.next_power_of_2(n))
    pitch = max(16, min(width, (n + 15) // 16 * 16))
    blocks = min(64, m)

    @T.prim_func
    def main(X: T.Tensor((m, n), dtype), W: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), Y: T.Tensor((m, n), dtype)):
        with T.Kernel(blocks) as bx:
            x = T.alloc_shared((pitch,), dtype)
            w = T.alloc_shared((pitch,), dtype)
            b = T.alloc_shared((pitch,), dtype)
            y = T.alloc_shared((pitch,), dtype)
            accum = T.alloc_shared((16,), 'float32')
            for step in T.serial((m + blocks - 1) // blocks):
                row = step * blocks + bx
                if row < m:
                    T.clear(accum)
                    for chunk in T.serial((n + width - 1) // width):
                        begin = chunk * width
                        valid = T.min(width, n - begin)
                        T.copy(X[row, begin:begin+valid], x[:valid])
                        with T.SimtVF(threads=min(256, width)):
                            total = T.alloc_reducer((1,), 'float32', op='sum')
                            T.reducer_init(total)
                            for i in T.Parallel(width):
                                if i < valid:
                                    v = T.cast(x[i], 'float32')
                                    T.reducer_update(total[0], v)
                            reduced = T.alloc_fragment((1,), 'float32')
                            T.finalize_reducer(total, reduced)
                            accum[0] = accum[0] + reduced[0]
                    with T.SimtVF(threads=32):
                        accum[0] = accum[0] / n
                    for chunk in T.serial((n + width - 1) // width):
                        begin = chunk * width
                        valid = T.min(width, n - begin)
                        T.copy(X[row, begin:begin+valid], x[:valid])
                        with T.SimtVF(threads=min(256, width)):
                            variance = T.alloc_reducer((1,), 'float32', op='sum')
                            T.reducer_init(variance)
                            for i in T.Parallel(width):
                                if i < valid:
                                    centered = T.cast(x[i], 'float32') - accum[0]
                                    T.reducer_update(variance[0], centered * centered)
                            reduced_var = T.alloc_fragment((1,), 'float32')
                            T.finalize_reducer(variance, reduced_var)
                            accum[1] = accum[1] + reduced_var[0]
                    with T.SimtVF(threads=32):
                        accum[1] = T.rsqrt(accum[1] / n + eps)
                    for chunk in T.serial((n + width - 1) // width):
                        begin = chunk * width
                        valid = T.min(width, n - begin)
                        T.copy(X[row, begin:begin+valid], x[:valid])
                        if weighted:
                            T.copy(W[begin:begin+valid], w[:valid])
                        if biased:
                            T.copy(B[begin:begin+valid], b[:valid])
                        with T.SimtVF(threads=min(256, width)):
                            for i in T.Parallel(width):
                                if i < valid:
                                    value = (T.cast(x[i], 'float32') - accum[0]) * accum[1]
                                    if weighted:
                                        if biased:
                                            y[i] = value * T.cast(w[i], 'float32') + T.cast(b[i], 'float32')
                                        else:
                                            y[i] = value * T.cast(w[i], 'float32')
                                    else:
                                        if biased:
                                            y[i] = value + T.cast(b[i], 'float32')
                                        else:
                                            y[i] = value
                        T.copy(y[:valid], Y[row, begin:begin+valid])
    return tilelang.compile(main, target='ascend', out_idx=[3])


@functools.lru_cache(maxsize=256)
def build(m, n, dtype, eps, weighted, biased):
    from tileops.kernels.ascend.norm import _layer_batched as candidate
    size = 4 if dtype == 'float32' else 2
    if n > 0 and n % 128 == 0 and dtype in ('float16', 'bfloat16', 'float32') and 6 * n * size < 208 * 1024:
        if dtype == 'float32' and m % 2 == 0 and n % 256 == 0 and 10 * n * size < 208 * 1024:
            return candidate.build(m, n, dtype, eps, weighted, biased, 2, 256)
        return candidate.original.build(m, n, dtype, eps, weighted, biased, 1, 256)
    return _build_fallback(m, n, dtype, eps, weighted, biased)


class AscendLayerNormKernel(Kernel, LayerNormFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({'npu'})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError('Unsupported LayerNorm dtype')
        if call.n <= 0:
            raise ValueError('LayerNorm width must be positive')

    def forward(self, x, weight=None, bias=None):
        n = self.call.n
        if x.numel() == 0:
            return torch.empty_like(x)
        if x.numel() % n or x.dtype != self.call.dtype:
            raise ValueError('Invalid LayerNorm input signature')
        if weight is not None and (weight.numel() != n or weight.dtype != x.dtype):
            raise ValueError('Invalid LayerNorm weight signature')
        if bias is not None and (bias.numel() != n or bias.dtype != x.dtype):
            raise ValueError('Invalid LayerNorm bias signature')
        rows = x.reshape(-1, n)
        # An unused input view avoids an allocation for the unweighted signature.
        w = rows[0] if weight is None else weight.reshape(n)
        kernel = build(rows.shape[0], n, str(x.dtype).split('.')[-1], self.call.eps, weight is not None, bias is not None)
        b = rows[0] if bias is None else bias.reshape(n)
        return kernel(rows, w, b).reshape(x.shape)


register_kernel_type('LayerNormFwdOp', 'ascend', AscendLayerNormKernel)
