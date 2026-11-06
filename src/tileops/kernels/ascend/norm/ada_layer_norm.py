"""Ascend adaptive LayerNorm with per-element modulation and optional gate."""

# Registration is optional when the Ascend toolchain is unavailable.
try:
    import tilelang.ascend.language as T
except ImportError:
    pass
else:
    import functools
    from typing import ClassVar
    import torch
    from tileops.backend import register_kernel_type
    from tileops.kernels.kernel_base import Kernel
    from tileops.kernels.norm.call_spec import AdaLayerNormFwdInterface, AdaLayerNormZeroFwdInterface
    import tilelang


    @functools.lru_cache(maxsize=256)
    def _build_fallback(m, n, dtype, eps, gated):
        width = min(4096, tilelang.next_power_of_2(n))
        pitch = max(16, min(width, (n + 15) // 16 * 16))
        blocks = min(64, m)

        @T.prim_func
        def main(X: T.Tensor((m, n), dtype), W: T.Tensor((m, n), dtype), B: T.Tensor((m, n), dtype), G: T.Tensor((m,n), dtype), Y: T.Tensor((m, n), dtype)):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((pitch,), dtype)
                w = T.alloc_shared((pitch,), dtype)
                b = T.alloc_shared((pitch,), dtype)
                g = T.alloc_shared((pitch,), dtype)
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
                            T.copy(W[row,begin:begin+valid], w[:valid])
                            T.copy(B[row,begin:begin+valid], b[:valid])
                            if gated:
                                T.copy(G[row,begin:begin+valid],g[:valid])
                            with T.SimtVF(threads=min(256, width)):
                                for i in T.Parallel(width):
                                    if i < valid:
                                        value = (T.cast(x[i], 'float32') - accum[0]) * accum[1]
                                        modulated = value * T.cast(w[i], 'float32') + T.cast(b[i], 'float32')
                                        if gated:
                                            y[i] = modulated * T.cast(g[i], 'float32')
                                        else:
                                            y[i] = modulated
                            T.copy(y[:valid], Y[row, begin:begin+valid])
        return tilelang.compile(main, target='ascend', out_idx=[4])


    @functools.lru_cache(maxsize=256)
    def build(m, n, dtype, eps, gated):
        from . import _ada_dispatch
        return _ada_dispatch.select_build(_build_fallback, m, n, dtype, eps, gated)


    class _AdaptiveNorm(Kernel):
        devices: ClassVar[frozenset[str]] = frozenset({'npu'})
        @classmethod
        def entry_for(cls, call):
            return call, lambda: cls(call)
        def __init__(self, call):
            super().__init__(); self.call = call
            if call.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise TypeError('Unsupported adaptive normalization dtype')
        def forward(self, x, scale, shift, gate=None):
            c=self.call
            tensors=(x,scale,shift)+((gate,) if self.gated else ())
            if any(t is None or t.shape!=x.shape or t.dtype!=c.dtype for t in tensors):
                raise ValueError('Invalid adaptive normalization signature')
            if c.n<=0:raise ValueError('Normalization width must be positive')
            if not x.numel():return torch.empty_like(x)
            xx=x.reshape(-1,c.n)
            self.compiled=build(xx.shape[0],c.n,str(c.dtype).split('.')[-1],c.eps,self.gated)
            return self.compiled(xx,scale.reshape_as(xx),shift.reshape_as(xx),xx if gate is None else gate.reshape_as(xx)).reshape_as(x)

    class AscendAdaLayerNormKernel(_AdaptiveNorm,AdaLayerNormFwdInterface):
        gated=False
    class AscendAdaLayerNormZeroKernel(_AdaptiveNorm,AdaLayerNormZeroFwdInterface):
        gated=True
    register_kernel_type('AdaLayerNormFwdOp','ascend',AscendAdaLayerNormKernel)
    register_kernel_type('AdaLayerNormZeroFwdOp','ascend',AscendAdaLayerNormZeroKernel)
