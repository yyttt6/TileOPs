"""Ascend FusedAddLayerNorm: normalize the input-dtype rounded residual sum."""

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
    from tileops.kernels.norm.call_spec import FusedAddLayerNormFwdInterface
    import tilelang


    @functools.lru_cache(maxsize=256)
    def build_resident(m,n,dtype,eps,weighted,biased):
        assert 0<n<=8192
        width=tilelang.next_power_of_2(n)
        pitch=max(16,(n+15)//16*16)
        blocks=min(64,m)
        @T.prim_func
        def main(X:T.Tensor((m,n),dtype),R:T.Tensor((m,n),dtype),W:T.Tensor((n,),dtype),B:T.Tensor((n,),dtype),Y:T.Tensor((m,n),dtype),O:T.Tensor((m,n),dtype)):
            with T.Kernel(blocks) as bx:
                x=T.alloc_shared((pitch,),dtype)
                residual=T.alloc_shared((pitch,),dtype)
                w=T.alloc_shared((pitch,),dtype)
                b=T.alloc_shared((pitch,),dtype)
                y=T.alloc_shared((pitch,),dtype)
                if weighted:T.copy(W[:n],w[:n])
                if biased:T.copy(B[:n],b[:n])
                for step in T.serial((m+blocks-1)//blocks):
                    row=step*blocks+bx
                    if row<m:
                        T.copy(X[row,:n],x[:n])
                        T.copy(R[row,:n],residual[:n])
                        with T.SimtVF(threads=256):
                            for i in T.Parallel(width):
                                if i<n:x[i]=T.cast(x[i],'float32')+T.cast(residual[i],'float32')
                        with T.SimtVF(threads=256):
                            total=T.alloc_reducer((1,),'float32',op='sum')
                            T.reducer_init(total)
                            for i in T.Parallel(width):
                                if i<n:T.reducer_update(total[0],T.cast(x[i],'float32'))
                            reduced=T.alloc_fragment((1,),'float32')
                            T.finalize_reducer(total,reduced)
                            mean=reduced[0]/n
                            variance=T.alloc_reducer((1,),'float32',op='sum')
                            T.reducer_init(variance)
                            for i in T.Parallel(width):
                                if i<n:
                                    centered=T.cast(x[i],'float32')-mean
                                    T.reducer_update(variance[0],centered*centered)
                            reduced_var=T.alloc_fragment((1,),'float32')
                            T.finalize_reducer(variance,reduced_var)
                            inverse=T.rsqrt(reduced_var[0]/n+eps)
                            for i in T.Parallel(width):
                                if i<n:
                                    value=(T.cast(x[i],'float32')-mean)*inverse
                                    affine=value*T.cast(w[i],'float32') if weighted else value
                                    y[i]=affine+T.cast(b[i],'float32') if biased else affine
                        T.copy(y[:n],Y[row,:n])
                        T.copy(x[:n],O[row,:n])
        return tilelang.compile(main,target='ascend',out_idx=[4,5])

    @functools.lru_cache(maxsize=256)
    def _build_fallback(m, n, dtype, eps, weighted, biased):
        if 0 < n <= 8192:
            return build_resident(m,n,dtype,eps,weighted,biased)
        width = min(4096, tilelang.next_power_of_2(n))
        pitch = max(16, min(width, (n + 15) // 16 * 16))
        blocks = min(64, m)

        @T.prim_func
        def main(X: T.Tensor((m, n), dtype), R: T.Tensor((m,n),dtype), W: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), Y: T.Tensor((m, n), dtype), O: T.Tensor((m,n),dtype)):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((pitch,), dtype)
                residual = T.alloc_shared((pitch,), dtype)
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
                            T.copy(R[row,begin:begin+valid],residual[:valid])
                            with T.SimtVF(threads=min(256,width)):
                                for i in T.Parallel(width):
                                    if i<valid:
                                        x[i]=T.cast(x[i],'float32')+T.cast(residual[i],'float32')
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
                            T.copy(R[row,begin:begin+valid],residual[:valid])
                            with T.SimtVF(threads=min(256,width)):
                                for i in T.Parallel(width):
                                    if i<valid:
                                        x[i]=T.cast(x[i],'float32')+T.cast(residual[i],'float32')
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
                            T.copy(R[row,begin:begin+valid],residual[:valid])
                            with T.SimtVF(threads=min(256,width)):
                                for i in T.Parallel(width):
                                    if i<valid:
                                        x[i]=T.cast(x[i],'float32')+T.cast(residual[i],'float32')
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
                            T.copy(x[:valid], O[row,begin:begin+valid])
        return tilelang.compile(main, target='ascend', out_idx=[4,5])


    @functools.lru_cache(maxsize=256)
    def build(m, n, dtype, eps, weighted, biased):
        from . import _fused_layer_simd
        return _fused_layer_simd.select_build(_build_fallback, m, n, dtype, eps, weighted, biased)


    class AscendFusedAddLayerNormKernel(Kernel,FusedAddLayerNormFwdInterface):
        devices:ClassVar[frozenset[str]]=frozenset({'npu'})
        @classmethod
        def entry_for(cls,call):return call,lambda:cls(call)
        def __init__(self,call):
            super().__init__();self.call=call
            if call.dtype not in (torch.float16,torch.bfloat16,torch.float32):raise TypeError('Unsupported norm dtype')
        def forward(self,x,residual,weight, bias):
            n=self.call.n
            if n<=0 or x.shape!=residual.shape or weight.numel()!=n:raise ValueError('Invalid fused norm signature')
            if not x.numel():return [torch.empty_like(x),torch.empty_like(x)]
            xx=x.reshape(-1,n)
            self.compiled=build(xx.shape[0],n,str(self.call.dtype).split('.')[-1],self.call.eps, True, True)
            y,r=self.compiled(xx,residual.reshape_as(xx),weight.reshape(n),bias.reshape(n))
            return [y.reshape_as(x),r.reshape_as(x)]

    register_kernel_type('FusedAddLayerNormFwdOp','ascend',AscendFusedAddLayerNormKernel)
