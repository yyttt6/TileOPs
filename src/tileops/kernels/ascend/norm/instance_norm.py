"""Ascend InstanceNorm input-statistics, training-update and inference paths."""

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
    from tileops.kernels.norm.call_spec import InstanceNormFwdInterface,InstanceNormTrainFwdInterface,InstanceNormInferFwdInterface
    from ._instance_untracked import build as _build_input_stats_fallback
    from ._instance_train import build_train as _build_train_fallback
    from ._instance_train_resident import build_update
    import tilelang


    def build_input_stats(batch,channels,spatial,groups,dtype,eps,weighted,biased):
        from ._instance_untracked_batched import build_batched,supported
        if channels==groups and supported(batch,channels,spatial,dtype,weighted,biased):
            return build_batched(batch,channels,spatial,groups,dtype,eps,weighted,biased)
        return _build_input_stats_fallback(batch,channels,spatial,groups,dtype,eps,weighted,biased)

    def build_train(batch,channels,spatial,dtype,eps,momentum,weighted,biased):
        size=4 if dtype=='float32' else 2
        pitch=max(16,(spatial+15)//16*16)
        if 1<spatial<=4096 and 4*pitch*size+2*channels*size+8*channels+320<192*1024:
            from ._instance_train_batched import build_train as resident
            return resident(batch,channels,spatial,dtype,eps,momentum,weighted,biased)
        return _build_train_fallback(batch,channels,spatial,dtype,eps,momentum,weighted,biased)

    @functools.lru_cache(maxsize=256)
    def build_infer(batch, channels, spatial, dtype, eps, weighted, biased):
        groups=channels
        m=batch*groups
        cpg=channels//groups
        n=cpg*spatial
        width = min(4096, tilelang.next_power_of_2(n))
        pitch = max(16, min(width, (n + 15) // 16 * 16))
        blocks = min(64, m)

        @T.prim_func
        def main(X: T.Tensor((m, n), dtype), W: T.Tensor((channels,), dtype), B: T.Tensor((channels,), dtype), RM: T.Tensor((channels,), 'float32'), RV: T.Tensor((channels,), 'float32'), Y: T.Tensor((m, n), dtype)):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((pitch,), dtype)
                w = T.alloc_shared((16,), dtype)
                b = T.alloc_shared((16,), dtype)
                y = T.alloc_shared((pitch,), dtype)
                accum = T.alloc_shared((16,), 'float32')
                for step in T.serial((m + blocks - 1) // blocks):
                    row = step * blocks + bx
                    if row < m:
                        T.copy(RM[row % channels:row % channels+1],accum[:1])
                        T.copy(RV[row % channels:row % channels+1],accum[8:9])
                        with T.SimtVF(threads=32):
                            accum[0]=T.cast(T.cast(accum[0],dtype),'float32')
                            accum[8]=T.rsqrt(T.cast(T.cast(accum[8],dtype),'float32')+eps)
                        for channel in T.serial(cpg):
                            for chunk in T.serial((spatial + width - 1) // width):
                                begin = channel * spatial + chunk * width
                                valid = T.min(width, spatial - chunk * width)
                                T.copy(X[row, begin:begin+valid], x[:valid])
                                if weighted:
                                    T.copy(W[(row % groups)*cpg+channel:(row % groups)*cpg+channel+1], w[:1])
                                if biased:
                                    T.copy(B[(row % groups)*cpg+channel:(row % groups)*cpg+channel+1], b[:1])
                                with T.SimtVF(threads=min(256, width)):
                                    for i in T.Parallel(width):
                                        if i < valid:
                                            value = (T.cast(x[i], 'float32') - accum[0]) * accum[8]
                                            if weighted:
                                                if biased:
                                                    y[i] = value * T.cast(w[0], 'float32') + T.cast(b[0], 'float32')
                                                else:
                                                    y[i] = value * T.cast(w[0], 'float32')
                                            else:
                                                if biased:
                                                    y[i] = value + T.cast(b[0], 'float32')
                                                else:
                                                    y[i] = value
                                T.copy(y[:valid], Y[row, begin:begin+valid])
        return tilelang.compile(main, target='ascend', out_idx=[5])


    class _InstanceBase(Kernel):
        devices:ClassVar[frozenset[str]]=frozenset({'npu'})
        @classmethod
        def entry_for(cls,call):return call,lambda:cls(call)
        def __init__(self,call):super().__init__();self.call=call
        def affine(self,x,weight,bias):
            unused=x.reshape(-1)[:self.call.c]
            return unused if weight is None else weight,unused if bias is None else bias

    class AscendInstanceNormKernel(_InstanceBase,InstanceNormFwdInterface):
        def forward(self,x,running_mean=None,running_var=None,weight=None,bias=None):
            c=self.call
            if c.spatial<=1:raise ValueError('Input statistics require more than one spatial value')
            w,b=self.affine(x,weight,bias)
            self.compiled=build_input_stats(c.n,c.c,c.spatial,c.c,str(c.dtype).split('.')[-1],c.eps,weight is not None,bias is not None)
            return self.compiled(x.reshape(c.n*c.c,c.spatial),w,b).reshape_as(x)

    class AscendInstanceNormTrainKernel(_InstanceBase,InstanceNormTrainFwdInterface):
        def forward(self,x,running_mean,running_var,weight=None,bias=None):
            c=self.call
            if c.spatial<=1:raise ValueError('Input statistics require more than one spatial value')
            w,b=self.affine(x,weight,bias);dtype=str(c.dtype).split('.')[-1]
            self.compiled=build_train(c.n,c.c,c.spatial,dtype,c.eps,c.momentum,weight is not None,bias is not None)
            y,updates=self.compiled(x.reshape(c.n*c.c,c.spatial),w,b,running_mean,running_var)
            self.update=build_update(c.n,c.c,dtype)
            self.update(updates,running_mean,running_var)
            return y.reshape_as(x)

    class AscendInstanceNormInferKernel(_InstanceBase,InstanceNormInferFwdInterface):
        def forward(self,x,running_mean,running_var,weight=None,bias=None):
            c=self.call;w,b=self.affine(x,weight,bias)
            self.compiled=build_infer(c.n,c.c,c.spatial,str(c.dtype).split('.')[-1],c.eps,weight is not None,bias is not None)
            return self.compiled(x.reshape(c.n*c.c,c.spatial),w,b,running_mean,running_var).reshape_as(x)

    register_kernel_type('InstanceNormFwdOp','ascend_input_stats',AscendInstanceNormKernel)
    register_kernel_type('InstanceNormFwdOp','ascend_train',AscendInstanceNormTrainKernel)
    register_kernel_type('InstanceNormFwdOp','ascend_infer',AscendInstanceNormInferKernel)
