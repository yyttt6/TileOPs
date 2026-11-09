"""Ascend BatchNorm forward with FP32 batch and running statistics."""

# Registration is optional when the Ascend toolchain is unavailable.
try:
    import tilelang.ascend.language as T
except ImportError:
    pass
else:
    import functools
    from typing import ClassVar
    import torch
    import tilelang
    from tileops.backend import register_kernel_type
    from tileops.kernels.kernel_base import Kernel
    from tileops.kernels.norm.call_spec import BatchNormTrainFwdInterface,BatchNormInferFwdInterface

    @functools.lru_cache(maxsize=128)
    def build(batch,channels,spatial,dtype,eps,momentum,training,tracked,weighted,biased):
        total=batch*channels*spatial
        if not training and tracked and total>0 and total*(32 if dtype=='float32' else 16)<(1<<31) and 20*channels+4*min(8192,1<<(total-1).bit_length())*(4 if dtype=='float32' else 2)<208*1024:
            from ._batch_infer_flat import build as build_infer
            return build_infer(batch,channels,spatial,dtype,eps,momentum,training,tracked,weighted,biased)
        if training and 1<batch*spatial<=8192 and (spatial==1 or spatial%16==0):
            from ._batch_simd_grouped import build as build_training
            return build_training(batch,channels,spatial,dtype,eps,momentum,training,tracked,weighted,biased)
        width=min(8192,tilelang.next_power_of_2(spatial));pitch=max(16,width)
        length=batch*spatial;blocks=min(64,channels)
        unbiased=length/(length-1) if length>1 else 0.0
        @T.prim_func
        def main(X:T.Tensor((batch,channels,spatial),dtype),RM:T.Tensor((channels,),'float32'),RV:T.Tensor((channels,),'float32'),W:T.Tensor((channels,),'float32'),B:T.Tensor((channels,),'float32'),Y:T.Tensor((batch,channels,spatial),dtype),Mean:T.Tensor((channels,),'float32'),Rstd:T.Tensor((channels,),'float32')):
            with T.Kernel(blocks) as bx:
                x=T.alloc_shared((pitch,),dtype);y=T.alloc_shared((pitch,),dtype)
                stats=T.alloc_shared((16,),'float32');w=T.alloc_shared((16,),'float32');b=T.alloc_shared((16,),'float32')
                old_m=T.alloc_shared((16,),'float32');old_v=T.alloc_shared((16,),'float32')
                for step in T.serial((channels+blocks-1)//blocks):
                    c=step*blocks+bx
                    if c<channels:
                        if weighted:T.copy(W[c:c+1],w[:1])
                        if biased:T.copy(B[c:c+1],b[:1])
                        if training:
                            T.clear(stats)
                            for n in T.serial(batch):
                                for chunk in T.serial((spatial+width-1)//width):
                                    start=chunk*width;valid=T.min(width,spatial-start)
                                    T.copy(X[n,c,start:start+valid],x[:valid])
                                    with T.SimtVF(threads=min(256,width)):
                                        total=T.alloc_reducer((1,),'float32',op='sum');T.reducer_init(total)
                                        for i in T.Parallel(width):
                                            if i<valid:T.reducer_update(total[0],T.cast(x[i],'float32'))
                                        result=T.alloc_fragment((1,),'float32');T.finalize_reducer(total,result)
                                        stats[0]=stats[0]+result[0]
                            with T.SimtVF(threads=32):stats[0]=stats[0]/length
                            for n in T.serial(batch):
                                for chunk in T.serial((spatial+width-1)//width):
                                    start=chunk*width;valid=T.min(width,spatial-start)
                                    T.copy(X[n,c,start:start+valid],x[:valid])
                                    with T.SimtVF(threads=min(256,width)):
                                        total=T.alloc_reducer((1,),'float32',op='sum');T.reducer_init(total)
                                        for i in T.Parallel(width):
                                            if i<valid:
                                                delta=T.cast(x[i],'float32')-stats[0]
                                                T.reducer_update(total[0],delta*delta)
                                        result=T.alloc_fragment((1,),'float32');T.finalize_reducer(total,result)
                                        stats[8]=stats[8]+result[0]
                            with T.SimtVF(threads=32):stats[8]=stats[8]/length
                            if tracked:
                                T.copy(RM[c:c+1],old_m[:1]);T.copy(RV[c:c+1],old_v[:1])
                                with T.SimtVF(threads=32):
                                    old_m[0]=(1-momentum)*old_m[0]+momentum*stats[0]
                                    old_v[0]=(1-momentum)*old_v[0]+momentum*stats[8]*unbiased
                                T.copy(old_m[:1],RM[c:c+1]);T.copy(old_v[:1],RV[c:c+1])
                        else:
                            T.copy(RM[c:c+1],stats[:1]);T.copy(RV[c:c+1],stats[8:9])
                        with T.SimtVF(threads=32):stats[8]=T.rsqrt(stats[8]+eps)
                        T.copy(stats[:1],Mean[c:c+1]);T.copy(stats[8:9],Rstd[c:c+1])
                        for n in T.serial(batch):
                            for chunk in T.serial((spatial+width-1)//width):
                                start=chunk*width;valid=T.min(width,spatial-start)
                                T.copy(X[n,c,start:start+valid],x[:valid])
                                with T.SimtVF(threads=min(256,width)):
                                    for i in T.Parallel(width):
                                        if i<valid:
                                            value=(T.cast(x[i],'float32')-stats[0])*stats[8]
                                            if weighted:value=value*w[0]
                                            if biased:value=value+b[0]
                                            y[i]=value
                                T.copy(y[:valid],Y[n,c,start:start+valid])
        return tilelang.compile(main,target='ascend',out_idx=[5,6,7])

    class _BatchForward(Kernel):
        devices:ClassVar[frozenset[str]]=frozenset({'npu'})
        @classmethod
        def entry_for(cls,call):return call,lambda:cls(call)
        def __init__(self,call):super().__init__();self.call=call
        def forward(self,x,running_mean,running_var,weight,bias):
            c=self.call
            if self.training and c.n*c.spatial<=1:raise ValueError('Training requires more than one value per channel')
            # The placeholders are never read in disabled branches.
            missing=any(t is None for t in (running_mean,running_var,weight,bias))
            dummy=torch.empty((c.c,),device=x.device,dtype=torch.float32) if missing else None
            tensors=tuple(dummy if t is None else t for t in (running_mean,running_var,weight,bias))
            self.compiled=build(c.n,c.c,c.spatial,str(c.dtype).split('.')[-1],c.eps,c.momentum,self.training,running_mean is not None,weight is not None,bias is not None)
            result=self.compiled(x,*tensors)
            return tuple(result) if self.training else result[0]
    class AscendBatchNormTrainKernel(_BatchForward,BatchNormTrainFwdInterface):
        training=True
    class AscendBatchNormInferKernel(_BatchForward,BatchNormInferFwdInterface):
        training=False
    register_kernel_type('BatchNormFwdOp','ascend_train',AscendBatchNormTrainKernel)
    register_kernel_type('BatchNormFwdOp','ascend_infer',AscendBatchNormInferKernel)
