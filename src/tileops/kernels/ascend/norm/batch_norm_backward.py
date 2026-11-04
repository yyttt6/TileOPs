"""Ascend BatchNorm backward with FP32 channel-gradient accumulation."""

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
    from tileops.kernels.norm.call_spec import BatchNormBwdInterface

    @functools.lru_cache(maxsize=128)
    def build(batch,channels,spatial,dtype):
        from . import _batch_bwd_optimized as optimized
        if optimized.route(batch,channels,spatial,dtype) is not None:
            return optimized.build(batch,channels,spatial,dtype)
        width=min(1024,tilelang.next_power_of_2(spatial));pitch=max(16,width)
        blocks=min(64,channels);length=batch*spatial
        @T.prim_func
        def main(D:T.Tensor((batch,channels,spatial),dtype),X:T.Tensor((batch,channels,spatial),dtype),W:T.Tensor((channels,),'float32'),M:T.Tensor((channels,),'float32'),R:T.Tensor((channels,),'float32'),DX:T.Tensor((batch,channels,spatial),dtype),DW:T.Tensor((channels,),'float32'),DB:T.Tensor((channels,),'float32')):
            with T.Kernel(blocks) as bx:
                x=T.alloc_shared((pitch,),dtype);dy=T.alloc_shared((pitch,),dtype);dx=T.alloc_shared((pitch,),dtype)
                w=T.alloc_shared((16,),'float32');mean=T.alloc_shared((16,),'float32');rstd=T.alloc_shared((16,),'float32')
                sums=T.alloc_shared((16,),'float32')
                for step in T.serial((channels+blocks-1)//blocks):
                    c=step*blocks+bx
                    if c<channels:
                        T.copy(W[c:c+1],w[:1]);T.copy(M[c:c+1],mean[:1]);T.copy(R[c:c+1],rstd[:1]);T.clear(sums)
                        for n in T.serial(batch):
                            for chunk in T.serial((spatial+width-1)//width):
                                start=chunk*width;valid=T.min(width,spatial-start)
                                T.copy(X[n,c,start:start+valid],x[:valid]);T.copy(D[n,c,start:start+valid],dy[:valid])
                                with T.SimtVF(threads=min(256,width)):
                                    db=T.alloc_reducer((1,),'float32',op='sum');dw=T.alloc_reducer((1,),'float32',op='sum')
                                    T.reducer_init(db);T.reducer_init(dw)
                                    for i in T.Parallel(width):
                                        if i<valid:
                                            value=T.cast(dy[i],'float32')
                                            xhat=(T.cast(x[i],'float32')-mean[0])*rstd[0]
                                            T.reducer_update(db[0],value);T.reducer_update(dw[0],value*xhat)
                                    dbv=T.alloc_fragment((1,),'float32');dwv=T.alloc_fragment((1,),'float32')
                                    T.finalize_reducer(db,dbv);T.finalize_reducer(dw,dwv)
                                    sums[0]=sums[0]+dbv[0];sums[8]=sums[8]+dwv[0]
                        T.copy(sums[:1],DB[c:c+1]);T.copy(sums[8:9],DW[c:c+1])
                        for n in T.serial(batch):
                            for chunk in T.serial((spatial+width-1)//width):
                                start=chunk*width;valid=T.min(width,spatial-start)
                                T.copy(X[n,c,start:start+valid],x[:valid]);T.copy(D[n,c,start:start+valid],dy[:valid])
                                with T.SimtVF(threads=min(256,width)):
                                    for i in T.Parallel(width):
                                        if i<valid:
                                            xhat=(T.cast(x[i],'float32')-mean[0])*rstd[0]
                                            dx[i]=(w[0]*rstd[0])/length*(length*T.cast(dy[i],'float32')-sums[0]-xhat*sums[8])
                                T.copy(dx[:valid],DX[n,c,start:start+valid])
        return tilelang.compile(main,target='ascend',out_idx=[5,6,7])

    class AscendBatchNormBwdKernel(Kernel,BatchNormBwdInterface):
        devices:ClassVar[frozenset[str]]=frozenset({'npu'})
        @classmethod
        def entry_for(cls,call):return call,lambda:cls(call)
        def __init__(self,call):super().__init__();self.call=call
        def forward(self,grad_out,x,weight,mean,rstd):
            c=self.call
            self.compiled=build(c.n,c.c,c.spatial,str(c.dtype).split('.')[-1])
            return tuple(self.compiled(grad_out,x,weight,mean,rstd))
    register_kernel_type('BatchNormBwdOp','ascend',AscendBatchNormBwdKernel)
