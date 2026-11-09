"""Flattened BatchNorm inference, preserving original centered affine operation order."""
import functools
import tilelang
import tilelang.ascend.language as T
@functools.lru_cache(maxsize=128)
def build(batch,channels,spatial,dtype,eps,momentum,training,tracked,weighted,biased):
    assert not training and tracked
    total=batch*channels*spatial;width=min(8192,tilelang.next_power_of_2(total));blocks=min(64,(total+width-1)//width)
    @T.prim_func
    def main(X:T.Tensor((batch,channels,spatial),dtype),RM:T.Tensor((channels,),'float32'),RV:T.Tensor((channels,),'float32'),W:T.Tensor((channels,),'float32'),B:T.Tensor((channels,),'float32'),Y:T.Tensor((batch,channels,spatial),dtype),Mean:T.Tensor((channels,),'float32'),Rstd:T.Tensor((channels,),'float32')):
        with T.Kernel(blocks) as bx:
            flat_x=T.view(X,shape=(total,),dtype=dtype);flat_y=T.view(Y,shape=(total,),dtype=dtype)
            x=T.alloc_shared((width,),dtype);y=T.alloc_shared((width,),dtype)
            mean=T.alloc_shared((channels,),'float32');var=T.alloc_shared((channels,),'float32');inv=T.alloc_shared((channels,),'float32')
            w=T.alloc_shared((channels,),'float32');bias=T.alloc_shared((channels,),'float32')
            T.annotate_buffer_versions({x:2,y:2})
            T.copy(RM[:channels],mean[:channels]);T.copy(RV[:channels],var[:channels])
            if weighted:T.copy(W[:channels],w[:channels])
            if biased:T.copy(B[:channels],bias[:channels])
            with T.SimtVF(threads=256):
                for c in T.Parallel(channels):inv[c]=T.rsqrt(var[c]+eps)
            if bx==0:
                T.copy(mean[:channels],Mean[:channels]);T.copy(inv[:channels],Rstd[:channels])
            for step in T.Pipelined((total+blocks*width-1)//(blocks*width),num_stages=2):
                start=(step*blocks+bx)*width;valid=T.min(width,total-start)
                if start<total:
                    T.copy(flat_x[start:start+valid],x[:valid])
                    with T.SimtVF(threads=256):
                        values=T.alloc_fragment((width,),'float32')
                        for i in T.Parallel(width):
                            if i<valid:
                                c=((start+i)//spatial)%channels
                                value=(T.cast(x[i],'float32')-mean[c])*inv[c]
                                if weighted:value=value*w[c]
                                if biased:value=value+bias[c]
                                values[i]=value
                        for i in T.Parallel(width):
                            if i<valid:y[i]=values[i]
                    T.copy(y[:valid],flat_y[start:start+valid])
    return tilelang.compile(main,target='ascend',out_idx=[5,6,7])
