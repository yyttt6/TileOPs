"""Resident low-precision GroupNorm with whole-group affine output."""
import functools,math
import tilelang
import tilelang.ascend.language as T
@functools.lru_cache(maxsize=128)
def build(batch,channels,spatial,groups,dtype,eps,weighted,biased):
    cpg=channels//groups;m=batch*groups;n=cpg*spatial
    assert dtype in ('float16','bfloat16') and 0<n<=32768
    width=tilelang.next_power_of_2(n);pitch=max(16,(n+15)//16*16);blocks=min(64,m)
    assert 4*pitch*2+2*channels*2<208*1024
    zero_inverse=1.0/math.sqrt(eps)
    @T.prim_func
    def main(X:T.Tensor((m,n),dtype),W:T.Tensor((channels,),dtype),B:T.Tensor((channels,),dtype),Y:T.Tensor((m,n),dtype)):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((pitch,),dtype);y=T.alloc_shared((pitch,),dtype)
            w=T.alloc_shared((channels,),dtype);b=T.alloc_shared((channels,),dtype)
            T.annotate_buffer_versions({x:2,y:2})
            if weighted:T.copy(W[:channels],w[:channels])
            if biased:T.copy(B[:channels],b[:channels])
            for step in T.Pipelined((m+blocks-1)//blocks,num_stages=2):
                row=step*blocks+bx
                if row<m:
                    T.copy(X[row,:n],x[:n])
                    with T.SimtVF(threads=256):
                        total=T.alloc_reducer((1,),'float32',op='sum');T.reducer_init(total)
                        for i in T.Parallel(width):
                            if i<n:T.reducer_update(total[0],T.cast(x[i],'float32'))
                        reduced=T.alloc_fragment((1,),'float32');T.finalize_reducer(total,reduced)
                        mean=reduced[0]/n
                        variance=T.alloc_reducer((1,),'float32',op='sum');T.reducer_init(variance)
                        for i in T.Parallel(width):
                            if i<n:
                                centered=T.cast(x[i],'float32')-mean
                                T.reducer_update(variance[0],centered*centered)
                        reduced_var=T.alloc_fragment((1,),'float32');T.finalize_reducer(variance,reduced_var)
                        inverse=T.if_then_else(reduced_var[0]==0,T.float32(zero_inverse),T.rsqrt(reduced_var[0]/n+eps))
                        values=T.alloc_fragment((width,),'float32')
                        for i in T.Parallel(width):
                            if i<n:
                                if weighted or biased:
                                    channel=(row%groups)*cpg+i//spatial
                                    scale=inverse
                                    if weighted:scale=scale*T.cast(w[channel],'float32')
                                    offset=-scale*mean
                                    if biased:offset=offset+T.cast(b[channel],'float32')
                                    values[i]=T.call_extern('float32','fmaf',scale,T.cast(x[i],'float32'),offset)
                                else:values[i]=(T.cast(x[i],'float32')-mean)*inverse
                        for i in T.Parallel(width):
                            if i<n:y[i]=values[i]
                    T.copy(y[:n],Y[row,:n])
    return tilelang.compile(main,target='ascend',out_idx=[3])
