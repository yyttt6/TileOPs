"""Retain one complete training channel across statistics and output."""
import functools
import tilelang
import tilelang.ascend.language as T
@functools.lru_cache(maxsize=128)
def build(batch,channels,spatial,dtype,eps,momentum,training,tracked,weighted,biased):
    length=batch*spatial
    assert training and 1<length<=8192 and (spatial==1 or spatial%16==0)
    width=tilelang.next_power_of_2(length);pitch=max(16,(length+15)//16*16)
    blocks=min(64,channels);unbiased=length/(length-1)
    @T.prim_func
    def main(X:T.Tensor((batch,channels,spatial),dtype),RM:T.Tensor((channels,),'float32'),RV:T.Tensor((channels,),'float32'),W:T.Tensor((channels,),'float32'),B:T.Tensor((channels,),'float32'),Y:T.Tensor((batch,channels,spatial),dtype),Mean:T.Tensor((channels,),'float32'),Rstd:T.Tensor((channels,),'float32')):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((pitch,),dtype);y=T.alloc_shared((pitch,),dtype)
            w=T.alloc_shared((channels,),'float32');b=T.alloc_shared((channels,),'float32')
            old_m=T.alloc_shared((channels,),'float32');old_v=T.alloc_shared((channels,),'float32')
            stats=T.alloc_shared((16,),'float32')
            if weighted:T.copy(W[:channels],w[:channels])
            if biased:T.copy(B[:channels],b[:channels])
            if tracked:
                T.copy(RM[:channels],old_m[:channels]);T.copy(RV[:channels],old_v[:channels])
            for step in T.serial((channels+blocks-1)//blocks):
                c=step*blocks+bx
                if c<channels:
                    if spatial==1:
                        with T.SimtVF(threads=min(256,width)):
                            for nb in T.Parallel(batch):x[nb]=X[nb,c,0]
                    else:
                        for nb in T.serial(batch):T.copy(X[nb,c,:spatial],x[nb*spatial:(nb+1)*spatial])
                    with T.SimtVF(threads=min(256,width)):
                        total=T.alloc_reducer((1,),'float32',op='sum');T.reducer_init(total)
                        for i in T.Parallel(width):
                            if i<length:T.reducer_update(total[0],T.cast(x[i],'float32'))
                        sums=T.alloc_fragment((1,),'float32');T.finalize_reducer(total,sums)
                        mean=sums[0]/length
                        varsum=T.alloc_reducer((1,),'float32',op='sum');T.reducer_init(varsum)
                        for i in T.Parallel(width):
                            if i<length:
                                delta=T.cast(x[i],'float32')-mean
                                T.reducer_update(varsum[0],delta*delta)
                        variances=T.alloc_fragment((1,),'float32');T.finalize_reducer(varsum,variances)
                        var=variances[0]/length;inverse=T.rsqrt(var+eps)
                        for i in T.Parallel(1):
                            Mean[c]=mean;Rstd[c]=inverse
                            if tracked:
                                RM[c]=(1-momentum)*old_m[c]+momentum*mean
                                RV[c]=(1-momentum)*old_v[c]+momentum*var*unbiased
                        values=T.alloc_fragment((width,),'float32')
                        for i in T.Parallel(width):
                            if i<length:
                                value=(T.cast(x[i],'float32')-mean)*inverse
                                if weighted:value=value*w[c]
                                if biased:value=value+b[c]
                                values[i]=value
                        for i in T.Parallel(width):
                            if i<length:y[i]=values[i]
                    if spatial==1:
                        with T.SimtVF(threads=min(256,width)):
                            for nb in T.Parallel(batch):Y[nb,c,0]=y[nb]
                    else:
                        for nb in T.serial(batch):T.copy(y[nb*spatial:(nb+1)*spatial],Y[nb,c,:spatial])
    return tilelang.compile(main,target='ascend',out_idx=[5,6,7])
