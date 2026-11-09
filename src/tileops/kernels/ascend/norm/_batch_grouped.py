"""BatchNorm small training rows: four adjacent channels per vector function."""
import functools
import tilelang
import tilelang.ascend.language as T
from . import _batch_resident as base
@functools.lru_cache(maxsize=128)
def build_grouped(batch,channels,spatial,dtype,eps,momentum,training,tracked,weighted,biased):
    rows=4;length=batch*spatial;width=tilelang.next_power_of_2(length)
    groups=channels//rows;blocks=min(64,groups);unbiased=length/(length-1)
    @T.prim_func
    def main(X:T.Tensor((batch,channels,spatial),dtype),RM:T.Tensor((channels,),'float32'),RV:T.Tensor((channels,),'float32'),W:T.Tensor((channels,),'float32'),B:T.Tensor((channels,),'float32'),Y:T.Tensor((batch,channels,spatial),dtype),Mean:T.Tensor((channels,),'float32'),Rstd:T.Tensor((channels,),'float32')):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((batch,rows,spatial),dtype);y=T.alloc_shared((batch,rows,spatial),dtype)
            w=T.alloc_shared((channels,),'float32');b=T.alloc_shared((channels,),'float32')
            means=T.alloc_shared((rows,),'float32');inverses=T.alloc_shared((rows,),'float32')
            old_m=T.alloc_shared((channels,),'float32');old_v=T.alloc_shared((channels,),'float32')
            if weighted:T.copy(W[:channels],w[:channels])
            if biased:T.copy(B[:channels],b[:channels])
            if tracked:
                T.copy(RM[:channels],old_m[:channels]);T.copy(RV[:channels],old_v[:channels])
            for step in T.serial((groups+blocks-1)//blocks):
                first=(step*blocks+bx)*rows
                if first<channels:
                    if spatial==1:
                        with T.SimtVF(threads=128):
                            for nb,r in T.Parallel(batch,rows):x[nb,r,0]=X[nb,first+r,0]
                    else:
                        for nb in T.serial(batch):T.copy(X[nb,first:first+rows,:spatial],x[nb,:rows,:spatial])
                    with T.SimtVF(threads=256):
                        sums=T.alloc_reducer((rows,),'float32',op='sum');T.reducer_init(sums)
                        for r,i in T.Parallel(rows,width):
                            if i<length:T.reducer_update(sums[r],T.cast(x[i//spatial,r,i%spatial],'float32'))
                        totals=T.alloc_fragment((rows,),'float32');T.finalize_reducer(sums,totals)
                        tx=T.get_thread_binding()
                        if tx==0:
                            for r in T.serial(rows):means[r]=totals[r]/length
                    with T.SimtVF(threads=256):
                        varsums=T.alloc_reducer((rows,),'float32',op='sum');T.reducer_init(varsums)
                        for r,i in T.Parallel(rows,width):
                            if i<length:
                                delta=T.cast(x[i//spatial,r,i%spatial],'float32')-means[r]
                                T.reducer_update(varsums[r],delta*delta)
                        variances=T.alloc_fragment((rows,),'float32');T.finalize_reducer(varsums,variances)
                        tx=T.get_thread_binding()
                        if tx==0:
                            for r in T.serial(rows):
                                var=variances[r]/length
                                inverses[r]=T.rsqrt(var+eps)
                                Mean[first+r]=means[r];Rstd[first+r]=inverses[r]
                                if tracked:
                                    RM[first+r]=(1-momentum)*old_m[first+r]+momentum*means[r]
                                    RV[first+r]=(1-momentum)*old_v[first+r]+momentum*var*unbiased
                    with T.SimtVF(threads=256):
                        values=T.alloc_fragment((rows,width),'float32')
                        for r,i in T.Parallel(rows,width):
                            if i<length:
                                value=(T.cast(x[i//spatial,r,i%spatial],'float32')-means[r])*inverses[r]
                                if weighted:value=value*w[first+r]
                                if biased:value=value+b[first+r]
                                values[r,i]=value
                        for r,i in T.Parallel(rows,width):
                            if i<length:y[i//spatial,r,i%spatial]=values[r,i]
                    if spatial==1:
                        with T.SimtVF(threads=128):
                            for nb,r in T.Parallel(batch,rows):Y[nb,first+r,0]=y[nb,r,0]
                    else:
                        for nb in T.serial(batch):T.copy(y[nb,:rows,:spatial],Y[nb,first:first+rows,:spatial])
    return tilelang.compile(main,target='ascend',out_idx=[5,6,7])
def build(batch,channels,spatial,dtype,eps,momentum,training,tracked,weighted,biased):
    if channels%4==0 and spatial==1 and 1<batch*spatial<=2048:
        return build_grouped(batch,channels,spatial,dtype,eps,momentum,training,tracked,weighted,biased)
    return base.build(batch,channels,spatial,dtype,eps,momentum,training,tracked,weighted,biased)
