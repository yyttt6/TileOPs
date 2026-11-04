"""Four resident InstanceNorm samples; retain every original summation/update order."""
import functools
import tilelang
import tilelang.ascend.language as T
from . import _instance_train_resident as original
train=original
build_update=original.build_update
@functools.lru_cache(maxsize=128)
def build_batched(batch,channels,n,dtype,eps,momentum,weighted,biased):
    rows=4;m=batch*channels;groups=m//rows;blocks=min(64,groups)
    width=tilelang.next_power_of_2(n);pitch=max(16,(n+15)//16*16)
    bf_momentum=train.bf16_constant(momentum);bf_previous=train.bf16_constant(1-bf_momentum);bf_divisor=train.bf16_constant(n-1)
    @T.prim_func
    def main(X:T.Tensor((m,n),dtype),W:T.Tensor((channels,),dtype),B:T.Tensor((channels,),dtype),RM:T.Tensor((channels,),'float32'),RV:T.Tensor((channels,),'float32'),Y:T.Tensor((m,n),dtype),U:T.Tensor((m,2),'float32')):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((rows,pitch),dtype);y=T.alloc_shared((rows,pitch),dtype)
            w=T.alloc_shared((channels,),dtype);b=T.alloc_shared((channels,),dtype)
            old_mean=T.alloc_shared((channels,),'float32');old_var=T.alloc_shared((channels,),'float32')
            accum=T.alloc_shared((rows,16),'float32');update=T.alloc_shared((rows,2),'float32');lanes=T.alloc_shared((rows,32),'float32')
            T.annotate_buffer_versions({x:2,y:2,update:2})
            if weighted:T.copy(W[:channels],w[:channels])
            if biased:T.copy(B[:channels],b[:channels])
            T.copy(RM[:channels],old_mean[:channels]);T.copy(RV[:channels],old_var[:channels])
            for step in T.Pipelined((groups+blocks-1)//blocks,num_stages=2):
                first=(step*blocks+bx)*rows
                if first<m:
                    T.copy(X[first:first+rows,:n],x[:rows,:n])
                    with T.SimtVF(threads=128):
                        chain=T.alloc_fragment((rows,32),'float32')
                        for r,i in T.Parallel(rows,32):
                            chain[r,i]=0
                            for j in T.serial(n//32):
                                chain[r,i]=chain[r,i]+T.cast(x[r,j*32+i%16],'float32')
                                chain[r,i]=chain[r,i]+T.cast(x[r,j*32+i%16+16],'float32')
                            lanes[r,i]=chain[r,i]
                    with T.SimtVF(threads=32):
                        tx=T.get_thread_binding()
                        if tx<rows:
                            accum[tx,0]=lanes[tx,0]
                            for i in T.serial(1,16):accum[tx,0]=accum[tx,0]+lanes[tx,i]
                            accum[tx,2]=0
                            for j in T.serial(n%32):accum[tx,2]=accum[tx,2]+T.cast(x[tx,(n//32)*32+j],'float32')
                            accum[tx,0]=(accum[tx,0]+accum[tx,2])/n
                    with T.SimtVF(threads=128):
                        chain=T.alloc_fragment((rows,32),'float32')
                        for r,i in T.Parallel(rows,32):
                            chain[r,i]=0
                            for j in T.serial(n//32):
                                a=T.cast(x[r,j*32+i%16],'float32')-accum[r,0]
                                bv=T.cast(x[r,j*32+i%16+16],'float32')-accum[r,0]
                                chain[r,i]=chain[r,i]+a*a
                                chain[r,i]=chain[r,i]+bv*bv
                            lanes[r,i]=chain[r,i]
                    with T.SimtVF(threads=32):
                        tx=T.get_thread_binding()
                        if tx<rows:
                            accum[tx,1]=lanes[tx,0]
                            for i in T.serial(1,16):accum[tx,1]=accum[tx,1]+lanes[tx,i]
                            accum[tx,2]=0
                            for j in T.serial(n%32):
                                value=T.cast(x[tx,(n//32)*32+j],'float32')-accum[tx,0]
                                accum[tx,2]=accum[tx,2]+value*value
                            accum[tx,1]=accum[tx,1]+accum[tx,2]
                            old_m=T.cast(T.cast(old_mean[(first+tx)%channels],dtype),'float32')
                            old_v=T.cast(T.cast(old_var[(first+tx)%channels],dtype),'float32')
                            if dtype=='bfloat16':
                                mom=T.float32(bf_momentum);previous=T.float32(bf_previous)
                                mean=T.cast(T.cast(accum[tx,0],dtype),'float32')
                                variance_sum=T.cast(T.cast(accum[tx,1],dtype),'float32')
                                unbiased=T.cast(T.cast(variance_sum/T.float32(bf_divisor),dtype),'float32')
                                mean_new=T.cast(T.cast(mom*mean,dtype),'float32')
                                mean_old=T.cast(T.cast(previous*old_m,dtype),'float32')
                                var_old=T.cast(T.cast(previous*old_v,dtype),'float32')
                                update[tx,0]=T.cast(T.cast(mean_new+mean_old,dtype),'float32')
                                update[tx,1]=T.cast(T.cast(mom*unbiased+var_old,dtype),'float32')
                            else:
                                update[tx,0]=T.cast(T.cast(momentum*accum[tx,0]+(1-momentum)*old_m,dtype),'float32')
                                update[tx,1]=T.cast(T.cast(momentum*(accum[tx,1]/(n-1))+(1-momentum)*old_v,dtype),'float32')
                            accum[tx,1]=T.rsqrt(accum[tx,1]/n+eps)
                    T.copy(update[:rows,:2],U[first:first+rows,:2])
                    with T.SimtVF(threads=256):
                        values=T.alloc_fragment((rows,width),'float32')
                        for r,i in T.Parallel(rows,width):
                            if i<n:
                                value=(T.cast(x[r,i],'float32')-accum[r,0])*accum[r,1]
                                if weighted:value=value*T.cast(w[(first+r)%channels],'float32')
                                if biased:value=value+T.cast(b[(first+r)%channels],'float32')
                                values[r,i]=value
                        for r,i in T.Parallel(rows,width):
                            if i<n:y[r,i]=values[r,i]
                    T.copy(y[:rows,:n],Y[first:first+rows,:n])
    return tilelang.compile(main,target='ascend',out_idx=[5,6])
def build_train(batch,channels,n,dtype,eps,momentum,weighted,biased):
    size=4 if dtype=='float32' else 2
    pitch=max(16,(n+15)//16*16)
    footprint=16*pitch*size+2*channels*size+8*channels+1024
    if 1<n<=4096 and batch*channels%4==0 and footprint<192*1024:return build_batched(batch,channels,n,dtype,eps,momentum,weighted,biased)
    return original.build_train(batch,channels,n,dtype,eps,momentum,weighted,biased)
