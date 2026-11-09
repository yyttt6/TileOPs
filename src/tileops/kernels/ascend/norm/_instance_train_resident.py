"""Resident InstanceNorm training preserving original 16-chain sums and update rounding."""
import functools,struct
import tilelang
import tilelang.ascend.language as T
from ._instance_train import bf16_constant
@functools.lru_cache(maxsize=128)
def build_train(batch,channels,n,dtype,eps,momentum,weighted,biased):
    assert 1<n<=4096
    m=batch*channels;blocks=min(64,m);width=tilelang.next_power_of_2(n);pitch=max(16,(n+15)//16*16)
    bf_momentum=bf16_constant(momentum);bf_previous=bf16_constant(1.0-bf_momentum);bf_divisor=bf16_constant(n-1)
    @T.prim_func
    def main(X:T.Tensor((m,n),dtype),W:T.Tensor((channels,),dtype),B:T.Tensor((channels,),dtype),RM:T.Tensor((channels,),'float32'),RV:T.Tensor((channels,),'float32'),Y:T.Tensor((m,n),dtype),U:T.Tensor((m,2),'float32')):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((pitch,),dtype);y=T.alloc_shared((pitch,),dtype)
            w=T.alloc_shared((channels,),dtype);b=T.alloc_shared((channels,),dtype)
            old_mean=T.alloc_shared((channels,),'float32');old_var=T.alloc_shared((channels,),'float32')
            accum=T.alloc_shared((16,),'float32');update=T.alloc_shared((16,),'float32');lanes=T.alloc_shared((32,),'float32')
            T.annotate_buffer_versions({x:2,y:2,update:2})
            if weighted:T.copy(W[:channels],w[:channels])
            if biased:T.copy(B[:channels],b[:channels])
            T.copy(RM[:channels],old_mean[:channels]);T.copy(RV[:channels],old_var[:channels])
            for step in T.Pipelined((m+blocks-1)//blocks,num_stages=2):
                row=step*blocks+bx
                if row<m:
                    T.copy(X[row,:n],x[:n])
                    with T.SimtVF(threads=32):
                        chain=T.alloc_fragment((32,),'float32')
                        for i in T.Parallel(32):
                            chain[i]=0
                            for j in T.serial(n//32):
                                chain[i]=chain[i]+T.cast(x[j*32+i%16],'float32')
                                chain[i]=chain[i]+T.cast(x[j*32+i%16+16],'float32')
                            lanes[i]=chain[i]
                        accum[0]=lanes[0]
                        for i in T.serial(1,16):accum[0]=accum[0]+lanes[i]
                        accum[2]=0
                        for j in T.serial(n%32):accum[2]=accum[2]+T.cast(x[(n//32)*32+j],'float32')
                        accum[0]=(accum[0]+accum[2])/n
                        for i in T.Parallel(32):
                            chain[i]=0
                            for j in T.serial(n//32):
                                a=T.cast(x[j*32+i%16],'float32')-accum[0]
                                bv=T.cast(x[j*32+i%16+16],'float32')-accum[0]
                                chain[i]=chain[i]+a*a
                                chain[i]=chain[i]+bv*bv
                            lanes[i]=chain[i]
                        accum[1]=lanes[0]
                        for i in T.serial(1,16):accum[1]=accum[1]+lanes[i]
                        accum[2]=0
                        for j in T.serial(n%32):
                            value=T.cast(x[(n//32)*32+j],'float32')-accum[0]
                            accum[2]=accum[2]+value*value
                        accum[1]=accum[1]+accum[2]
                        old_m=T.cast(T.cast(old_mean[row%channels],dtype),'float32')
                        old_v=T.cast(T.cast(old_var[row%channels],dtype),'float32')
                        if dtype=='bfloat16':
                            mom=T.float32(bf_momentum);previous=T.float32(bf_previous)
                            mean=T.cast(T.cast(accum[0],dtype),'float32')
                            variance_sum=T.cast(T.cast(accum[1],dtype),'float32')
                            unbiased=T.cast(T.cast(variance_sum/T.float32(bf_divisor),dtype),'float32')
                            mean_new=T.cast(T.cast(mom*mean,dtype),'float32')
                            mean_old=T.cast(T.cast(previous*old_m,dtype),'float32')
                            var_old=T.cast(T.cast(previous*old_v,dtype),'float32')
                            update[0]=T.cast(T.cast(mean_new+mean_old,dtype),'float32')
                            update[1]=T.cast(T.cast(mom*unbiased+var_old,dtype),'float32')
                        else:
                            update[0]=T.cast(T.cast(momentum*accum[0]+(1-momentum)*old_m,dtype),'float32')
                            update[1]=T.cast(T.cast(momentum*(accum[1]/(n-1))+(1-momentum)*old_v,dtype),'float32')
                        accum[1]=T.rsqrt(accum[1]/n+eps)
                    T.copy(update[:2],U[row,:2])
                    with T.SimtVF(threads=min(256,width)):
                        values=T.alloc_fragment((width,),'float32')
                        for i in T.Parallel(width):
                            if i<n:
                                value=(T.cast(x[i],'float32')-accum[0])*accum[1]
                                if weighted:value=value*T.cast(w[row%channels],'float32')
                                if biased:value=value+T.cast(b[row%channels],'float32')
                                values[i]=value
                        for i in T.Parallel(width):
                            if i<n:y[i]=values[i]
                    T.copy(y[:n],Y[row,:n])
    return tilelang.compile(main,target='ascend',out_idx=[5,6])

@functools.lru_cache(maxsize=128)
def build_update(batch,channels,dtype):
    width=32;blocks=(channels+width-1)//width
    @T.prim_func
    def main(U:T.Tensor((batch*channels,2),'float32'),RM:T.Tensor((channels,),'float32'),RV:T.Tensor((channels,),'float32')):
        with T.Kernel(blocks) as bx:
            with T.SimtVF(threads=32):
                means=T.alloc_fragment((width,),'float32');variances=T.alloc_fragment((width,),'float32')
                for i in T.Parallel(width):
                    c=bx*width+i
                    if c<channels:
                        means[i]=0;variances[i]=0
                        for n in T.serial(batch):
                            means[i]=means[i]+U[n*channels+c,0]
                            variances[i]=variances[i]+U[n*channels+c,1]
                        RM[c]=T.cast(T.cast(means[i]*(1.0/batch),dtype),'float32')
                        RV[c]=T.cast(T.cast(variances[i]*(1.0/batch),dtype),'float32')
    return tilelang.compile(main,target='ascend',out_idx=[])
