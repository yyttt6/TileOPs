"""Aligned FP32 training channels with SIMD centered statistics and output."""
import functools
import tilelang
import tilelang.ascend.language as T
from . import _batch_grouped as base
@functools.lru_cache(maxsize=128)
def build_simd(batch,channels,spatial,dtype,eps,momentum,training,tracked,weighted,biased):
    n=batch*spatial;rows=4;pitch=(spatial+63)//64*64;groups=channels//rows;blocks=min(64,groups);unbiased=n/(n-1)
    @T.prim_func
    def main(X:T.Tensor((batch,channels,spatial),dtype),RM:T.Tensor((channels,),'float32'),RV:T.Tensor((channels,),'float32'),W:T.Tensor((channels,),'float32'),B:T.Tensor((channels,),'float32'),Y:T.Tensor((batch,channels,spatial),dtype),Mean:T.Tensor((channels,),'float32'),Rstd:T.Tensor((channels,),'float32')):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((batch,rows,pitch),dtype);y=T.alloc_shared((batch,rows,pitch),dtype)
            w=T.alloc_shared((channels,),'float32');b=T.alloc_shared((channels,),'float32')
            old_m=T.alloc_shared((channels,),'float32');old_v=T.alloc_shared((channels,),'float32')
            stats=T.alloc_shared((rows,3,64),'float32')
            if weighted:T.copy(W[:channels],w[:channels])
            if biased:T.copy(B[:channels],b[:channels])
            if tracked:
                T.copy(RM[:channels],old_m[:channels]);T.copy(RV[:channels],old_v[:channels])
            for step in T.serial((groups+blocks-1)//blocks):
                first=(step*blocks+bx)*rows
                if first<channels:
                    for nb in T.serial(batch):T.copy(X[nb,first:first+rows,:spatial],x[nb,:rows,:spatial])
                    with T.SimdVF():
                        for r in range(rows):
                            mask=T.simd.pset(32)
                            zero=T.simd.vdup(T.float32(0),'float32')
                            total=T.simd.alloc_local(1,'float32');variance=T.simd.alloc_local(1,'float32')
                            total[0]=T.simd.vdup(T.float32(0),'float32');variance[0]=T.simd.vdup(T.float32(0),'float32')
                            for nb in range(batch):
                                for i in range(pitch//64):
                                    tail=T.simd.update_mask(T.min(64,spatial-i*64),32)
                                    value=T.simd.vsel(T.simd.vld(x[nb,r,i*64]),zero,tail)
                                    total[0]=T.simd.vadd(total[0],value,mask)
                            mean=T.simd.vdiv(T.simd.vdupv(T.simd.vcadd(total[0],mask),mask),T.simd.vdup(T.float32(n),'float32'),mask)
                            for nb in range(batch):
                                for i in range(pitch//64):
                                    tail=T.simd.update_mask(T.min(64,spatial-i*64),32)
                                    centered=T.simd.vsel(T.simd.vsub(T.simd.vld(x[nb,r,i*64]),mean,mask),zero,tail)
                                    variance[0]=T.simd.vadd(variance[0],T.simd.vmul(centered,centered,mask),mask)
                            var=T.simd.vdiv(T.simd.vdupv(T.simd.vcadd(variance[0],mask),mask),T.simd.vdup(T.float32(n),'float32'),mask)
                            inverse=T.simd.vdiv(T.simd.vdup(T.float32(1),'float32'),T.simd.vsqrt(T.simd.vadds(var,T.float32(eps),mask),mask),mask)
                            T.simd.vsts(stats[r,0,0],mean,mask);T.simd.vsts(stats[r,1,0],var,mask);T.simd.vsts(stats[r,2,0],inverse,mask)
                            for nb in range(batch):
                                for i in range(pitch//64):
                                    tail=T.simd.update_mask(T.min(64,spatial-i*64),32)
                                    value=T.simd.vmul(T.simd.vsub(T.simd.vld(x[nb,r,i*64]),mean,mask),inverse,mask)
                                    if weighted:value=T.simd.vmul(value,T.simd.vld(w[first+r],dist='BRC_B32'),mask)
                                    if biased:value=T.simd.vadd(value,T.simd.vld(b[first+r],dist='BRC_B32'),mask)
                                    T.simd.vsts(y[nb,r,i*64],value,tail)
                    with T.SimtVF(threads=32):
                        for r in T.Parallel(rows):
                            Mean[first+r]=stats[r,0,0];Rstd[first+r]=stats[r,2,0]
                            if tracked:
                                RM[first+r]=(1-momentum)*old_m[first+r]+momentum*stats[r,0,0]
                                RV[first+r]=(1-momentum)*old_v[first+r]+momentum*stats[r,1,0]*unbiased
                    for nb in T.serial(batch):T.copy(y[nb,:rows,:spatial],Y[nb,first:first+rows,:spatial])
    return tilelang.compile(main,target='ascend',out_idx=[5,6,7])
def build(batch,channels,spatial,dtype,eps,momentum,training,tracked,weighted,biased):
    if channels%4==0 and 32*batch*((spatial+63)//64*64)+16*channels+3072<208*1024 and dtype=='float32' and training and spatial%16==0 and batch*spatial%64==0 and 1<batch*spatial<=8192:
        return build_simd(batch,channels,spatial,dtype,eps,momentum,training,tracked,weighted,biased)
    return base.build(batch,channels,spatial,dtype,eps,momentum,training,tracked,weighted,biased)
