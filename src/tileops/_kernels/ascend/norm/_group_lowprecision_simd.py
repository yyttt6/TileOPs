"""Aligned low-precision GroupNorm SIMD; private-output fallback for tails."""
import functools,math
import tilelang
import tilelang.ascend.language as T
from . import _group_lowprecision_resident as fallback
@functools.lru_cache(maxsize=128)
def build_simd(batch,channels,spatial,groups,dtype,eps,weighted,biased):
    cpg=channels//groups;m=batch*groups;n=cpg*spatial;pitch=n+128;blocks=min(64,m)
    assert n%128==0 and spatial%16==0 and 8*pitch+12*channels<208*1024
    zero_inverse=1.0/math.sqrt(eps)
    @T.prim_func
    def main(X:T.Tensor((m,n),dtype),W:T.Tensor((channels,),dtype),B:T.Tensor((channels,),dtype),Y:T.Tensor((m,n),dtype)):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((pitch,),dtype);y=T.alloc_shared((pitch,),dtype)
            w=T.alloc_shared((channels,),dtype);b=T.alloc_shared((channels,),dtype)
            wf=T.alloc_shared((channels,),'float32');bf=T.alloc_shared((channels,),'float32')
            T.annotate_buffer_versions({x:2,y:2})
            if weighted:T.copy(W[:channels],w[:channels])
            if biased:T.copy(B[:channels],b[:channels])
            with T.SimtVF(threads=128):
                for c in T.Parallel(channels):
                    if weighted:wf[c]=T.cast(w[c],'float32')
                    if biased:bf[c]=T.cast(b[c],'float32')
            for step in T.Pipelined((m+blocks-1)//blocks,num_stages=2):
                row=step*blocks+bx
                if row<m:
                    T.copy(X[row,:n],x[:n])
                    with T.SimdVF():
                        mask=T.simd.pset(32);narrow=T.simd.pset(16)
                        total=T.simd.alloc_local(1,'float32');variance=T.simd.alloc_local(1,'float32')
                        total[0]=T.simd.vdup(T.float32(0),'float32');variance[0]=T.simd.vdup(T.float32(0),'float32')
                        for i in range(n//128):
                            packed=T.simd.vld(x[i*128])
                            even=T.simd.vcvt(packed,'float32',part=0);odd=T.simd.vcvt(packed,'float32',part=1)
                            total[0]=T.simd.vadd(total[0],T.simd.vadd(even,odd,mask),mask)
                        den=T.simd.vdup(T.float32(n),'float32')
                        sum_value=T.simd.vdupv(T.simd.vcadd(total[0],mask),mask)
                        quotient=T.simd.vdiv(sum_value,den,mask)
                        residual=T.simd.alloc_local(1,'float32')
                        residual[0]=T.simd.vneg(quotient,mask)
                        T.simd.vmadd(residual[0],den,sum_value,mask)
                        mean=T.simd.vadd(quotient,T.simd.vdiv(residual[0],den,mask),mask)
                        for i in range(n//128):
                            packed=T.simd.vld(x[i*128])
                            even=T.simd.vsub(T.simd.vcvt(packed,'float32',part=0),mean,mask)
                            odd=T.simd.vsub(T.simd.vcvt(packed,'float32',part=1),mean,mask)
                            squares=T.simd.vadd(T.simd.vmul(even,even,mask),T.simd.vmul(odd,odd,mask),mask)
                            variance[0]=T.simd.vadd(variance[0],squares,mask)
                        square_sum=T.simd.vdupv(T.simd.vcadd(variance[0],mask),mask)
                        variance_quotient=T.simd.vdiv(square_sum,den,mask)
                        residual[0]=T.simd.vneg(variance_quotient,mask)
                        T.simd.vmadd(residual[0],den,square_sum,mask)
                        var=T.simd.vadd(variance_quotient,T.simd.vdiv(residual[0],den,mask),mask)
                        inverse=T.simd.vdiv(T.simd.vdup(T.float32(1),'float32'),T.simd.vsqrt(T.simd.vadds(var,T.float32(eps),mask),mask),mask)
                        inverse=T.simd.vsel(T.simd.vdup(T.float32(zero_inverse),'float32'),inverse,T.simd.vcmps(var,T.float32(0),mask,op='eq'))
                        offset=T.simd.alloc_local(1,'float32');ev=T.simd.alloc_local(1,'float32');od=T.simd.alloc_local(1,'float32')
                        for c in range(cpg):
                            channel=(row%groups)*cpg+c
                            scale=inverse
                            if weighted:scale=T.simd.vmul(scale,T.simd.vld(wf[channel],dist='BRC_B32'),mask)
                            bias=T.simd.vdup(T.float32(0),'float32')
                            if biased:bias=T.simd.vld(bf[channel],dist='BRC_B32')
                            if weighted or biased:
                                offset[0]=T.simd.vneg(scale,mask)
                                T.simd.vmadd(offset[0],mean,bias,mask)
                            for i in range((spatial+127)//128):
                                tail=T.simd.update_mask(T.min(64,(spatial-i*128)//2),32)
                                narrow_tail=T.simd.update_mask(T.min(128,spatial-i*128),16)
                                packed=T.simd.vld(x[c*spatial+i*128])
                                ev[0]=T.simd.vcvt(packed,'float32',part=0);od[0]=T.simd.vcvt(packed,'float32',part=1)
                                if weighted or biased:
                                    T.simd.vmadd(ev[0],scale,offset[0],tail);T.simd.vmadd(od[0],scale,offset[0],tail)
                                else:
                                    ev[0]=T.simd.vmul(T.simd.vsub(ev[0],mean,tail),inverse,tail)
                                    od[0]=T.simd.vmul(T.simd.vsub(od[0],mean,tail),inverse,tail)
                                even_out=T.simd.vcvt(ev[0],dtype,part=0,sat=False);odd_out=T.simd.vcvt(od[0],dtype,part=1,sat=False)
                                merged=T.simd.vor(T.reinterpret(even_out,'uint16x128'),T.reinterpret(odd_out,'uint16x128'),narrow)
                                T.simd.vsts(y[c*spatial+i*128],T.reinterpret(merged,dtype+'x128'),narrow_tail)
                    T.copy(y[:n],Y[row,:n])
    return tilelang.compile(main,target='ascend',out_idx=[3])
def build(batch,channels,spatial,groups,dtype,eps,weighted,biased):
    n=channels//groups*spatial
    if n%128==0 and spatial%16==0 and 8*(n+128)+12*channels<208*1024:
        return build_simd(batch,channels,spatial,groups,dtype,eps,weighted,biased)
    return fallback.build(batch,channels,spatial,groups,dtype,eps,weighted,biased)
