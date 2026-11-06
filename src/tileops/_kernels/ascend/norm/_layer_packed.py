"""Aligned LayerNorm: centered variance and affine in one pipelined SIMD VF."""
import functools
import tilelang
import tilelang.ascend.language as T
@functools.lru_cache(maxsize=128)
def build(m,n,dtype,eps,weighted,biased,rows=1,threads=256):
    assert rows==1 and dtype in ('float32','float16','bfloat16') and n%128==0
    size=4 if dtype=='float32' else 2
    lanes=64 if dtype=='float32' else 128
    assert 6*n*size<208*1024
    blocks=min(64,m)
    @T.prim_func
    def main(X:T.Tensor((m,n),dtype),W:T.Tensor((n,),dtype),B:T.Tensor((n,),dtype),Y:T.Tensor((m,n),dtype)):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((n,),dtype);y=T.alloc_shared((n,),dtype)
            w=T.alloc_shared((n,),dtype);b=T.alloc_shared((n,),dtype)
            T.annotate_buffer_versions({x:2,y:2})
            if weighted:T.copy(W[:n],w[:n])
            if biased:T.copy(B[:n],b[:n])
            for step in T.Pipelined((m+blocks-1)//blocks,num_stages=2):
                row=step*blocks+bx
                if row<m:
                    T.copy(X[row,:n],x[:n])
                    with T.SimdVF():
                        mask=T.simd.pset(32);narrow_mask=T.simd.pset(16)
                        total=T.simd.alloc_local(1,'float32');variance=T.simd.alloc_local(1,'float32')
                        total[0]=T.simd.vdup(T.float32(0),'float32')
                        variance[0]=T.simd.vdup(T.float32(0),'float32')
                        for i in range(n//lanes):
                            packed=T.simd.vld(x[i*lanes])
                            if dtype=='float32':total[0]=T.simd.vadd(total[0],packed,mask)
                            else:
                                even=T.simd.vcvt(packed,'float32',part=0)
                                odd=T.simd.vcvt(packed,'float32',part=1)
                                total[0]=T.simd.vadd(total[0],T.simd.vadd(even,odd,mask),mask)
                        mean=T.simd.vmuls(T.simd.vdupv(T.simd.vcadd(total[0],mask),mask),T.float32(1.0/n),mask)
                        for i in range(n//lanes):
                            packed=T.simd.vld(x[i*lanes])
                            if dtype=='float32':
                                centered=T.simd.vsub(packed,mean,mask)
                                variance[0]=T.simd.vadd(variance[0],T.simd.vmul(centered,centered,mask),mask)
                            else:
                                even=T.simd.vsub(T.simd.vcvt(packed,'float32',part=0),mean,mask)
                                odd=T.simd.vsub(T.simd.vcvt(packed,'float32',part=1),mean,mask)
                                squares=T.simd.vadd(T.simd.vmul(even,even,mask),T.simd.vmul(odd,odd,mask),mask)
                                variance[0]=T.simd.vadd(variance[0],squares,mask)
                        var=T.simd.vmuls(T.simd.vdupv(T.simd.vcadd(variance[0],mask),mask),T.float32(1.0/n),mask)
                        denom=T.simd.vsqrt(T.simd.vadds(var,T.float32(eps),mask),mask)
                        inverse=T.simd.vdiv(T.simd.vdup(T.float32(1),'float32'),denom,mask)
                        for i in range(n//lanes):
                            packed=T.simd.vld(x[i*lanes])
                            if dtype=='float32':
                                value=T.simd.vmul(T.simd.vsub(packed,mean,mask),inverse,mask)
                                if weighted:value=T.simd.vmul(value,T.simd.vld(w[i*lanes]),mask)
                                if biased:value=T.simd.vadd(value,T.simd.vld(b[i*lanes]),mask)
                                T.simd.vsts(y[i*lanes],value,mask)
                            else:
                                even=T.simd.vmul(T.simd.vsub(T.simd.vcvt(packed,'float32',part=0),mean,mask),inverse,mask)
                                odd=T.simd.vmul(T.simd.vsub(T.simd.vcvt(packed,'float32',part=1),mean,mask),inverse,mask)
                                if weighted:
                                    weight=T.simd.vld(w[i*lanes])
                                    even=T.simd.vmul(even,T.simd.vcvt(weight,'float32',part=0),mask)
                                    odd=T.simd.vmul(odd,T.simd.vcvt(weight,'float32',part=1),mask)
                                if biased:
                                    bias=T.simd.vld(b[i*lanes])
                                    even=T.simd.vadd(even,T.simd.vcvt(bias,'float32',part=0),mask)
                                    odd=T.simd.vadd(odd,T.simd.vcvt(bias,'float32',part=1),mask)
                                out_even=T.simd.vcvt(even,dtype,part=0,sat=False)
                                out_odd=T.simd.vcvt(odd,dtype,part=1,sat=False)
                                merged=T.simd.vor(T.reinterpret(out_even,'uint16x128'),T.reinterpret(out_odd,'uint16x128'),narrow_mask)
                                T.simd.vsts(y[i*lanes],T.reinterpret(merged,dtype+'x128'),narrow_mask)
                    T.copy(y[:n],Y[row,:n])
    return tilelang.compile(main,target='ascend',out_idx=[3])
