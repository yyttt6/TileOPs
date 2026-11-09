"""Fused rounded residual/RMS in one SIMD VF with two-stage row buffering."""
import functools
import tilelang
import tilelang.ascend.language as T
@functools.lru_cache(maxsize=128)
def build(m,n,dtype,eps,weighted):
    assert dtype in ('float16','bfloat16') and 0<n<=16384 and n%128==0
    rows=4 if n<=4096 and m%4==0 else (2 if n<=8192 and m%2==0 else 1)
    groups=m//rows;blocks=min(64,groups)
    @T.prim_func
    def main(X:T.Tensor((m,n),dtype),R:T.Tensor((m,n),dtype),W:T.Tensor((n,),dtype),Y:T.Tensor((m,n),dtype),O:T.Tensor((m,n),dtype)):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((rows,n),dtype);residual=T.alloc_shared((rows,n),dtype)
            y=T.alloc_shared((rows,n),dtype);w=T.alloc_shared((n,),dtype)
            T.annotate_buffer_versions({x:2,residual:2,y:2})
            if weighted:T.copy(W[:n],w[:n])
            for step in T.Pipelined((groups+blocks-1)//blocks,num_stages=2):
                row=(step*blocks+bx)*rows
                if row<m:
                    T.copy(X[row:row+rows,:n],x[:rows,:n]);T.copy(R[row:row+rows,:n],residual[:rows,:n])
                    with T.SimdVF():
                        mask=T.simd.pset(32);narrow=T.simd.pset(16)
                        for r in range(rows):
                            total=T.simd.alloc_local(1,'float32');total[0]=T.simd.vdup(T.float32(0),'float32')
                            for i in range(n//128):
                                px=T.simd.vld(x[r,i*128]);pr=T.simd.vld(residual[r,i*128])
                                even=T.simd.vadd(T.simd.vcvt(px,'float32',part=0),T.simd.vcvt(pr,'float32',part=0),mask)
                                odd=T.simd.vadd(T.simd.vcvt(px,'float32',part=1),T.simd.vcvt(pr,'float32',part=1),mask)
                                lo=T.simd.vcvt(even,dtype,part=0,sat=False);hi=T.simd.vcvt(odd,dtype,part=1,sat=False)
                                bits=T.simd.vor(T.reinterpret(lo,'uint16x128'),T.reinterpret(hi,'uint16x128'),narrow)
                                rounded=T.reinterpret(bits,dtype+'x128');T.simd.vsts(x[r,i*128],rounded,narrow)
                                a=T.simd.vcvt(rounded,'float32',part=0);b=T.simd.vcvt(rounded,'float32',part=1)
                                squares=T.simd.vadd(T.simd.vmul(a,a,mask),T.simd.vmul(b,b,mask),mask)
                                total[0]=T.simd.vadd(total[0],squares,mask)
                            mean=T.simd.vmuls(T.simd.vdupv(T.simd.vcadd(total[0],mask),mask),T.float32(1.0/n),mask)
                            denom=T.simd.vsqrt(T.simd.vadds(mean,T.float32(eps),mask),mask)
                            inverse=T.simd.vdiv(T.simd.vdup(T.float32(1),'float32'),denom,mask)
                            for i in range(n//128):
                                packed=T.simd.vld(x[r,i*128])
                                even=T.simd.vmul(T.simd.vcvt(packed,'float32',part=0),inverse,mask)
                                odd=T.simd.vmul(T.simd.vcvt(packed,'float32',part=1),inverse,mask)
                                if weighted:
                                    pw=T.simd.vld(w[i*128]);even=T.simd.vmul(even,T.simd.vcvt(pw,'float32',part=0),mask);odd=T.simd.vmul(odd,T.simd.vcvt(pw,'float32',part=1),mask)
                                lo=T.simd.vcvt(even,dtype,part=0,sat=False);hi=T.simd.vcvt(odd,dtype,part=1,sat=False)
                                bits=T.simd.vor(T.reinterpret(lo,'uint16x128'),T.reinterpret(hi,'uint16x128'),narrow)
                                T.simd.vsts(y[r,i*128],T.reinterpret(bits,dtype+'x128'),narrow)
                    T.copy(y[:rows,:n],Y[row:row+rows,:n]);T.copy(x[:rows,:n],O[row:row+rows,:n])
    return tilelang.compile(main,target='ascend',out_idx=[3,4])
