"""Aligned LayerNorm: centered variance and affine in one pipelined SIMD VF."""
import functools
import tilelang
import tilelang.ascend.language as T
@functools.lru_cache(maxsize=128)
def build_simd(m,n,dtype,eps,weighted,biased,rows=1,threads=256):
    assert rows==2 and m%rows==0 and dtype=='float32' and n%256==0
    size=4 if dtype=='float32' else 2
    lanes=64 if dtype=='float32' else 128
    assert (4*rows+2)*n*size<208*1024
    blocks=min(64,m//rows)
    @T.prim_func
    def main(X:T.Tensor((m,n),dtype),W:T.Tensor((n,),dtype),B:T.Tensor((n,),dtype),Y:T.Tensor((m,n),dtype)):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((rows,n),dtype);y=T.alloc_shared((rows,n),dtype)
            w=T.alloc_shared((n,),dtype);b=T.alloc_shared((n,),dtype)
            T.annotate_buffer_versions({x:2,y:2})
            if weighted:T.copy(W[:n],w[:n])
            if biased:T.copy(B[:n],b[:n])
            for step in T.Pipelined((m//rows+blocks-1)//blocks,num_stages=2):
                row=(step*blocks+bx)*rows
                if row<m:
                    T.copy(X[row:row+rows,:n],x[:rows,:n])
                    with T.SimdVF():
                        for r in range(rows):
                            mask=T.simd.pset(32);narrow_mask=T.simd.pset(16)
                            total=T.simd.alloc_local(4,'float32');variance=T.simd.alloc_local(4,'float32')
                            for k in T.unroll(4):
                                total[k]=T.simd.vdup(T.float32(0),'float32')
                                variance[k]=T.simd.vdup(T.float32(0),'float32')
                            for i in range(n//256):
                                for k in T.unroll(4):
                                    total[k]=T.simd.vadd(total[k],T.simd.vld(x[r,i*256+k*64]),mask)
                            total[0]=T.simd.vadd(T.simd.vadd(total[0],total[1],mask),T.simd.vadd(total[2],total[3],mask),mask)
                            mean=T.simd.vmuls(T.simd.vdupv(T.simd.vcadd(total[0],mask),mask),T.float32(1.0/n),mask)
                            for i in range(n//256):
                                for k in T.unroll(4):
                                    centered=T.simd.vsub(T.simd.vld(x[r,i*256+k*64]),mean,mask)
                                    variance[k]=T.simd.vadd(variance[k],T.simd.vmul(centered,centered,mask),mask)
                            variance[0]=T.simd.vadd(T.simd.vadd(variance[0],variance[1],mask),T.simd.vadd(variance[2],variance[3],mask),mask)
                            var=T.simd.vmuls(T.simd.vdupv(T.simd.vcadd(variance[0],mask),mask),T.float32(1.0/n),mask)
                            denom=T.simd.vsqrt(T.simd.vadds(var,T.float32(eps),mask),mask)
                            inverse=T.simd.vdiv(T.simd.vdup(T.float32(1),'float32'),denom,mask)
                            for i in range(n//lanes):
                                packed=T.simd.vld(x[r,i*lanes])
                                if dtype=='float32':
                                    value=T.simd.vmul(T.simd.vsub(packed,mean,mask),inverse,mask)
                                    if weighted:value=T.simd.vmul(value,T.simd.vld(w[i*lanes]),mask)
                                    if biased:value=T.simd.vadd(value,T.simd.vld(b[i*lanes]),mask)
                                    T.simd.vsts(y[r,i*lanes],value,mask)
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
                                    T.simd.vsts(y[r,i*lanes],T.reinterpret(merged,dtype+'x128'),narrow_mask)
                    T.copy(y[:rows,:n],Y[row:row+rows,:n])
    return tilelang.compile(main,target='ascend',out_idx=[3])

from tileops.kernels.ascend.norm import _layer_packed as original

def build(m,n,dtype,eps,weighted,biased,rows=1,threads=256):
    if dtype=='float32' and n%256==0 and m%2==0 and rows==2:return build_simd(m,n,dtype,eps,weighted,biased,rows,threads)
    return original.build(m,n,dtype,eps,weighted,biased,rows,threads)
