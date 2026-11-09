"""Aligned wide RMSNorm: SIMD accumulation and pipelined bounded-UB chunks."""
import functools
import tilelang
import tilelang.ascend.language as T
from tileops.kernels.ascend.norm import _rms_resident as narrow
from tileops.kernels.ascend.norm import _rms_grouped_inverse as grouped
from tileops.kernels.ascend.norm import _rms_inplace as resident
@functools.lru_cache(maxsize=128)
def build_wide(m,n,dtype,eps,weighted):
    assert dtype in ('float16','bfloat16') and n>=16384 and n%16384==0
    width=16384;chunks=n//width;blocks=min(64,m)
    @T.prim_func
    def main(X:T.Tensor((m,n),dtype),W:T.Tensor((n,),dtype),Y:T.Tensor((m,n),dtype)):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((width,),dtype);w=T.alloc_shared((width,),dtype);y=T.alloc_shared((width,),dtype)
            accum=T.alloc_shared((64,),'float32')
            T.annotate_buffer_versions({x:2,w:2,y:2})
            for step in T.serial((m+blocks-1)//blocks):
                row=step*blocks+bx
                if row<m:
                    T.clear(accum)
                    for chunk in T.Pipelined(chunks,num_stages=2):
                        start=chunk*width
                        T.copy(X[row,start:start+width],x[:width])
                        with T.SimdVF():
                            mask=T.simd.pset(32)
                            total=T.simd.alloc_local(1,'float32');total[0]=T.simd.vdup(T.float32(0),'float32')
                            for i in range(width//128):
                                packed=T.simd.vld(x[i*128]);even=T.simd.vcvt(packed,'float32',part=0);odd=T.simd.vcvt(packed,'float32',part=1)
                                total[0]=T.simd.vadd(total[0],T.simd.vadd(T.simd.vmul(even,even,mask),T.simd.vmul(odd,odd,mask),mask),mask)
                            T.simd.vsts(accum[0],T.simd.vadd(T.simd.vld(accum[0]),total[0],mask),mask)
                    with T.SimdVF():
                        mask=T.simd.pset(32)
                        mean=T.simd.vmuls(T.simd.vdupv(T.simd.vcadd(T.simd.vld(accum[0]),mask),mask),T.float32(1.0/n),mask)
                        denom=T.simd.vsqrt(T.simd.vadds(mean,T.float32(eps),mask),mask)
                        inverse=T.simd.vdiv(T.simd.vdup(T.float32(1),'float32'),denom,mask)
                        T.simd.vsts(accum[0],inverse,mask)
                    for chunk in T.Pipelined(chunks,num_stages=2):
                        start=chunk*width
                        T.copy(X[row,start:start+width],x[:width])
                        if weighted:T.copy(W[start:start+width],w[:width])
                        with T.SimdVF():
                            mask=T.simd.pset(32);narrow_mask=T.simd.pset(16);inverse=T.simd.vld(accum[0])
                            for i in range(width//128):
                                packed=T.simd.vld(x[i*128]);even=T.simd.vmul(T.simd.vcvt(packed,'float32',part=0),inverse,mask);odd=T.simd.vmul(T.simd.vcvt(packed,'float32',part=1),inverse,mask)
                                if weighted:
                                    weight=T.simd.vld(w[i*128]);even=T.simd.vmul(even,T.simd.vcvt(weight,'float32',part=0),mask);odd=T.simd.vmul(odd,T.simd.vcvt(weight,'float32',part=1),mask)
                                out_even=T.simd.vcvt(even,dtype,part=0,sat=False);out_odd=T.simd.vcvt(odd,dtype,part=1,sat=False)
                                merged=T.simd.vor(T.reinterpret(out_even,'uint16x128'),T.reinterpret(out_odd,'uint16x128'),narrow_mask)
                                T.simd.vsts(y[i*128],T.reinterpret(merged,dtype+'x128'),narrow_mask)
                        T.copy(y[:width],Y[row,start:start+width],l2_cache_ctrl=0)
    return tilelang.compile(main,target='ascend',out_idx=[2])
def select_build(original,m,n,dtype,eps,weighted):
    if n==32768:return resident.select_build(original,m,n,dtype,eps,weighted)
    if n<=128:return original(m,n,dtype,eps,weighted)
    if n<=8192 and m%min(64,max(1,16384//n))==0:return grouped.select_build(original,m,n,dtype,eps,weighted)
    # A partial 256-wide SIMT row group mixes reduction rows on SDK 9.2.
    # Use the already-validated one-row SIMD body for this tail only.
    if dtype in ('float16','bfloat16') and n==256 and m%32!=0:
        return narrow.build_batched(m,n,dtype,eps,weighted,rows=1)
    if dtype in ('float16','bfloat16') and n>16384 and n%16384==0:return build_wide(m,n,dtype,eps,weighted)
    return narrow.select_build(original,m,n,dtype,eps,weighted)
