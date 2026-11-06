"""Keep normalized Q/K values in vector registers through rotary embedding."""
import functools
import tilelang
import tilelang.ascend.language as T
from . import _qk_token_simd as fallback
@functools.lru_cache(maxsize=128)
def build_fused(tokens,heads,kv_heads,dim,cache_rows,rotary,dtype,cache_dtype,eps,layout):
    active=heads+kv_heads;packed=(heads+2*kv_heads)*dim;blocks=min(64,tokens)
    @T.prim_func
    def main(QKV:T.Tensor((tokens,packed),dtype),QW:T.Tensor((dim,),dtype),KW:T.Tensor((dim,),dtype),Cache:T.Tensor((cache_rows,rotary),cache_dtype),Positions:T.Tensor((tokens,),'int64')):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((active*dim,),dtype);y=T.alloc_shared((active*dim,),dtype)
            qw=T.alloc_shared((dim,),dtype);kw=T.alloc_shared((dim,),dtype)
            cache=T.alloc_shared((128,),cache_dtype)
            indices=T.alloc_shared((3,64),'uint32')
            T.annotate_buffer_versions({x:2,y:2,cache:2})
            T.copy(QW[:dim],qw[:dim]);T.copy(KW[:dim],kw[:dim])
            if rotary==64:
                with T.SimtVF(threads=64):
                    for i in T.Parallel(64):
                        indices[0,i]=(i+32)%64;indices[1,i]=i%32;indices[2,i]=i%32+32
            for step in T.Pipelined((tokens+blocks-1)//blocks,num_stages=2):
                token=step*blocks+bx
                if token<tokens:
                    T.copy(QKV[token,:active*dim],x[:active*dim])
                    if rotary==64 and cache_dtype!='float32':T.clear(cache)
                    with T.SimtVF(threads=128):
                        for i in T.Parallel(128):
                            if i<rotary:cache[i]=Cache[Positions[token],i]
                    with T.SimdVF():
                        mask=T.simd.pset(32);narrow=T.simd.pset(16);partial=T.simd.update_mask(32,32)
                        cosine=T.simd.alloc_local(1,'float32');sine=T.simd.alloc_local(1,'float32')
                        if cache_dtype=='float32':
                            if rotary==128:
                                cosine[0]=T.simd.vld(cache[0]);sine[0]=T.simd.vld(cache[64])
                            else:
                                cache_values=T.simd.vld(cache[0])
                                cosine[0]=T.simd.vselr(cache_values,T.simd.vld(indices[1,0]))
                                sine[0]=T.simd.vselr(cache_values,T.simd.vld(indices[2,0]))
                        else:
                            cache_packed=T.simd.vld(cache[0])
                            low,high=T.simd.vintlv(T.simd.vcvt(cache_packed,'float32',part=0),T.simd.vcvt(cache_packed,'float32',part=1))
                            if rotary==128:cosine[0]=low;sine[0]=high
                            else:
                                cosine[0]=T.simd.vselr(low,T.simd.vld(indices[1,0]))
                                sine[0]=T.simd.vselr(low,T.simd.vld(indices[2,0]))
                        qweight=T.simd.vld(qw[0]);kweight=T.simd.vld(kw[0])
                        qwe=T.simd.vcvt(qweight,'float32',part=0);qwo=T.simd.vcvt(qweight,'float32',part=1)
                        kwe=T.simd.vcvt(kweight,'float32',part=0);kwo=T.simd.vcvt(kweight,'float32',part=1)
                        for h in range(active):
                            packed_x=T.simd.vld(x[h*dim]);xe=T.simd.vcvt(packed_x,'float32',part=0);xo=T.simd.vcvt(packed_x,'float32',part=1)
                            squares=T.simd.vadd(T.simd.vmul(xe,xe,mask),T.simd.vmul(xo,xo,mask),mask)
                            mean=T.simd.vmuls(T.simd.vdupv(T.simd.vcadd(squares,mask),mask),T.float32(1.0/dim),mask)
                            inverse=T.simd.vdiv(T.simd.vdup(T.float32(1),'float32'),T.simd.vsqrt(T.simd.vadds(mean,T.float32(eps),mask),mask),mask)
                            even=T.simd.alloc_local(1,'float32');odd=T.simd.alloc_local(1,'float32')
                            if h<heads:
                                even[0]=T.simd.vmul(xe,T.simd.vmul(inverse,qwe,mask),mask);odd[0]=T.simd.vmul(xo,T.simd.vmul(inverse,qwo,mask),mask)
                            else:
                                even[0]=T.simd.vmul(xe,T.simd.vmul(inverse,kwe,mask),mask);odd[0]=T.simd.vmul(xo,T.simd.vmul(inverse,kwo,mask),mask)
                            if layout=='neox':
                                low,high=T.simd.vintlv(even[0],odd[0])
                                if rotary==128:
                                    out_low=T.simd.vsub(T.simd.vmul(low,cosine[0],mask),T.simd.vmul(high,sine[0],mask),mask)
                                    out_high=T.simd.vadd(T.simd.vmul(high,cosine[0],mask),T.simd.vmul(low,sine[0],mask),mask)
                                else:
                                    peer=T.simd.vselr(low,T.simd.vld(indices[0,0]))
                                    signed_peer=T.simd.vsel(T.simd.vneg(peer,mask),peer,partial)
                                    out_low=T.simd.vadd(T.simd.vmul(low,cosine[0],mask),T.simd.vmul(signed_peer,sine[0],mask),mask)
                                    out_high=high
                                out_even,out_odd=T.simd.vdintlv(out_low,out_high)
                            else:
                                re=T.simd.vsub(T.simd.vmul(even[0],cosine[0],mask),T.simd.vmul(odd[0],sine[0],mask),mask)
                                ro=T.simd.vadd(T.simd.vmul(odd[0],cosine[0],mask),T.simd.vmul(even[0],sine[0],mask),mask)
                                if rotary==128:out_even=re;out_odd=ro
                                else:
                                    out_even=T.simd.vsel(re,even[0],partial);out_odd=T.simd.vsel(ro,odd[0],partial)
                            e=T.simd.vcvt(out_even,dtype,part=0,sat=False);o=T.simd.vcvt(out_odd,dtype,part=1,sat=False)
                            merged=T.simd.vor(T.reinterpret(e,'uint16x128'),T.reinterpret(o,'uint16x128'),narrow)
                            T.simd.vsts(y[h*dim],T.reinterpret(merged,dtype+'x128'),narrow)
                    T.copy(y[:active*dim],QKV[token,:active*dim])
    return tilelang.compile(main,target='ascend',out_idx=[])
def build(tokens,heads,kv_heads,dim,cache_rows,rotary,dtype,cache_dtype,eps,layout):
    if dim==128 and rotary in (64,128) and (heads+kv_heads)*dim*8+dim*4+128*8+768<192*1024:
        return build_fused(tokens,heads,kv_heads,dim,cache_rows,rotary,dtype,cache_dtype,eps,layout)
    return fallback.build(tokens,heads,kv_heads,dim,cache_rows,rotary,dtype,cache_dtype,eps,layout)
