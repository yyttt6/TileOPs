"""SIMD per-head normalization with pipelined token copies."""
import functools
import tilelang
import tilelang.ascend.language as T
@functools.lru_cache(maxsize=128)
def build(tokens,heads,kv_heads,dim,cache_rows,rotary,dtype,cache_dtype,eps,layout):
    assert dim%128==0 and dtype in ('float16','bfloat16')
    width=tilelang.next_power_of_2(dim);active=heads+kv_heads;rows=tilelang.next_power_of_2(active)
    packed=(heads+2*kv_heads)*dim;blocks=min(64,tokens);half=rotary//2
    cache_pitch=max(16,(rotary+15)//16*16)
    assert active*dim*12+dim*4+cache_pitch*8<192*1024
    @T.prim_func
    def main(QKV:T.Tensor((tokens,packed),dtype),QW:T.Tensor((dim,),dtype),KW:T.Tensor((dim,),dtype),Cache:T.Tensor((cache_rows,rotary),cache_dtype),Positions:T.Tensor((tokens,),'int64')):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((active*dim,),dtype);y=T.alloc_shared((active*dim,),dtype)
            qw=T.alloc_shared((dim,),dtype);kw=T.alloc_shared((dim,),dtype)
            normalized=T.alloc_shared((active*dim,),'float32')
            cache=T.alloc_shared((cache_pitch,),cache_dtype)
            T.annotate_buffer_versions({x:2,y:2,cache:2})
            T.copy(QW[:dim],qw[:dim]);T.copy(KW[:dim],kw[:dim])
            for step in T.Pipelined((tokens+blocks-1)//blocks,num_stages=2):
                token=step*blocks+bx
                if token<tokens:
                    T.copy(QKV[token,:active*dim],x[:active*dim])
                    with T.SimtVF(threads=32):
                        for i in T.Parallel(cache_pitch):
                            if i<rotary:cache[i]=Cache[Positions[token],i]
                    with T.SimdVF():
                        mask=T.simd.pset(32)
                        for h in range(active):
                            total=T.simd.alloc_local(1,'float32')
                            total[0]=T.simd.vdup(T.float32(0),'float32')
                            for i in range(dim//128):
                                packed_x=T.simd.vld(x[h*dim+i*128])
                                xe=T.simd.vcvt(packed_x,'float32',part=0)
                                xo=T.simd.vcvt(packed_x,'float32',part=1)
                                total[0]=T.simd.vadd(total[0],T.simd.vadd(T.simd.vmul(xe,xe,mask),T.simd.vmul(xo,xo,mask),mask),mask)
                            mean=T.simd.vmuls(T.simd.vdupv(T.simd.vcadd(total[0],mask),mask),T.float32(1.0/dim),mask)
                            denom=T.simd.vsqrt(T.simd.vadds(mean,T.float32(eps),mask),mask)
                            inverse=T.simd.vdiv(T.simd.vdup(T.float32(1),'float32'),denom,mask)
                            for i in range(dim//128):
                                packed_x=T.simd.vld(x[h*dim+i*128])
                                weight=T.simd.alloc_local(1,dtype)
                                if h<heads:weight[0]=T.simd.vld(qw[i*128])
                                else:weight[0]=T.simd.vld(kw[i*128])
                                we=T.simd.vcvt(weight[0],'float32',part=0)
                                wo=T.simd.vcvt(weight[0],'float32',part=1)
                                xe=T.simd.vcvt(packed_x,'float32',part=0)
                                xo=T.simd.vcvt(packed_x,'float32',part=1)
                                even=T.simd.vmul(xe,T.simd.vmul(inverse,we,mask),mask)
                                odd=T.simd.vmul(xo,T.simd.vmul(inverse,wo,mask),mask)
                                low,high=T.simd.vintlv(even,odd)
                                T.simd.vsts(normalized[h*dim+i*128],low,mask)
                                T.simd.vsts(normalized[h*dim+i*128+64],high,mask)
                    with T.SimtVF(threads=128):
                        for h,i in T.Parallel(rows,width):
                            if h<active and i<dim:
                                if i<rotary:
                                    if layout=='neox':
                                        index=i%half;peer=T.if_then_else(i<half,i+half,i-half)
                                        sign=T.if_then_else(i<half,T.float32(-1),T.float32(1))
                                    else:
                                        index=i//2;peer=T.if_then_else(i%2==0,i+1,i-1)
                                        sign=T.if_then_else(i%2==0,T.float32(-1),T.float32(1))
                                    y[h*dim+i]=normalized[h*dim+i]*T.cast(cache[index],'float32')+sign*normalized[h*dim+peer]*T.cast(cache[index+half],'float32')
                                else:y[h*dim+i]=normalized[h*dim+i]
                    T.copy(y[:active*dim],QKV[token,:active*dim])
    return tilelang.compile(main,target='ascend',out_idx=[])
