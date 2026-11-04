"""Short aligned rows: one SIMD VF, explicit even/odd FP16/BF16 conversion."""
import functools
import tilelang
import tilelang.ascend.language as T
@functools.lru_cache(maxsize=128)
def build_batched(m,n,dtype,eps,weighted,rows=32,threads=128,max_blocks=64):
    assert dtype in ('float16','bfloat16') and n%128==0 and n<=32768 and m%rows==0
    assert (2*rows+1)*n*2<208*1024
    groups=m//rows;blocks=min(max_blocks,groups)
    @T.prim_func
    def main(X:T.Tensor((m,n),dtype),W:T.Tensor((n,),dtype),Y:T.Tensor((m,n),dtype)):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((rows,n),dtype)
            w=T.alloc_shared((n,),dtype)
            T.annotate_buffer_versions({x:2})
            if weighted:T.copy(W[:n],w[:n])
            for step in T.Pipelined((groups+blocks-1)//blocks,num_stages=2):
                first=(step*blocks+bx)*rows
                if first<m:
                    T.copy(X[first:first+rows,:n],x[:rows,:n])
                    with T.SimdVF():
                        mask=T.simd.pset(32)
                        narrow_mask=T.simd.pset(16)
                        for r in range(rows):
                            total=T.simd.alloc_local(1,'float32')
                            total[0]=T.simd.vdup(T.float32(0),'float32')
                            for i in range(n//128):
                                packed=T.simd.vld(x[r,i*128])
                                even=T.simd.vcvt(packed,'float32',part=0)
                                odd=T.simd.vcvt(packed,'float32',part=1)
                                squares=T.simd.vadd(T.simd.vmul(even,even,mask),T.simd.vmul(odd,odd,mask),mask)
                                total[0]=T.simd.vadd(total[0],squares,mask)
                            mean=T.simd.vmuls(T.simd.vdupv(T.simd.vcadd(total[0],mask),mask),T.float32(1.0/n),mask)
                            denom=T.simd.vsqrt(T.simd.vadds(mean,T.float32(eps),mask),mask)
                            inverse=T.simd.vdiv(T.simd.vdup(T.float32(1),'float32'),denom,mask)
                            for i in range(n//128):
                                packed=T.simd.vld(x[r,i*128])
                                even=T.simd.vmul(T.simd.vcvt(packed,'float32',part=0),inverse,mask)
                                odd=T.simd.vmul(T.simd.vcvt(packed,'float32',part=1),inverse,mask)
                                if weighted:
                                    weight=T.simd.vld(w[i*128])
                                    even=T.simd.vmul(even,T.simd.vcvt(weight,'float32',part=0),mask)
                                    odd=T.simd.vmul(odd,T.simd.vcvt(weight,'float32',part=1),mask)
                                out_even=T.simd.vcvt(even,dtype,part=0,sat=False)
                                out_odd=T.simd.vcvt(odd,dtype,part=1,sat=False)
                                merged=T.simd.vor(T.reinterpret(out_even,'uint16x128'),T.reinterpret(out_odd,'uint16x128'),narrow_mask)
                                T.simd.vsts(x[r,i*128],T.reinterpret(merged,dtype+'x128'),narrow_mask)
                    T.copy(x[:rows,:n],Y[first:first+rows,:n],l2_cache_ctrl=0)
    return tilelang.compile(main,target='ascend',out_idx=[2])


def select_build(original,m,n,dtype,eps,weighted):
    if dtype in ('float16','bfloat16') and 0<n<=32768 and n%128==0:
        rows=min(32,max(1,8192//n))
        if m%rows==0:return build_batched(m,n,dtype,eps,weighted,rows,128,64)
    return original(m,n,dtype,eps,weighted)
