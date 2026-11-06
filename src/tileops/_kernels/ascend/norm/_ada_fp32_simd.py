"""FP32-only adaptive normalization candidate; one SIMD VF per pipelined row."""
import functools
import tilelang
import tilelang.ascend.language as T
@functools.lru_cache(maxsize=128)
def build(m,n,dtype,eps,gated):
    assert dtype=='float32' and n%64==0 and 10*n*4<208*1024
    blocks=min(64,m)
    @T.prim_func
    def main(X:T.Tensor((m,n),dtype),W:T.Tensor((m,n),dtype),B:T.Tensor((m,n),dtype),G:T.Tensor((m,n),dtype),Y:T.Tensor((m,n),dtype)):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((n,),dtype)
            w=T.alloc_shared((n,),dtype)
            b=T.alloc_shared((n,),dtype)
            g=T.alloc_shared((n,),dtype)
            y=T.alloc_shared((n,),dtype)
            T.annotate_buffer_versions({x:2,w:2,b:2,g:2,y:2})
            for step in T.Pipelined((m+blocks-1)//blocks,num_stages=2):
                row=step*blocks+bx
                if row<m:
                    T.copy(X[row,:n],x[:n]);T.copy(W[row,:n],w[:n]);T.copy(B[row,:n],b[:n])
                    if gated:T.copy(G[row,:n],g[:n])
                    with T.SimdVF():
                        mask=T.simd.pset(32)
                        total=T.simd.alloc_local(1,'float32')
                        variance=T.simd.alloc_local(1,'float32')
                        total[0]=T.simd.vdup(T.float32(0),'float32')
                        variance[0]=T.simd.vdup(T.float32(0),'float32')
                        origin=T.simd.vld(x[0],dist='BRC_B32')
                        for i in range(n//64):
                            total[0]=T.simd.vadd(total[0],T.simd.vsub(T.simd.vld(x[i*64]),origin,mask),mask)
                        mean=T.simd.vadd(origin,T.simd.vmuls(T.simd.vdupv(T.simd.vcadd(total[0],mask),mask),T.float32(1.0/n),mask),mask)
                        for i in range(n//64):
                            centered=T.simd.vsub(T.simd.vld(x[i*64]),mean,mask)
                            variance[0]=T.simd.vadd(variance[0],T.simd.vmul(centered,centered,mask),mask)
                        var=T.simd.vmuls(T.simd.vdupv(T.simd.vcadd(variance[0],mask),mask),T.float32(1.0/n),mask)
                        denom=T.simd.vsqrt(T.simd.vadds(var,T.float32(eps),mask),mask)
                        inverse=T.simd.vdiv(T.simd.vdup(T.float32(1),'float32'),denom,mask)
                        for i in range(n//64):
                            value=T.simd.vmul(T.simd.vsub(T.simd.vld(x[i*64]),mean,mask),inverse,mask)
                            affine=T.simd.vadd(T.simd.vmul(value,T.simd.vld(w[i*64]),mask),T.simd.vld(b[i*64]),mask)
                            if gated:
                                T.simd.vsts(y[i*64],T.simd.vmul(affine,T.simd.vld(g[i*64]),mask),mask)
                            else:T.simd.vsts(y[i*64],affine,mask)
                    T.copy(y[:n],Y[row,:n])
    return tilelang.compile(main,target='ascend',out_idx=[4])
