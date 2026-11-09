"""Fuse initial-state projection with boundary-state assembly."""
import functools
import tilelang
import tilelang.ascend.language as T

@functools.lru_cache(maxsize=64)
def build(B,N,H,K,V,dtype):
    J=B*N*H;grid=min(32,J);R=K//2
    @T.prim_func
    def main(M:T.Tensor((J,K,K),dtype),U:T.Tensor((J,K,V),'float32'),
             H0:T.Tensor((B*H,K,V),dtype),HB:T.Tensor((J,K,V),dtype),
             HT:T.Tensor((B,H,K,V),'float32')):
        with T.MixedKernel(grid) as (bx,sid):
            a=T.alloc_l1((K,K),dtype);b=T.alloc_l1((V,K),dtype)
            aa=T.alloc_l0a((K,K),dtype);bb=T.alloc_l0b((V,K),dtype)
            cc=T.alloc_l0c((K,V),'float32')
            product=T.alloc_shared((R,V),'float32');u=T.alloc_shared((R,V),'float32')
            state=T.alloc_shared((R,V),dtype);initial=T.alloc_shared((R,V),dtype)
            for step in T.Pipelined((J+grid-1)//grid,num_stages=2):
                j=step*grid+bx
                if j<J:
                    batch=j//(N*H);n=j//H%N;h=j%H;r=batch*H+h
                    T.copy(M[j,:,:],a);T.copy(H0[r,:,:],b,transpose=True)
                    T.copy(a,aa);T.copy(b,bb)
                    T.gemm(aa,bb,cc,transpose_B=True,clear_accum=True)
                    T.dual_copy(cc,product);T.dual_copy(U[j,:,:],u)
                    T.dual_copy(H0[r,:,:],initial)
                    with T.SimtVF(threads=256):
                        for i,k in T.Parallel(R,V):
                            product[i,k]=product[i,k]+u[i,k]
                            state[i,k]=product[i,k]
                    if n+1<N:T.dual_copy(state,HB[j+H,:,:])
                    if n==N-1:T.dual_copy(product,HT[batch,h,:,:])
                    if n==0:T.dual_copy(initial,HB[j,:,:])
    return tilelang.compile(main,target='ascend',out_idx=[3,4])
