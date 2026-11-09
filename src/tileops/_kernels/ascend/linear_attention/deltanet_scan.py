"""Affine prefix composition with on-chip Cube-to-Vector transfer."""
import functools
import tilelang
import tilelang.ascend.language as T

@functools.lru_cache(maxsize=64)
def build_scan(B,N,H,K,V,dtype,save_full=True,group_size=0):
    J=B*N*H;grid=min(32,J);R=K//2
    fullshape=(J,K,V) if save_full else (1,)
    @T.prim_func
    def main(M:T.Tensor((J,K,K),dtype),U:T.Tensor((J,K,V),dtype),shift:T.int32,
             MO:T.Tensor((J,K,K),dtype),UO:T.Tensor((J,K,V),dtype),UF:T.Tensor(fullshape,'float32')):
        with T.Kernel(grid) as bx:
            a=T.alloc_l1((K,K),dtype);b=T.alloc_l1((K,K),dtype);v=T.alloc_l1((V,K),dtype)
            aa=T.alloc_l0a((K,K),dtype);bb=T.alloc_l0b((K,K),dtype);vv=T.alloc_l0b((V,K),dtype)
            mm=T.alloc_l0c((K,K),'float32');uu=T.alloc_l0c((K,V),'float32')
            mp=T.alloc_shared((R,K),'float32');up=T.alloc_shared((R,V),'float32')
            oldm=T.alloc_shared((R,K),dtype);oldu=T.alloc_shared((R,V),dtype)
            newm=T.alloc_shared((R,K),dtype);newu=T.alloc_shared((R,V),dtype)
            for step in T.serial((J+grid-1)//grid):
                j=step*grid+bx
                if j<J:
                    n=(j//H%N)%group_size if group_size else j//H%N;r=j-T.min(n,shift)*H
                    T.copy(M[j,:,:],a);T.copy(M[r,:,:],b,transpose=True)
                    T.copy(a,aa);T.copy(b,bb)
                    T.gemm(aa,bb,mm,transpose_B=True,clear_accum=True)
                    T.dual_copy(mm,mp)
                    T.dual_copy(M[j,:,:],oldm)
                    with T.SimtVF(threads=256):
                        for i,k in T.Parallel(R,K):newm[i,k]=mp[i,k] if n>=shift else T.cast(oldm[i,k],'float32')
                    T.dual_copy(newm,MO[j,:,:])
                    T.copy(U[r,:,:],v,transpose=True);T.copy(v,vv)
                    T.gemm(aa,vv,uu,transpose_B=True,clear_accum=True)
                    T.dual_copy(uu,up);T.dual_copy(U[j,:,:],oldu)
                    with T.SimtVF(threads=256):
                        for i,k in T.Parallel(R,V):
                            value=T.cast(oldu[i,k],'float32')+(up[i,k] if n>=shift else T.float32(0))
                            up[i,k]=value;newu[i,k]=value
                    T.dual_copy(newu,UO[j,:,:])
                    # Only the terminal prefix stage needs the FP32 state;
                    # earlier stages feed their native-dtype UO to the next GEMM.
                    if save_full and shift*2>=N:T.dual_copy(up,UF[j,:,:])
    return tilelang.compile(main,target='ascend',out_idx=[3,4,5])
