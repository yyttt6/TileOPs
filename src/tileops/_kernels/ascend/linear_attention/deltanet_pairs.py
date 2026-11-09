"""Two chunk products sharing a left operand and one host launch."""
import functools
import tilelang
import tilelang.ascend.language as T

@functools.lru_cache(maxsize=64)
def build(M,N1,N2,D,dtype,ta=False,first_mode=0,second_fp32=False,save_full=False):
    J=T.dynamic('jobs');R=M//2;grid=32
    ashape=(J,D,M) if ta else (J,M,D)
    xshape=(J,M,N1) if first_mode==2 else (J,D,N1)
    second_dtype='float32' if second_fp32 else dtype
    fullshape=(J,M,N2) if save_full else (1,)
    @T.prim_func
    def main(A:T.Tensor(ashape,dtype),B:T.Tensor((J,D,N1),dtype),
             C:T.Tensor((J,D,N2),dtype),X:T.Tensor(xshape,dtype),
             O1:T.Tensor((J,M,N1),dtype),O2:T.Tensor((J,M,N2),second_dtype),
             Full:T.Tensor(fullshape,'float32')):
        with T.MixedKernel(grid) as (bx,sid):
            a=T.alloc_l1((M,D),dtype);b=T.alloc_l1((N1,D),dtype);c=T.alloc_l1((N2,D),dtype)
            aa=T.alloc_l0a((M,D),dtype);bb=T.alloc_l0b((N1,D),dtype);cc=T.alloc_l0b((N2,D),dtype)
            p1=T.alloc_l0c((M,N1),'float32');p2=T.alloc_l0c((M,N2),'float32')
            u1=T.alloc_shared((R,N1),'float32');u2=T.alloc_shared((R,N2),'float32')
            x=T.alloc_shared((R,N1),dtype);o1=T.alloc_shared((R,N1),dtype);o2=T.alloc_shared((R,N2),second_dtype)
            for step in T.Pipelined((J+grid-1)//grid,num_stages=2):
                j=step*grid+bx
                if j<J:
                    if ta:T.copy(A[j,:,:],a,transpose=True)
                    else:T.copy(A[j,:,:],a)
                    T.copy(a,aa);T.copy(B[j,:,:],b,transpose=True);T.copy(b,bb)
                    T.gemm(aa,bb,p1,transpose_B=True,clear_accum=True)
                    T.dual_copy(p1,u1)
                    if first_mode==2:T.dual_copy(X[j,:,:],x)
                    with T.SimtVF(threads=256):
                        for i,k in T.Parallel(R,N1):
                            if first_mode==1:o1[i,k]=T.if_then_else(i+sid*R==k,T.float32(1),T.float32(0))-u1[i,k]
                            elif first_mode==2:o1[i,k]=T.cast(x[i,k],'float32')-u1[i,k]
                            else:o1[i,k]=u1[i,k]
                    T.dual_copy(o1,O1[j,:,:])
                    T.copy(C[j,:,:],c,transpose=True);T.copy(c,cc)
                    T.gemm(aa,cc,p2,transpose_B=True,clear_accum=True)
                    T.dual_copy(p2,u2)
                    with T.SimtVF(threads=256):
                        for i,k in T.Parallel(R,N2):o2[i,k]=u2[i,k]
                    T.dual_copy(o2,O2[j,:,:])
                    if save_full:T.dual_copy(u2,Full[j,:,:])
    return tilelang.compile(main,target='ascend',out_idx=[4,5,6])
