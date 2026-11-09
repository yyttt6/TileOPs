"""Independent inverse rows avoid inter-thread pivot barriers."""
import tilelang.ascend.language as T

def inverse(J,dtype,C=32):
    assert C==32
    grid=min(64,J)
    @T.prim_func
    def main(Gram:T.Tensor((J,C,C),'float32'),QK:T.Tensor((J,C,C),'float32'),
             Beta:T.Tensor((J,C),'float32'),AB:T.Tensor((J,C,C),dtype),Attention:T.Tensor((J,C,C),dtype)):
        with T.Kernel(grid) as bx:
            gram=T.alloc_shared((C,C),'float32');beta=T.alloc_shared((C,),'float32')
            for step in T.serial((J+grid-1)//grid):
                job=step*grid+bx
                if job<J:
                    T.copy(Gram[job,:,:],gram);T.copy(Beta[job,:],beta)
                    # Every inverse row owns one full warp; its pivot
                    # coefficient is broadcast without shared-memory barriers.
                    with T.SimtVF(threads=1024):
                        tid=T.get_thread_binding();i=tid//32;j=tid%32
                        value=T.alloc_local((1,),'float32')
                        value[0]=T.if_then_else(i==j,T.float32(1),T.if_then_else(i>j,-beta[i]*gram[i,j],T.float32(0)))
                        for pivot in T.serial(C-1):
                            u=C-1-pivot
                            coefficient=T.call_extern('float32','asc_shfl',value[0],u,32)*beta[u]
                            value[0]=T.if_then_else(i>u and j<u,value[0]-coefficient*gram[u,j],value[0])
                        AB[job,i,j]=value[0]*beta[j]
                        Attention[job,i,j]=T.if_then_else(j<=i,QK[job,i,j],T.float32(0))
    return main
