"""DMA readout epilogue, including full padded rows before cropping."""
import tilelang.ascend.language as T

def output(B,S,H,V,dtype,C=32):
    N=(S+C-1)//C;J=B*N*H;grid=min(64,J)
    @T.prim_func
    def main(Inter:T.Tensor((J,C,V),'float32'),Bias:T.Tensor((J,C,V),'float32'),O:T.Tensor((B,N*C,H,V),dtype)):
        with T.Kernel(grid) as bx:
            x=T.alloc_shared((C,V),'float32');y=T.alloc_shared((C,V),'float32');z=T.alloc_shared((C,V),dtype)
            for step in T.serial((J+grid-1)//grid):
                j=step*grid+bx
                if j<J:
                    b=j//(N*H);n=j//H%N;h=j%H
                    T.copy(Inter[j,:,:],x);T.copy(Bias[j,:,:],y)
                    with T.SimtVF(threads=256):
                        for t,i in T.Parallel(C,V):z[t,i]=x[t,i]+y[t,i]
                    T.copy(z,O[b,n*C:(n+1)*C,h,:])
    return main
