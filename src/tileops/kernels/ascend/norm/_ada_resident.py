"""Candidate-only bounded adaptive normalization with one vector function per row."""
import functools
import tilelang
import tilelang.ascend.language as T
@functools.lru_cache(maxsize=256)
def build(m,n,dtype,eps,gated):
    assert 0<n<=4096
    width=tilelang.next_power_of_2(n)
    pitch=max(16,(n+15)//16*16)
    blocks=min(64,m)
    @T.prim_func
    def main(X:T.Tensor((m,n),dtype),W:T.Tensor((m,n),dtype),B:T.Tensor((m,n),dtype),G:T.Tensor((m,n),dtype),Y:T.Tensor((m,n),dtype)):
        with T.Kernel(blocks) as bx:
            x=T.alloc_shared((pitch,),dtype)
            w=T.alloc_shared((pitch,),dtype)
            b=T.alloc_shared((pitch,),dtype)
            g=T.alloc_shared((pitch,),dtype)
            y=T.alloc_shared((pitch,),dtype)
            for step in T.serial((m+blocks-1)//blocks):
                row=step*blocks+bx
                if row<m:
                    T.copy(X[row,:n],x[:n])
                    T.copy(W[row,:n],w[:n])
                    T.copy(B[row,:n],b[:n])
                    if gated:T.copy(G[row,:n],g[:n])
                    with T.SimtVF(threads=min(256,width)):
                        total=T.alloc_reducer((1,),'float32',op='sum')
                        T.reducer_init(total)
                        for i in T.Parallel(width):
                            if i<n:T.reducer_update(total[0],T.cast(x[i],'float32'))
                        reduced=T.alloc_fragment((1,),'float32')
                        T.finalize_reducer(total,reduced)
                        mean=reduced[0]/n
                        variance=T.alloc_reducer((1,),'float32',op='sum')
                        T.reducer_init(variance)
                        for i in T.Parallel(width):
                            if i<n:
                                centered=T.cast(x[i],'float32')-mean
                                T.reducer_update(variance[0],centered*centered)
                        reduced_var=T.alloc_fragment((1,),'float32')
                        T.finalize_reducer(variance,reduced_var)
                        inverse=T.rsqrt(reduced_var[0]/n+eps)
                        for i in T.Parallel(width):
                            if i<n:
                                value=(T.cast(x[i],'float32')-mean)*inverse
                                modulated=value*T.cast(w[i],'float32')+T.cast(b[i],'float32')
                                if gated:y[i]=modulated*T.cast(g[i],'float32')
                                else:y[i]=modulated
                    T.copy(y[:n],Y[row,:n])
    return tilelang.compile(main,target='ascend',out_idx=[4])
