"""Chunk delta rule with Cube epilogues and hierarchical affine prefixes."""
import functools
import tilelang
import tilelang.ascend.language as T
from .deltanet_scan import build_scan
from .deltanet_pack import pack as dma_pack
from .deltanet_pairs import build as pair_matmul
from .deltanet_inverse import inverse as row_inverse
from .deltanet_group_scan import build_group_scan
from .deltanet_output import output as dma_output
from .deltanet_seeded_states import build as seeded_states


@functools.lru_cache(maxsize=128)
def matmul(J,M,N,D,dtype,ta=False,tb=False,heads=1,chunks=1,right_batches=0):
    RB=right_batches or J
    ashape=(J,D,M) if ta else (J,M,D)
    bshape=(RB,N,D) if tb else (RB,D,N)
    grid=min(32,J)
    @T.prim_func
    def main(A:T.Tensor(ashape,dtype),B:T.Tensor(bshape,dtype),shift:T.int32,
             O:T.Tensor((J,M,N),'float32')):
        with T.Kernel(grid) as bx:
            a=T.alloc_l1((M,D),dtype);b=T.alloc_l1((N,D),dtype)
            aa=T.alloc_l0a((M,D),dtype);bb=T.alloc_l0b((N,D),dtype)
            cc=T.alloc_l0c((M,N),'float32')
            for step in T.Pipelined((J+grid-1)//grid,num_stages=2):
                j=step*grid+bx
                if j<J:
                    r=j//(chunks*heads)*heads+j%heads if right_batches else j-T.min(j//heads%chunks,shift)*heads
                    if ta:T.copy(A[j,:,:],a,transpose=True)
                    else:T.copy(A[j,:,:],a)
                    if tb:T.copy(B[r,:,:],b)
                    else:T.copy(B[r,:,:],b,transpose=True)
                    T.copy(a,aa);T.copy(b,bb)
                    T.gemm(aa,bb,cc,transpose_B=True,clear_accum=True)
                    T.copy(cc,O[j,:,:])
    return tilelang.compile(main,target='ascend',out_idx=[3])


def state_inputs(B,N,H,K,V,dtype,initial):
    J=B*N*H;grid=min(64,J)
    @T.prim_func
    def main(U:T.Tensor((J,K,V),'float32'),MH:T.Tensor((J,K,V),'float32'),
             H0:T.Tensor((B,H,K,V),'float32'),HB:T.Tensor((J,K,V),dtype),HT:T.Tensor((B,H,K,V),'float32')):
        with T.Kernel(grid) as bx:
            u=T.alloc_shared((K,V),'float32');mh=T.alloc_shared((K,V),'float32')
            result=T.alloc_shared((K,V),dtype)
            for step in T.serial((J+grid-1)//grid):
                job=step*grid+bx
                if job<J:
                    b=job//(N*H);n=job//H%N;h=job%H
                    if n>0:
                        T.copy(U[job-H,:,:],u)
                        if initial:T.copy(MH[job-H,:,:],mh)
                    elif initial:T.copy(H0[b,h,:,:],u)
                    with T.SimtVF(threads=256):
                        for i,j in T.Parallel(K,V):
                            result[i,j]=T.if_then_else(n>0,u[i,j]+(mh[i,j] if initial else T.float32(0)),u[i,j] if initial else T.float32(0))
                    T.copy(result,HB[job,:,:])
                    if n==N-1:
                        T.copy(U[job,:,:],u)
                        if initial:T.copy(MH[job,:,:],mh)
                        with T.SimtVF(threads=256):
                            for i,j in T.Parallel(K,V):u[i,j]=u[i,j]+(mh[i,j] if initial else T.float32(0))
                        T.copy(u,HT[b,h,:,:])
    return main


class ChunkedDelta:
    def __init__(self,B,S,H,K,V,dtype,scale,initial,norm):
        self.length=S;self.N=N=(S+31)//32;self.initial=initial;J=B*N*H;C=32
        self.J=J;self.K=K;self.V=V
        compile=lambda f,idx:tilelang.compile(f,target='ascend',out_idx=idx)
        self.pack=compile(dma_pack(B,S,H,K,V,dtype,scale,initial,norm),[5,6,7,8,9])
        self.gram=matmul(J,C,C,K,dtype,tb=True)
        self.inverse=compile(row_inverse(J,dtype),[3,4])
        self.wu=pair_matmul(C,K,V,C,dtype)
        self.affine=pair_matmul(K,K,V,C,dtype,ta=True,first_mode=1,save_full=N==1)
        self.qbias=pair_matmul(C,K,V,C,dtype,first_mode=2,second_fp32=True)
        group=4
        self.group_scan=build_group_scan(B,N,H,K,V,dtype,initial,G=group) if N>=64 and group else None
        self.scan=build_scan(B,N,H,K,V,dtype) if self.group_scan is None else None
        self.states=seeded_states(B,N,H,K,V,dtype) if initial else compile(state_inputs(B,N,H,K,V,dtype,False),[3,4])
        self.readout=matmul(J,C,V,K,dtype)
        self.output=compile(dma_output(B,S,H,V,dtype),[2])

    def __call__(self,q,k,v,beta,h0):
        qp,kp,vp,bp,hp=self.pack(q,k,v,beta,h0)
        gram=self.gram(kp,kp,0);qk=self.gram(qp,kp,0)
        ab,attention=self.inverse(gram,qk,bp)
        w,u,_=self.wu(ab,kp,vp,kp)
        m,state,raw_u=self.affine(kp,w,u,w)
        qe,bias,_=self.qbias(attention,w,u,qp)
        full_state=raw_u
        if self.group_scan is not None:
            m,full_state=self.group_scan(m,state)
        else:
            stride=1
            while stride<self.N:
                m,state,full_state=self.scan(m,state,stride)
                stride*=2
        before,final=self.states(m,full_state,hp) if self.initial else self.states(full_state,full_state,h0)
        result=self.output(self.readout(qe,before,0),bias)
        return result[:,:self.length].contiguous(),final

    def get_kernel_source(self):
        kernels={id(value):value for value in vars(self).values() if hasattr(value,'get_kernel_source')}
        return '\n'.join(value.get_kernel_source() for value in kernels.values())


build_chunked=functools.lru_cache(maxsize=64)(ChunkedDelta)
