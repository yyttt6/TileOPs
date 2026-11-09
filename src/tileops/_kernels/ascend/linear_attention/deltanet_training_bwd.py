"""DeltaNet training backward using chunk contractions and reverse affine scan."""
import functools
from typing import ClassVar
import torch
import torch.nn.functional as F
import tilelang
import tilelang.ascend.language as T
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.linear_attention import DeltaNetChunkBwdInterface
from .deltanet_chunked import matmul
from .deltanet_scan import build_scan
from .deltanet_group_scan import build_group_scan


def after_states(BH,N,K,V,dtype):
    J=BH*N;grid=min(64,J)
    @T.prim_func
    def main(Reverse:T.Tensor((J,K,V),'float32'),After:T.Tensor((J,K,V),dtype)):
        with T.Kernel(grid) as bx:
            x=T.alloc_shared((K,V),'float32');y=T.alloc_shared((K,V),dtype)
            for step in T.serial((J+grid-1)//grid):
                j=step*grid+bx
                if j<J:
                    n=j%N;b=j//N
                    if n+1<N:T.copy(Reverse[b*N+N-2-n,:,:],x)
                    with T.SimtVF(threads=256):
                        for i,k in T.Parallel(K,V):y[i,k]=x[i,k] if n+1<N else T.float32(0)
                    T.copy(y,After[j,:,:])
    return tilelang.compile(main,target='ascend',out_idx=[1])


def reverse_inputs(BH,N,K,V,dtype):
    J=BH*N;grid=min(64,J)
    @T.prim_func
    def main(Product:T.Tensor((J,K,K),'float32'),Local:T.Tensor((J,K,V),'float32'),
             MR:T.Tensor((J,K,K),dtype),UR:T.Tensor((J,K,V),dtype)):
        with T.Kernel(grid) as bx:
            product=T.alloc_shared((K,K),'float32');local=T.alloc_shared((K,V),'float32')
            mr=T.alloc_shared((K,K),dtype);ur=T.alloc_shared((K,V),dtype)
            for step in T.serial((J+grid-1)//grid):
                j=step*grid+bx
                if j<J:
                    r=j//N*N+N-1-j%N
                    T.copy(Product[r,:,:],product);T.copy(Local[r,:,:],local)
                    with T.SimtVF(threads=256):
                        for i,k in T.Parallel(K,K):
                            mr[i,k]=(T.float32(1) if i==k else T.float32(0))-product[k,i]
                        for i,v in T.Parallel(K,V):ur[i,v]=local[i,v]
                    T.copy(mr,MR[j,:,:]);T.copy(ur,UR[j,:,:])
    return tilelang.compile(main,target='ascend',out_idx=[2,3])


class Backward:
    def __init__(self,B,H,L,K,V,C,output_dtype):
        self.shape=B,H,L,K,V,C;self.N=N=L//C;self.J=B*H*N
        self.C=(C+31)//32*32;self.K=(K+31)//32*32;self.V=(V+31)//32*32
        self.dtype='float16' if output_dtype=='float32' else output_dtype
        self.kernels={}
        self.group=build_group_scan(B*H,N,1,self.K,self.V,self.dtype,False,G=4) if N>=64 else None
        self.scan=build_scan(B*H,N,1,self.K,self.V,self.dtype) if self.group is None and N>1 else None
        self.reverse=reverse_inputs(B*H,N,self.K,self.V,self.dtype) if N>1 else None
        self.after=after_states(B*H,N,self.K,self.V,self.dtype)
    def mm(self,a,b,ta=False,tb=False):
        M=a.shape[2] if ta else a.shape[1];D=a.shape[1] if ta else a.shape[2]
        N=b.shape[1] if tb else b.shape[2]
        key=M,N,D,ta,tb
        if key not in self.kernels:self.kernels[key]=matmul(self.J,M,N,D,self.dtype,ta=ta,tb=tb)
        return self.kernels[key](a,b,0)
    def __call__(self,do,q,k,v,beta,S,Aw,Au,w,u):
        B,H,L,K,V,C=self.shape;N=self.N;J=self.J;dt=getattr(torch,self.dtype);native=lambda x:x.to(dt)
        def pad(x, widths):
            return F.pad(x, widths) if any(widths) else x
        def vectors(x,width,padded):return pad(x.reshape(J,C,width),(0,padded-width,0,self.C-C)).to(dt)
        qp,kp,wp=(vectors(x,K,self.K) for x in (q,k,w));vp,up,dop=(vectors(x,V,self.V) for x in (v,u,do))
        bp=pad(beta.reshape(J,C).float(),(0,self.C-C))
        sp=pad(S[:,:,:N].contiguous().reshape(J,K,V),(0,self.V-V,0,self.K-K)).to(dt)
        aw,au=(pad(x.reshape(J,C,C),(0,self.C-C,0,self.C-C)).to(dt) for x in (Aw,Au))
        att=native(torch.tril(self.mm(qp,kp,tb=True)))
        residual=native(up.float()-self.mm(wp,sp))
        qe=native(qp.float()-self.mm(att,wp))
        product=self.mm(kp,wp,ta=True)
        local=self.mm(qe,dop,ta=True)
        # Fuse identity-minus-product, cast, reverse and transpose; scan overwrites full.
        full=local
        if self.reverse is not None:
            mr,ur=self.reverse(product,local)
        if self.group is not None:_,full=self.group(mr,ur)
        elif self.scan is not None:
            stride=1
            while stride<N:mr,ur,full=self.scan(mr,ur,stride);stride*=2
        after=self.after(full)
        da=native(torch.tril(self.mm(dop,residual,tb=True)))
        dq=self.mm(dop,sp,tb=True)+self.mm(da,kp)
        dk=self.mm(da,qp,ta=True)+self.mm(residual,after,tb=True)
        dr=native(self.mm(att,dop,ta=True)+self.mm(kp,after))
        dw=native(-self.mm(dr,sp,tb=True))
        dbk=self.mm(aw,dw,ta=True);dbv=self.mm(au,dr,ta=True)
        dl=-torch.tril(self.mm(native(dbk),wp,tb=True)+self.mm(native(dbv),up,tb=True),diagonal=-1)
        dbeta=(dbk*kp.float()).sum(-1)+(dbv*vp.float()).sum(-1)+(dl*self.mm(kp,kp,tb=True)).sum(-1)
        dg=dl*bp[:,:,None]
        dk=dk+dbk*bp[:,:,None]+self.mm(native(dg+dg.transpose(-1,-2)),kp)
        dv=dbv*bp[:,:,None]
        return (dq[:,:C,:K].to(q.dtype).contiguous().reshape(B,H,L,K),
                dk[:,:C,:K].to(q.dtype).contiguous().reshape(B,H,L,K),
                dv[:,:C,:V].to(q.dtype).contiguous().reshape(B,H,L,V),
                dbeta[:,:C].to(q.dtype).contiguous().reshape(B,H,L))
    def get_kernel_source(self):
        ks=list(self.kernels.values())+[x for x in (self.scan,self.group,self.after,self.reverse) if x is not None]
        return '\n'.join(x.get_kernel_source() for x in ks)

build=functools.lru_cache(maxsize=64)(Backward)
class AscendDeltaNetChunkBwdKernel(Kernel,DeltaNetChunkBwdInterface):
    devices:ClassVar[frozenset[str]]=frozenset({'npu'})
    @classmethod
    def entry_for(cls,call):return call,lambda:cls(call)
    def __init__(self,call):
        super().__init__();self.call=call
        self.compiled=build(call.batch,call.heads,call.seq_len,call.dim_k,call.dim_v,call.chunk_size,str(call.dtype).removeprefix('torch.'))
    def forward(self,*xs):return self.compiled(*(x.contiguous() for x in xs))

from tileops.backend._registry import register_kernel_type
register_kernel_type('DeltaNetChunkBwdOp','ascend',AscendDeltaNetChunkBwdKernel)
