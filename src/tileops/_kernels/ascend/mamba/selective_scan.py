"""FP32 selective recurrence, including packed restarts and carried state."""
import functools
from typing import ClassVar
import torch
import tilelang
import tilelang.ascend.language as T
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.mamba.call_spec import SelectiveScanFwdInterface
from .selective_scan_decode import build_decode

@functools.lru_cache(maxsize=128)
def build(batch,dim,seqlen,n,groups,dtype,has_skip,has_gate,has_bias,has_seq,has_init,softplus):
    threads=128
    total=batch*dim
    blocks=min(64,(total+threads-1)//threads)
    @T.prim_func
    def main(U:T.Tensor((batch,dim,seqlen),dtype),
             Delta:T.Tensor((batch,dim,seqlen),dtype),
             A:T.Tensor((dim,n),'float32'),
             B:T.Tensor((batch,groups,n,seqlen),dtype),
             C:T.Tensor((batch,groups,n,seqlen),dtype),
             D:T.Tensor((dim,),'float32'),
             Z:T.Tensor((batch,dim,seqlen),dtype),
             Bias:T.Tensor((dim,),'float32'),
             Seq:T.Tensor((batch,seqlen),'int32'),
             Initial:T.Tensor((batch,dim,n),'float32'),
             Y:T.Tensor((batch,dim,seqlen),'float32'),
             Final:T.Tensor((batch,dim,n),'float32')):
        with T.Kernel(blocks) as bx:
            with T.SimtVF(threads=threads):
                state=T.alloc_local((n,),'float32')
                for step in T.serial(T.ceildiv(total,blocks*threads)):
                    for lane in T.Parallel(threads):
                        row=(step*blocks+bx)*threads+lane
                        if row<total:
                            bi=row//dim
                            channel=row%dim
                            group=channel//(dim//groups)
                            for k in T.serial(n):
                                state[k]=T.float32(0)
                                if has_init:
                                    state[k]=Initial[bi,channel,k]
                            for t in T.serial(seqlen):
                                if has_seq:
                                    if t>0:
                                        if Seq[bi,t]!=Seq[bi,t-1]:
                                            for k in T.serial(n):
                                                state[k]=T.float32(0)
                                dt=T.alloc_var('float32')
                                dt=T.cast(Delta[bi,channel,t],'float32')
                                if has_bias:
                                    dt=dt+Bias[channel]
                                if softplus:
                                    if dt<=T.float32(20):
                                        ev=T.exp(dt)
                                        if dt<T.float32(-4):
                                            dt=ev*(T.float32(1)+ev*(T.float32(-.5)+ev*(T.float32(1.0/3.0)+ev*T.float32(-.25))))
                                        else:
                                            dt=T.log(T.float32(1)+ev)
                                uf=T.cast(U[bi,channel,t],'float32')
                                drive=dt*uf
                                acc=T.alloc_var('float32')
                                acc=T.float32(0)
                                for k in T.serial(n):
                                    state[k]=T.exp(dt*A[channel,k])*state[k]+drive*T.cast(B[bi,group,k,t],'float32')
                                    acc=acc+state[k]*T.cast(C[bi,group,k,t],'float32')
                                if has_skip:
                                    acc=acc+uf*D[channel]
                                if has_gate:
                                    zz=T.cast(Z[bi,channel,t],'float32')
                                    acc=acc*(zz/(T.float32(1)+T.exp(-zz)))
                                Y[bi,channel,t]=acc
                            for k in T.serial(n):
                                Final[bi,channel,k]=state[k]
    return tilelang.compile(main,target='ascend',out_idx=[10,11])

class AscendSelectiveScanKernel(Kernel,SelectiveScanFwdInterface):
    devices:ClassVar[frozenset[str]]=frozenset({'npu'})
    @classmethod
    def entry_for(cls,call):
        return call,lambda:cls(call)
    def __init__(self,call):
        super().__init__()
        self.call=call
    def forward(self,u,delta,A,B_in,C_in,D_in=None,z=None,delta_bias=None,seq_idx=None,initial_states=None):
        c=self.call;b=u.shape[0]
        builder=build_decode if c.seqlen==1 and c.dstate==16 and b*c.dim>=8192 else build
        self.compiled=builder(b,c.dim,c.seqlen,c.dstate,c.groups,str(c.dtype).removeprefix('torch.'),c.has_skip,c.has_gate,c.has_bias,c.has_seq_idx,c.has_init,c.softplus)
        per_channel=torch.empty((c.dim,),device=u.device,dtype=torch.float32) if D_in is None or delta_bias is None else D_in
        seq=seq_idx if seq_idx is not None else torch.empty((b,c.seqlen),device=u.device,dtype=torch.int32)
        initial=initial_states if initial_states is not None else torch.empty((b,c.dim,c.dstate),device=u.device,dtype=torch.float32)
        y,last=self.compiled(u,delta,A,B_in,C_in,D_in if D_in is not None else per_channel,z if z is not None else u,delta_bias if delta_bias is not None else per_channel,seq,initial)
        return y.to(c.dtype),last
