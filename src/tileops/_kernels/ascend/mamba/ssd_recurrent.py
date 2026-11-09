"""FP32 recurrent state update and projection, preserving caller-owned state."""
import functools
from typing import ClassVar
import torch
import tilelang
import tilelang.ascend.language as T
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.mamba.call_spec import SSDDecodeFwdInterface
from tileops.backend._registry import register_kernel_type

@functools.lru_cache(maxsize=128)
def build(b, h, p, n, g, dtype):
    width = min(1024, max(64, 1 << (n-1).bit_length()))
    rows = min(32, 4096 // width)
    dma = n == width and n % 8 == 0 and p % rows == 0
    total = b*h*p
    tiles = (total+rows-1)//rows
    blocks = min(64, tiles)
    @T.prim_func
    def main(A:T.Tensor((h*p,n),'float32'), DT:T.Tensor((total,),'float32'),
             X:T.Tensor((total,),dtype), B:T.Tensor((b,g,n),dtype),
             C:T.Tensor((b,g,n),dtype), State:T.Tensor((total,n),'float32'),
             Y:T.Tensor((total,),'float32')):
        with T.Kernel(blocks) as bx:
            decay=T.alloc_shared((rows,width),'float32')
            old=T.alloc_shared((rows,width),'float32')
            update=T.alloc_shared((rows,width),'float32')
            projection=T.alloc_shared((rows,width),'float32')
            work=T.alloc_shared((rows,width),'float32')
            reduced=T.alloc_shared((rows,8),'float32')
            accumulated=T.alloc_shared((rows,),'float32')
            for step in T.serial(T.ceildiv(tiles,blocks)):
                tile=step*blocks+bx
                if tile<tiles:
                    T.fill(accumulated,0)
                    for part in T.serial(T.ceildiv(n,width)):
                        if dma:
                            T.copy(A[(tile*rows)%(h*p):(tile*rows)%(h*p)+rows, :],decay)
                            T.copy(State[tile*rows:tile*rows+rows, :],old)
                        with T.SimtVF(threads=256):
                            for r,k in T.Parallel(rows,width):
                                row=tile*rows+r
                                col=part*width+k
                                if not dma:
                                    decay[r,k]=0
                                    old[r,k]=0
                                update[r,k]=0
                                projection[r,k]=0
                                if row<total and col<n:
                                    batch=row//(h*p)
                                    group=(row//p%h)//(h//g)
                                    if dma:
                                        decay[r,k]=DT[row]*decay[r,k]
                                    else:
                                        decay[r,k]=DT[row]*A[row%(h*p),col]
                                        old[r,k]=State[row,col]
                                    update[r,k]=(DT[row]*T.cast(X[row],'float32'))*T.cast(B[batch,group,col],'float32')
                                    projection[r,k]=T.cast(C[batch,group,col],'float32')
                        with T.SimdVF():
                            mask=T.simd.pset(32)
                            for r in T.serial(rows):
                                for j in T.serial(width//64):
                                    dv=T.simd.vexp(T.simd.vld(decay[r,j*64]),mask,precision='ftz_false')
                                    sv=T.simd.vmul(dv,T.simd.vld(old[r,j*64]),mask)
                                    sv=T.simd.vadd(sv,T.simd.vld(update[r,j*64]),mask)
                                    T.simd.vsts(old[r,j*64],sv,mask,extent=64)
                                    value=T.simd.vmul(sv,T.simd.vld(projection[r,j*64]),mask)
                                    T.simd.vsts(work[r,j*64],value,mask,extent=64)
                            T.simd.mem_bar('VST_VLD')
                            for r in T.serial(rows):
                                acc=T.simd.alloc_local(1,'float32')
                                acc[0]=T.simd.vdup(T.float32(0),'float32')
                                for j in T.serial(width//64):
                                    acc[0]=T.simd.vadd(acc[0],T.simd.vld(work[r,j*64]),mask)
                                value=T.simd.vcadd(acc[0],mask)
                                one=T.simd.pset(32,'PAT_VL1')
                                T.simd.vsts(reduced[r,0],value,one,extent=1)
                        if dma:
                            T.copy(old,State[tile*rows:tile*rows+rows, :])
                        with T.SimtVF(threads=256):
                            if not dma:
                                for r,k in T.Parallel(rows,width):
                                    row=tile*rows+r
                                    col=part*width+k
                                    if row<total and col<n:
                                        State[row,col]=old[r,k]
                            for r in T.Parallel(rows):
                                accumulated[r]=accumulated[r]+reduced[r,0]
                    with T.SimtVF(threads=256):
                        for r in T.Parallel(rows):
                            if tile*rows+r<total:
                                Y[tile*rows+r]=accumulated[r]
    return tilelang.compile(main,target='ascend',out_idx=[6])

class AscendSSDRecurrentKernel(Kernel,SSDDecodeFwdInterface):
    devices:ClassVar[frozenset[str]]=frozenset({'npu'})
    @classmethod
    def entry_for(cls,call):
        return call,lambda:cls(call)
    def __init__(self,call):
        super().__init__()
        self.call=call
        self.compiled=build(call.batch,call.n_heads,call.d_head,call.d_state,call.n_groups,str(call.dtype).removeprefix('torch.'))
    def forward(self,A,dt,x,B_in,C_in,state):
        c=self.call
        return self.compiled(A.reshape(c.n_heads*c.d_head,c.d_state),dt.reshape(-1),x.reshape(-1),B_in,C_in,state.view(-1,c.d_state)).view(c.batch,c.n_heads,c.d_head)

register_kernel_type('SSDRecurrentFwdOp','ascend',AscendSSDRecurrentKernel)
