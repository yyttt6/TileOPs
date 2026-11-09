"""Dynamic-token paged K/V scatter, including saturated FP8 storage."""
import functools
from typing import ClassVar
import torch
import tilelang
import tilelang.ascend.language as T
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.attention.call_spec import PagedKVCacheWriteFwdInterface


@functools.lru_cache(maxsize=64)
def build(width, physical_rows, dtype, quantized):
    tokens = T.dynamic('tokens')
    tile = min(1024, (width + 63) // 64 * 64)
    columns = (width + tile - 1) // tile
    cache_dtype = 'uint8' if quantized else dtype
    if quantized:
        @T.prim_func
        def main(K:T.Tensor((tokens,width),dtype), V:T.Tensor((tokens,width),dtype),
                 KP:T.Tensor((physical_rows,width),cache_dtype), VP:T.Tensor((physical_rows,width),cache_dtype),
                 Slots:T.Tensor((tokens,),'int64'), KS:T.Tensor((1,),'float32'), VS:T.Tensor((1,),'float32')):
            T.func_attr({'tilelang_callee_allocated_outputs':0})
            with T.Kernel(64) as bx:
                rawk=T.alloc_shared((tile,),dtype);rawv=T.alloc_shared((tile,),dtype)
                qk=T.alloc_shared((tile,),'uint8');qv=T.alloc_shared((tile,),'uint8')
                ks=T.alloc_shared((1,),'float32');vs=T.alloc_shared((1,),'float32')
                T.copy(KS,ks);T.copy(VS,vs)
                for step in T.serial(T.ceildiv(tokens*columns,64)):
                    job=step*64+bx
                    if job<tokens*columns:
                        token=job//columns;col=job%columns*tile;count=T.min(tile,width-col)
                        slot=Slots[token]
                        if slot>=0:
                            T.copy(K[token,col:col+count],rawk[:count]);T.copy(V[token,col:col+count],rawv[:count])
                            with T.SimdVF():
                                full=T.simd.pset(32)
                                sk=T.simd.vdup(ks[0],'float32');sv=T.simd.vdup(vs[0],'float32')
                                lo=T.simd.vdup(T.float32(-448),'float32');hi=T.simd.vdup(T.float32(448),'float32')
                                for part in T.serial(tile//64):
                                    k=T.simd.vcvt(T.simd.vld(rawk[part*64],dist='UNPK_B16'),'float32',part=0)
                                    v=T.simd.vcvt(T.simd.vld(rawv[part*64],dist='UNPK_B16'),'float32',part=0)
                                    k=T.simd.vdiv(k,sk,full,precision='exact');v=T.simd.vdiv(v,sv,full,precision='exact')
                                    k=T.simd.vmin(T.simd.vmax(k,lo,full),hi,full);v=T.simd.vmin(T.simd.vmax(v,lo,full),hi,full)
                                    ek=T.simd.vcvt(k,'float8_e4m3fn',full,round='ROUND_R',sat=True,part=0)
                                    ev=T.simd.vcvt(v,'float8_e4m3fn',full,round='ROUND_R',sat=True,part=0)
                                    T.simd.vsts(qk[part*64],T.reinterpret(ek,'uint8x256'),full,'PK4_B32')
                                    T.simd.vsts(qv[part*64],T.reinterpret(ev,'uint8x256'),full,'PK4_B32')
                            with T.SimtVF(threads=256):
                                for i in T.Parallel(tile):
                                    if i<count:
                                        if T.isnan(T.float32(rawk[i])):
                                            sign=(T.reinterpret(T.float32(rawk[i]),'uint32')>>24)&T.uint32(128)
                                            qk[i]=T.cast(sign|T.uint32(127),'uint8')
                                        if T.isnan(T.float32(rawv[i])):
                                            sign=(T.reinterpret(T.float32(rawv[i]),'uint32')>>24)&T.uint32(128)
                                            qv[i]=T.cast(sign|T.uint32(127),'uint8')
                            T.copy(qk[:count],KP[slot,col:col+count]);T.copy(qv[:count],VP[slot,col:col+count])
    else:
        @T.prim_func
        def main(K:T.Tensor((tokens,width),dtype), V:T.Tensor((tokens,width),dtype),
                 KP:T.Tensor((physical_rows,width),dtype), VP:T.Tensor((physical_rows,width),dtype),
                 Slots:T.Tensor((tokens,),'int64')):
            T.func_attr({'tilelang_callee_allocated_outputs':0})
            with T.Kernel(64) as bx:
                rawk=T.alloc_shared((tile,),dtype);rawv=T.alloc_shared((tile,),dtype)
                for step in T.serial(T.ceildiv(tokens*columns,64)):
                    job=step*64+bx
                    if job<tokens*columns:
                        token=job//columns;col=job%columns*tile;count=T.min(tile,width-col)
                        slot=Slots[token]
                        if slot>=0:
                            T.copy(K[token,col:col+count],rawk[:count]);T.copy(V[token,col:col+count],rawv[:count])
                            T.copy(rawk[:count],KP[slot,col:col+count]);T.copy(rawv[:count],VP[slot,col:col+count])
    return tilelang.compile(main,target='ascend',out_idx=[])


class AscendPagedKVCacheWriteKernel(Kernel,PagedKVCacheWriteFwdInterface):
    devices:ClassVar[frozenset[str]]=frozenset({'npu'})
    @classmethod
    def entry_for(cls,call):
        return call,lambda:cls(call)
    def __init__(self,call):
        super().__init__();self.call=call
        self.compiled=build(call.row_width,call.physical_tokens,str(call.dtype).removeprefix('torch.'),call.quantized)
    def forward(self,k,v,k_pages,v_pages,slot_mapping,k_scale=None,v_scale=None):
        n=k.shape[0]
        if n==0:return None
        width=self.call.row_width;physical=self.call.physical_tokens
        kp=k_pages.view(torch.uint8) if self.call.quantized else k_pages
        vp=v_pages.view(torch.uint8) if self.call.quantized else v_pages
        args=[k.reshape(n,width),v.reshape(n,width),kp.reshape(physical,width),vp.reshape(physical,width),slot_mapping]
        if self.call.quantized:args.extend([k_scale,v_scale])
        self.compiled(*args)
        return None


from tileops.backend._registry import register_kernel_type
register_kernel_type('PagedKVCacheWriteFwdOp','ascend',AscendPagedKVCacheWriteKernel)
