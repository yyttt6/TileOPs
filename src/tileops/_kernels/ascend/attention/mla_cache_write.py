"""Dynamic MLA scatter with explicit input-dtype rounding before FP8 conversion."""
import functools
from typing import ClassVar
import torch
import tilelang
import tilelang.ascend.language as T
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.attention.call_spec import MLAKVCacheWriteFwdInterface


@functools.lru_cache(maxsize=64)
def build_plain_batch(dc, pe, physical_rows, dtype):
    tokens=T.dynamic('tokens');width=dc+pe;batch=8;stride=(width+31)//32*32
    @T.prim_func
    def main(KC:T.Tensor((tokens,dc),dtype), PE:T.Tensor((tokens,pe),dtype),
             Cache:T.Tensor((physical_rows,width),dtype),Slots:T.Tensor((tokens,),'int64')):
        T.func_attr({'tilelang_callee_allocated_outputs':0})
        with T.Kernel(64) as bx:
            raw=T.alloc_shared((batch,stride),dtype)
            slots=T.alloc_shared((batch,),'int64')
            for step in T.serial(T.ceildiv(tokens,64*batch)):
                start=(step*64+bx)*batch
                if start<tokens:
                    count=T.min(batch,tokens-start)
                    T.copy(Slots[start:start+count],slots[:count])
                    if dc>0:T.copy(KC[start:start+count,:],raw[:count,:dc])
                    if pe>0:T.copy(PE[start:start+count,:],raw[:count,dc:width])
                    for row in T.serial(count):
                        slot=slots[row]
                        if slot>=0:T.copy(raw[row,:width],Cache[slot,:])
    return tilelang.compile(main,target='ascend',out_idx=[])


@functools.lru_cache(maxsize=64)
def build(dc, pe, physical_rows, dtype, quantized, fuse_rope, layout, table_dtype, max_position):
    if not quantized and (not fuse_rope or pe==0) and dc%16==0 and pe%16==0:
        return build_plain_batch(dc,pe,physical_rows,dtype)
    tokens = T.dynamic('tokens')
    width = dc + pe
    tile = (width + 63) // 64 * 64
    petile = max(64, (pe + 63) // 64 * 64)
    half = max(1, pe//2)
    cache_dtype = 'uint8' if quantized else dtype
    scale_dtype = 'float32' if quantized else 'int64'
    tdtype = table_dtype if fuse_rope else dtype
    tshape = (max_position, pe) if fuse_rope else (tokens, dc)
    @T.prim_func
    def main(KC:T.Tensor((tokens,dc),dtype), PE:T.Tensor((tokens,pe),dtype),
             Cache:T.Tensor((physical_rows,width),cache_dtype), Slots:T.Tensor((tokens,),'int64'),
             Scale:T.Tensor((1,),scale_dtype), Positions:T.Tensor((tokens,),'int64'),
             Table:T.Tensor(tshape,tdtype)):
        T.func_attr({'tilelang_callee_allocated_outputs':0})
        with T.Kernel(64) as bx:
            raw=T.alloc_shared((tile,),dtype)
            latent=T.alloc_shared((max(64,(dc+63)//64*64),),dtype)
            rope=T.alloc_shared((petile,),dtype)
            table=T.alloc_shared((petile,),tdtype)
            encoded=T.alloc_shared((tile,),'uint8')
            scale=T.alloc_shared((1,),scale_dtype)
            if quantized:
                T.copy(Scale,scale)
            for step in T.serial(T.ceildiv(tokens,64)):
                token=step*64+bx
                if token<tokens:
                    slot=Slots[token]
                    if slot>=0:
                        if dc>0:T.copy(KC[token,:],latent[:dc])
                        if pe>0:T.copy(PE[token,:],rope[:pe])
                        if fuse_rope and pe>0:
                            position=Positions[token]
                            T.copy(Table[position,:],table[:pe])
                        with T.SimtVF(threads=256):
                            for j in T.Parallel(dc):
                                raw[j]=latent[j]
                            for i in T.Parallel(pe):
                                if fuse_rope:
                                    if layout=='neox':
                                        pair=i%half
                                        a=T.float32(rope[pair]);b=T.float32(rope[pair+half])
                                        c=T.float32(table[pair]);s=T.float32(table[pair+half])
                                        raw[dc+i]=T.cast(T.if_then_else(i<half,a*c-b*s,b*c+a*s),dtype)
                                    else:
                                        pair=i//2
                                        a=T.float32(rope[2*pair]);b=T.float32(rope[2*pair+1])
                                        c=T.float32(table[pair]);s=T.float32(table[pair+half])
                                        raw[dc+i]=T.cast(T.if_then_else(i%2==0,a*c-b*s,b*c+a*s),dtype)
                                else:
                                    raw[dc+i]=rope[i]
                        if quantized:
                            with T.SimdVF():
                                full=T.simd.pset(32)
                                divisor=T.simd.vdup(scale[0],'float32')
                                lo=T.simd.vdup(T.float32(-448),'float32');hi=T.simd.vdup(T.float32(448),'float32')
                                for part in T.serial(tile//64):
                                    x=T.simd.vcvt(T.simd.vld(raw[part*64],dist='UNPK_B16'),'float32',part=0)
                                    x=T.simd.vdiv(x,divisor,full,precision='exact')
                                    x=T.simd.vmin(T.simd.vmax(x,lo,full),hi,full)
                                    q=T.simd.vcvt(x,'float8_e4m3fn',full,round='ROUND_R',sat=True,part=0)
                                    T.simd.vsts(encoded[part*64],T.reinterpret(q,'uint8x256'),full,'PK4_B32')
                            with T.SimtVF(threads=256):
                                for i in T.Parallel(width):
                                    if T.isnan(T.float32(raw[i])):
                                        sign=(T.reinterpret(T.float32(raw[i]),'uint32')>>24)&T.uint32(128)
                                        encoded[i]=T.cast(sign|T.uint32(127),'uint8')
                            T.copy(encoded[:width],Cache[slot,:])
                        else:
                            T.copy(raw[:width],Cache[slot,:])
    return tilelang.compile(main,target='ascend',out_idx=[])


class AscendMLAKVCacheWriteKernel(Kernel,MLAKVCacheWriteFwdInterface):
    devices:ClassVar[frozenset[str]]=frozenset({'npu'})
    @classmethod
    def entry_for(cls,call):
        return call,lambda:cls(call)
    def __init__(self,call):
        super().__init__();self.call=call
        self.compiled=build(call.latent_dim,call.rope_dim,call.physical_tokens,str(call.dtype).removeprefix('torch.'),call.quantized,call.fuse_rope,call.rope_layout,str(call.cos_sin_dtype).removeprefix('torch.'),call.max_position)
    def forward(self,kv_c,k_pe,kv_cache,slot_mapping,scale=None,positions=None,cos_sin_cache=None):
        if kv_c.shape[0]==0:return None
        cache=kv_cache.view(torch.uint8) if self.call.quantized else kv_cache
        if not self.call.quantized and (not self.call.fuse_rope or self.call.rope_dim==0) and self.call.latent_dim%16==0 and self.call.rope_dim%16==0:
            self.compiled(kv_c,k_pe,cache.reshape(self.call.physical_tokens,self.call.row_width),slot_mapping)
            return None
        self.compiled(kv_c,k_pe,cache.reshape(self.call.physical_tokens,self.call.row_width),slot_mapping,
                      scale if self.call.quantized else slot_mapping[:1],
                      positions if self.call.fuse_rope else slot_mapping,
                      cos_sin_cache if self.call.fuse_rope else kv_c)
        return None


from tileops.backend._registry import register_kernel_type
register_kernel_type('MLAKVCacheWriteFwdOp','ascend',AscendMLAKVCacheWriteKernel)
