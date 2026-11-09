"""In-place Q/K head RMSNorm and RoPE; packed V heads are never written."""

# Registration is optional when the Ascend toolchain is unavailable.
try:
    import tilelang.ascend.language as T
except ImportError:
    pass
else:
    import functools
    from typing import ClassVar
    import tilelang
    from tileops.backend import register_kernel_type
    from tileops.kernels.kernel_base import Kernel
    from tileops.kernels.norm.fused_extended_call_spec import QKNormRoPEInterface

    @functools.lru_cache(maxsize=128)
    def build(tokens,heads,kv_heads,dim,cache_rows,rotary,dtype,cache_dtype,eps,layout):
        active = heads + kv_heads
        cache_pitch = max(16, (rotary + 15) // 16 * 16)
        if (dim % 128 == 0 and dtype in ('float16', 'bfloat16')
                and active * dim * 12 + dim * 4 + cache_pitch * 8 < 192 * 1024):
            from ._qk_fused_simd import build as build_simd
            return build_simd(tokens, heads, kv_heads, dim, cache_rows,
                              rotary, dtype, cache_dtype, eps, layout)
        width=tilelang.next_power_of_2(dim);pitch=max(16,(dim+15)//16*16)
        cache_pitch=max(16,(rotary+15)//16*16)
        active=heads+kv_heads;packed=(heads+2*kv_heads)*dim
        jobs=tokens*active;blocks=min(64,jobs);half=rotary//2
        assert (3*pitch*2+pitch*4+cache_pitch*4)<208*1024
        @T.prim_func
        def main(QKV:T.Tensor((tokens,packed),dtype),QW:T.Tensor((dim,),dtype),KW:T.Tensor((dim,),dtype),Cache:T.Tensor((cache_rows,rotary),cache_dtype),Positions:T.Tensor((tokens,),'int64')):
            with T.Kernel(blocks) as bx:
                x=T.alloc_shared((pitch,),dtype);w=T.alloc_shared((pitch,),dtype);y=T.alloc_shared((pitch,),dtype)
                normalized=T.alloc_shared((pitch,),'float32')
                cache=T.alloc_shared((cache_pitch,),cache_dtype)
                for step in T.serial((jobs+blocks-1)//blocks):
                    job=step*blocks+bx
                    if job<jobs:
                        token=job//active;head=job%active
                        T.copy(QKV[token,head*dim:(head+1)*dim],x[:dim])
                        if head<heads:T.copy(QW[:dim],w[:dim])
                        else:T.copy(KW[:dim],w[:dim])
                        # Keep the indirect GM load inside SIMT; no scalar UB dereference.
                        with T.SimtVF(threads=32):
                            for i in T.Parallel(cache_pitch):
                                if i<rotary:cache[i]=Cache[Positions[token],i]
                        with T.SimtVF(threads=min(256,width)):
                            total=T.alloc_reducer((1,),'float32',op='sum');T.reducer_init(total)
                            for i in T.Parallel(width):
                                if i<dim:
                                    value=T.cast(x[i],'float32')
                                    T.reducer_update(total[0],value*value)
                            result=T.alloc_fragment((1,),'float32');T.finalize_reducer(total,result)
                            result[0]=T.rsqrt(result[0]/dim+eps)
                            for i in T.Parallel(width):
                                if i<dim:normalized[i]=T.cast(x[i],'float32')*(result[0]*T.cast(w[i],'float32'))
                        with T.SimtVF(threads=min(256,width)):
                            for i in T.Parallel(width):
                                if i<dim:
                                    if i<rotary:
                                        if layout=='neox':
                                            index=i%half
                                            peer=T.if_then_else(i<half,i+half,i-half)
                                            sign=T.if_then_else(i<half,T.float32(-1),T.float32(1))
                                        else:
                                            index=i//2
                                            peer=T.if_then_else(i%2==0,i+1,i-1)
                                            sign=T.if_then_else(i%2==0,T.float32(-1),T.float32(1))
                                        y[i]=normalized[i]*T.cast(cache[index],'float32')+sign*normalized[peer]*T.cast(cache[index+half],'float32')
                                    else:y[i]=normalized[i]
                        T.copy(y[:dim],QKV[token,head*dim:(head+1)*dim])
        return tilelang.compile(main,target='ascend',out_idx=[])

    class AscendQKNormRoPEKernel(Kernel,QKNormRoPEInterface):
        devices:ClassVar[frozenset[str]]=frozenset({'npu'})
        @classmethod
        def entry_for(cls,call):return call,lambda:cls(call)
        def __init__(self,call):super().__init__();self.call=call
        def forward(self,qkv,q_weight,k_weight,cos_sin_cache,positions):
            c=self.call
            if not c.tokens:return None
            self.compiled=build(c.tokens,c.heads,c.kv_heads,c.dim,c.cache_rows,c.rotary_dim,str(c.dtype).split('.')[-1],str(c.cache_dtype).split('.')[-1],c.eps,c.layout)
            self.compiled(qkv,q_weight,k_weight,cos_sin_cache,positions)
    register_kernel_type('FusedQKNormRoPEFwdOp','ascend',AscendQKNormRoPEKernel)
