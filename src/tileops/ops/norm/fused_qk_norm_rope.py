"""In-place headwise RMSNorm and RoPE on packed QKV, preserving V."""
from tileops.ops.op_base import Op
from tileops.kernels.norm.fused_extended_call_spec import QKNormRoPECall,QKNormRoPEInterface

class FusedQKNormRoPEFwdOp(Op):
    kernel_types={}
    interfaces={'qk_norm_rope':QKNormRoPEInterface}
    def __init__(self,num_heads,num_kv_heads,eps=1e-6,rope_layout='neox',*,target=None):
        self.num_heads=num_heads;self.num_kv_heads=num_kv_heads;self.eps=eps;self.rope_layout=rope_layout
        super().__init__(target=target)
    def forward(self,qkv,q_weight,k_weight,cos_sin_cache,positions):
        call=QKNormRoPECall(device=qkv.device,tokens=qkv.shape[0],heads=self.num_heads,kv_heads=self.num_kv_heads,dim=q_weight.numel(),cache_rows=cos_sin_cache.shape[0],rotary_dim=cos_sin_cache.shape[1],dtype=qkv.dtype,cache_dtype=cos_sin_cache.dtype,eps=self.eps,layout=self.rope_layout)
        contiguous=qkv.contiguous()
        self.kernel_for('qk_norm_rope',call)(contiguous,q_weight.contiguous(),k_weight.contiguous(),cos_sin_cache.contiguous(),positions.contiguous())
        if contiguous is not qkv:qkv.copy_(contiguous)
