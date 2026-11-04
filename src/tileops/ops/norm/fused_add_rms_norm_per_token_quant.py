"""Residual RMSNorm followed by dynamic per-row INT8/FP8 quantization."""
from tileops.ops.op_base import Op
from tileops.kernels.norm.fused_extended_call_spec import FusedRMSQuantCall,FusedRMSQuantInterface

class FusedAddRMSNormPerTokenQuantFwdOp(Op):
    kernel_types={}
    interfaces={'fused_rms_quant':FusedRMSQuantInterface}
    def __init__(self,quant_dtype='float8_e4m3fn',eps=1e-6,scale_ub=None,*,target=None):
        quant_dtype = str(quant_dtype).removeprefix("torch.")
        if quant_dtype not in ("int8", "float8_e4m3fn"):
            raise ValueError("quant_dtype must be int8 or float8_e4m3fn")
        self.quant_dtype=quant_dtype;self.eps=eps;self.scale_ub=scale_ub
        super().__init__(target=target)
    def forward(self,x,residual,weight):
        width=x.shape[-1]
        call=FusedRMSQuantCall(device=x.device,rows=x.numel()//width,cols=width,dtype=x.dtype,quant_dtype=self.quant_dtype,eps=self.eps,scale_ub=self.scale_ub)
        q,scale,res=self.kernel_for('fused_rms_quant',call)(x.contiguous().reshape(-1,width),residual.contiguous().reshape(-1,width),weight.contiguous())
        return q.reshape_as(x),scale.reshape(*x.shape[:-1],1),res.reshape_as(x)
