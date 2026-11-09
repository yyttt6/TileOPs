"""Call interfaces for the extended fused normalization manifest entries."""
import dataclasses
from abc import abstractmethod
import torch
from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import KernelInterface

@dataclasses.dataclass(frozen=True)
class FusedRMSQuantCall(CallSpec):
    rows:int=0
    cols:int=0
    dtype:torch.dtype=torch.float16
    quant_dtype:str='float8_e4m3fn'
    eps:float=1e-6
    scale_ub:float|None=None

class FusedRMSQuantInterface(KernelInterface):
    request=FusedRMSQuantCall
    @abstractmethod
    def forward(self,x,residual,weight):
        """Return q, FP32 per-row scale and input-dtype residual sum; inputs unchanged."""

@dataclasses.dataclass(frozen=True)
class QKNormRoPECall(CallSpec):
    tokens:int=0
    heads:int=0
    kv_heads:int=0
    dim:int=0
    cache_rows:int=0
    rotary_dim:int=0
    dtype:torch.dtype=torch.float16
    cache_dtype:torch.dtype=torch.float32
    eps:float=1e-6
    layout:str='neox'

class QKNormRoPEInterface(KernelInterface):
    request=QKNormRoPECall
    @abstractmethod
    def forward(self,qkv,q_weight,k_weight,cos_sin_cache,positions):
        """Mutate Q/K in packed contiguous QKV; preserve V and other inputs; return None."""
