"""Keep CUDA dispatch keys while supplying the full-domain Ascend implementation."""
from .hadamard_register import HadamardWarpRowsKernel as CUDAWarp, HadamardExchangeKernel as CUDAExchange, HadamardBaseExchangeKernel as CUDABase
from .hadamard_shared import HadamardSharedRowsKernel as CUDAShared, HadamardMultiPassKernel as CUDAMulti
from .call_spec import HadamardTransformFwdInterface
from ..kernel_base import Kernel

def _device(call):
    from tileops._utils.device import current_device
    return call.device or current_device()

class _DeviceHadamard(Kernel,HadamardTransformFwdInterface):
    devices=frozenset({'cuda','npu'})
    supported_archs=None
    cuda_type=None
    @classmethod
    def unavailable(cls,call):
        if _device(call).type=='cuda':
            return cls.cuda_type.unavailable(call)
        return super().unavailable(call)
    @classmethod
    def refusal(cls,call):
        if _device(call).type=='cuda':
            return cls.cuda_type.refusal(call)
        if cls.cuda_type is not CUDAMulti:
            return 'Ascend uses the full-domain multi-pass implementation'
        return super().refusal(call)
    @classmethod
    def entry_for(cls,call):
        if _device(call).type=='cuda':
            return cls.cuda_type.entry_for(call)
        from tileops._kernels.ascend.transform.hadamard import AscendHadamardKernel
        return AscendHadamardKernel.entry_for(call)
    def forward(self,x):
        raise RuntimeError('Device dispatch must select a concrete kernel')

class HadamardWarpRowsKernel(_DeviceHadamard):
    cuda_type=CUDAWarp
    preferred_over=CUDAWarp.preferred_over
class HadamardExchangeKernel(_DeviceHadamard):
    cuda_type=CUDAExchange
    preferred_over=CUDAExchange.preferred_over
class HadamardBaseExchangeKernel(_DeviceHadamard):
    cuda_type=CUDABase
    preferred_over=CUDABase.preferred_over
class HadamardSharedRowsKernel(_DeviceHadamard):
    cuda_type=CUDAShared
    preferred_over=CUDAShared.preferred_over
class HadamardMultiPassKernel(_DeviceHadamard):
    cuda_type=CUDAMulti
    general=True
