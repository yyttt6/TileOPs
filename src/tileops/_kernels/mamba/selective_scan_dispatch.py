"""Device delegation preserving the public step/chunk dispatch keys."""
from .selective_scan import SelectiveScanChunkKernel as CUDAChunk
from .selective_scan import SelectiveScanStepKernel as CUDAStep
from .call_spec import SelectiveScanFwdInterface
from ..kernel_base import Kernel

class _DeviceScan(Kernel, SelectiveScanFwdInterface):
    devices = frozenset({'cuda', 'npu'})
    supported_archs = None
    cuda_type = None

    @classmethod
    def unavailable(cls, call):
        from tileops._utils.device import current_device
        device = call.device or current_device()
        if device is not None and device.type == 'cuda':
            return cls.cuda_type.unavailable(call)
        return super().unavailable(call)

    @classmethod
    def refusal(cls, call):
        return cls.cuda_type.refusal(call)

    @classmethod
    def entry_for(cls, call):
        from tileops._utils.device import current_device
        device = call.device or current_device()
        if device.type == 'cuda':
            return cls.cuda_type.entry_for(call)
        from tileops._kernels.ascend.mamba.selective_scan import AscendSelectiveScanKernel
        return AscendSelectiveScanKernel.entry_for(call)

    def forward(self, u, delta, A, B_in, C_in, D_in=None, z=None,
                delta_bias=None, seq_idx=None, initial_states=None):
        raise RuntimeError('device delegation must build the selected concrete kernel')

class SelectiveScanChunkKernel(_DeviceScan):
    general = True
    cuda_type = CUDAChunk

class SelectiveScanStepKernel(_DeviceScan):
    preferred_over = frozenset({'selective_scan_chunk'})
    cuda_type = CUDAStep
