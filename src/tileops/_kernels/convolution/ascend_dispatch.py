"""Preserve CUDA selections and route NPU Conv1d to its full-domain implementation."""
from . import conv1d as cuda
from .call_spec import Conv1dFwdInterface
from ..kernel_base import Kernel

class _DeviceConv1d(Kernel, Conv1dFwdInterface):
    devices = frozenset({'cuda', 'npu'})
    supported_archs = None
    @classmethod
    def unavailable(cls, call):
        if call.device.type == 'cuda':
            return cls.cuda_type.unavailable(call)
        return super().unavailable(call)
    @classmethod
    def refusal(cls, call):
        if call.device.type == 'cuda':
            return cls.cuda_type.refusal(call)
        if cls.cuda_type is not cuda.Conv1dKernel:
            return 'Ascend uses the general Conv1d implementation'
        return super().refusal(call)
    @classmethod
    def entry_for(cls, call):
        if call.device.type == 'cuda':
            return cls.cuda_type.entry_for(call)
        from tileops._kernels.ascend.convolution.conv1d import AscendConv1dKernel
        return AscendConv1dKernel.entry_for(call)
    def forward(self, x, weight, bias=None):
        raise RuntimeError('Device dispatch must select a concrete kernel')

class Conv1dKernel(_DeviceConv1d):
    cuda_type = cuda.Conv1dKernel
    preferred_over = cuda.Conv1dKernel.preferred_over
    general = True

class Conv1dPointwiseKernel(_DeviceConv1d):
    cuda_type = cuda.Conv1dPointwiseKernel
    preferred_over = cuda.Conv1dPointwiseKernel.preferred_over
    general = False

class Conv1dUnitStrideKernel(_DeviceConv1d):
    cuda_type = cuda.Conv1dUnitStrideKernel
    preferred_over = cuda.Conv1dUnitStrideKernel.preferred_over
    general = False

class DepthwiseConv1dKernel(_DeviceConv1d):
    cuda_type = cuda.DepthwiseConv1dKernel
    preferred_over = cuda.DepthwiseConv1dKernel.preferred_over
    general = False

class GroupConv1dKernel(_DeviceConv1d):
    cuda_type = cuda.GroupConv1dKernel
    preferred_over = cuda.GroupConv1dKernel.preferred_over
    general = False
