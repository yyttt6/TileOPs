"""Ascend kernels for sequence-modeling operators."""

from tileops.kernels.ascend.sequence_modeling import decode
from tileops.kernels.ascend.sequence_modeling.gate_conv_bwd import AscendEngramGateConvBwdKernel
from tileops.kernels.ascend.sequence_modeling.gate_conv_fwd import AscendEngramGateConvFwdKernel

__all__ = ["AscendEngramGateConvBwdKernel", "AscendEngramGateConvFwdKernel", "decode"]
