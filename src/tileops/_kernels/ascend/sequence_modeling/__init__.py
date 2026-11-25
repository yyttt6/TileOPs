"""Ascend kernels for sequence-modeling operators."""

from tileops._kernels.ascend.sequence_modeling import decode, hyper_connection
from tileops._kernels.ascend.sequence_modeling.gate_conv_bwd import AscendEngramGateConvBwdKernel
from tileops._kernels.ascend.sequence_modeling.gate_conv_fwd import AscendEngramGateConvFwdKernel
from tileops._kernels.ascend.sequence_modeling.mhc_post import AscendMHCPostKernel
from tileops._kernels.ascend.sequence_modeling.mhc_pre import AscendMHCPreKernel
from tileops._kernels.ascend.sequence_modeling.ngram_hash import AscendNGramHashKernel

__all__ = [
    "AscendEngramGateConvBwdKernel",
    "AscendEngramGateConvFwdKernel",
    "AscendMHCPostKernel",
    "AscendMHCPreKernel",
    "AscendNGramHashKernel",
    "decode",
    "hyper_connection",
]
