"""Ascend 950 elementwise kernels."""

from tileops.kernels.ascend.elementwise.abs import AscendAbsKernel
from tileops.kernels.ascend.elementwise.bitwise_not import AscendBitwiseNotKernel
from tileops.kernels.ascend.elementwise.ceil import AscendCeilKernel
from tileops.kernels.ascend.elementwise.cos import AscendCosKernel
from tileops.kernels.ascend.elementwise.dropout import AscendDropoutKernel
from tileops.kernels.ascend.elementwise.elu import AscendEluKernel
from tileops.kernels.ascend.elementwise.erf import AscendErfKernel
from tileops.kernels.ascend.elementwise.exp import AscendExpKernel
from tileops.kernels.ascend.elementwise.expm1 import AscendExpm1Kernel
from tileops.kernels.ascend.elementwise.floor import AscendFloorKernel
from tileops.kernels.ascend.elementwise.gelu import AscendGeluKernel
from tileops.kernels.ascend.elementwise.gelu_and_mul import AscendGeluAndMulKernel
from tileops.kernels.ascend.elementwise.gelu_tanh_and_mul import AscendGeluTanhAndMulKernel
from tileops.kernels.ascend.elementwise.hardsigmoid import AscendHardsigmoidKernel
from tileops.kernels.ascend.elementwise.hardswish import AscendHardswishKernel
from tileops.kernels.ascend.elementwise.hardtanh import AscendHardtanhKernel
from tileops.kernels.ascend.elementwise.isfinite import AscendIsfiniteKernel
from tileops.kernels.ascend.elementwise.isinf import AscendIsinfKernel
from tileops.kernels.ascend.elementwise.isnan import AscendIsnanKernel
from tileops.kernels.ascend.elementwise.leaky_relu import AscendLeakyReluKernel
from tileops.kernels.ascend.elementwise.log import AscendLogKernel
from tileops.kernels.ascend.elementwise.log1p import AscendLog1pKernel
from tileops.kernels.ascend.elementwise.logical_not import AscendLogicalNotKernel
from tileops.kernels.ascend.elementwise.mish import AscendMishKernel
from tileops.kernels.ascend.elementwise.neg import AscendNegKernel
from tileops.kernels.ascend.elementwise.reciprocal import AscendReciprocalKernel
from tileops.kernels.ascend.elementwise.relu import AscendReluKernel
from tileops.kernels.ascend.elementwise.round import AscendRoundDecimalsKernel, AscendRoundKernel
from tileops.kernels.ascend.elementwise.rsqrt import AscendRsqrtKernel
from tileops.kernels.ascend.elementwise.selu import AscendSeluKernel
from tileops.kernels.ascend.elementwise.sigmoid import AscendSigmoidKernel
from tileops.kernels.ascend.elementwise.sign import AscendSignKernel
from tileops.kernels.ascend.elementwise.silu import AscendSiluKernel
from tileops.kernels.ascend.elementwise.silu_and_mul import AscendSiluAndMulKernel
from tileops.kernels.ascend.elementwise.sin import AscendSinKernel
from tileops.kernels.ascend.elementwise.softplus import AscendSoftplusKernel
from tileops.kernels.ascend.elementwise.sqrt import AscendSqrtKernel
from tileops.kernels.ascend.elementwise.tanh import AscendTanhKernel
from tileops.kernels.ascend.elementwise.trunc import AscendTruncKernel

__all__ = [
    "AscendAbsKernel",
    "AscendBitwiseNotKernel",
    "AscendCeilKernel",
    "AscendCosKernel",
    "AscendDropoutKernel",
    "AscendEluKernel",
    "AscendErfKernel",
    "AscendExpKernel",
    "AscendExpm1Kernel",
    "AscendFloorKernel",
    "AscendGeluAndMulKernel",
    "AscendGeluKernel",
    "AscendGeluTanhAndMulKernel",
    "AscendHardsigmoidKernel",
    "AscendHardswishKernel",
    "AscendHardtanhKernel",
    "AscendIsfiniteKernel",
    "AscendIsinfKernel",
    "AscendIsnanKernel",
    "AscendLeakyReluKernel",
    "AscendLog1pKernel",
    "AscendLogKernel",
    "AscendLogicalNotKernel",
    "AscendMishKernel",
    "AscendNegKernel",
    "AscendReciprocalKernel",
    "AscendReluKernel",
    "AscendRoundDecimalsKernel",
    "AscendRoundKernel",
    "AscendRsqrtKernel",
    "AscendSeluKernel",
    "AscendSigmoidKernel",
    "AscendSignKernel",
    "AscendSiluAndMulKernel",
    "AscendSiluKernel",
    "AscendSinKernel",
    "AscendSoftplusKernel",
    "AscendSqrtKernel",
    "AscendTanhKernel",
    "AscendTruncKernel",
]
