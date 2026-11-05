"""Ascend elementwise implementations."""

from tileops.kernels.ascend.elementwise.abs import AscendAbsKernel
from tileops.kernels.ascend.elementwise.alibi import AscendAlibiKernel
from tileops.kernels.ascend.elementwise.arithmetic import AscendAddKernel
from tileops.kernels.ascend.elementwise.arithmetic_div import AscendDivKernel
from tileops.kernels.ascend.elementwise.arithmetic_extrema import (
    AscendExtremaKernel,
    AscendMaximumKernel,
    AscendMinimumKernel,
)
from tileops.kernels.ascend.elementwise.arithmetic_modulo import (
    AscendFloorDivideKernel,
    AscendRemainderKernel,
)
from tileops.kernels.ascend.elementwise.arithmetic_mul import AscendMulKernel
from tileops.kernels.ascend.elementwise.arithmetic_power import AscendPowKernel
from tileops.kernels.ascend.elementwise.arithmetic_sub import AscendSubKernel
from tileops.kernels.ascend.elementwise.bitwise_binary import AscendBitwiseAndKernel
from tileops.kernels.ascend.elementwise.bitwise_not import AscendBitwiseNotKernel
from tileops.kernels.ascend.elementwise.bitwise_or import AscendBitwiseOrKernel
from tileops.kernels.ascend.elementwise.bitwise_xor import AscendBitwiseXorKernel
from tileops.kernels.ascend.elementwise.ceil import AscendCeilKernel
from tileops.kernels.ascend.elementwise.clamp_scalar import AscendClampScalarKernel
from tileops.kernels.ascend.elementwise.clamp_tensor import AscendClampTensorKernel
from tileops.kernels.ascend.elementwise.comparison import (
    AscendEqKernel,
    AscendGeKernel,
    AscendGtKernel,
    AscendLeKernel,
    AscendLtKernel,
    AscendNeKernel,
)
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
from tileops.kernels.ascend.elementwise.lerp import AscendLerpScalarKernel, AscendLerpTensorKernel
from tileops.kernels.ascend.elementwise.log import AscendLogKernel
from tileops.kernels.ascend.elementwise.log1p import AscendLog1pKernel
from tileops.kernels.ascend.elementwise.logical_and import AscendLogicalAndKernel
from tileops.kernels.ascend.elementwise.logical_not import AscendLogicalNotKernel
from tileops.kernels.ascend.elementwise.logical_or import AscendLogicalOrKernel
from tileops.kernels.ascend.elementwise.masked_fill_scalar import AscendMaskedFillScalarKernel
from tileops.kernels.ascend.elementwise.masked_fill_tensor import AscendMaskedFillTensorKernel
from tileops.kernels.ascend.elementwise.mish import AscendMishKernel
from tileops.kernels.ascend.elementwise.nan_to_num import AscendNanToNumKernel
from tileops.kernels.ascend.elementwise.neg import AscendNegKernel
from tileops.kernels.ascend.elementwise.prelu import AscendPreluKernel
from tileops.kernels.ascend.elementwise.reciprocal import AscendReciprocalKernel
from tileops.kernels.ascend.elementwise.relu import AscendReluKernel
from tileops.kernels.ascend.elementwise.round import AscendRoundKernel
from tileops.kernels.ascend.elementwise.rsqrt import AscendRsqrtKernel
from tileops.kernels.ascend.elementwise.selu import AscendSeluKernel
from tileops.kernels.ascend.elementwise.sigmoid import AscendSigmoidKernel
from tileops.kernels.ascend.elementwise.sign import AscendSignKernel
from tileops.kernels.ascend.elementwise.silu import AscendSiluKernel
from tileops.kernels.ascend.elementwise.silu_and_mul import AscendSiluAndMulKernel
from tileops.kernels.ascend.elementwise.sin import AscendSinKernel
from tileops.kernels.ascend.elementwise.sinusoidal import AscendSinusoidalKernel
from tileops.kernels.ascend.elementwise.softplus import AscendSoftplusKernel
from tileops.kernels.ascend.elementwise.sqrt import AscendSqrtKernel
from tileops.kernels.ascend.elementwise.tanh import AscendTanhKernel
from tileops.kernels.ascend.elementwise.trunc import AscendTruncKernel
from tileops.kernels.ascend.elementwise.where import AscendWhereKernel

__all__ = [
    "AscendAbsKernel",
    "AscendAddKernel",
    "AscendAlibiKernel",
    "AscendBitwiseAndKernel",
    "AscendBitwiseNotKernel",
    "AscendBitwiseOrKernel",
    "AscendBitwiseXorKernel",
    "AscendCeilKernel",
    "AscendClampScalarKernel",
    "AscendClampTensorKernel",
    "AscendCosKernel",
    "AscendDivKernel",
    "AscendDropoutKernel",
    "AscendEluKernel",
    "AscendEqKernel",
    "AscendErfKernel",
    "AscendExpKernel",
    "AscendExpm1Kernel",
    "AscendExtremaKernel",
    "AscendFloorDivideKernel",
    "AscendFloorKernel",
    "AscendGeKernel",
    "AscendGeluAndMulKernel",
    "AscendGeluKernel",
    "AscendGeluTanhAndMulKernel",
    "AscendGtKernel",
    "AscendHardsigmoidKernel",
    "AscendHardswishKernel",
    "AscendHardtanhKernel",
    "AscendIsfiniteKernel",
    "AscendIsinfKernel",
    "AscendIsnanKernel",
    "AscendLeKernel",
    "AscendLeakyReluKernel",
    "AscendLerpScalarKernel",
    "AscendLerpTensorKernel",
    "AscendLog1pKernel",
    "AscendLogKernel",
    "AscendLogicalAndKernel",
    "AscendLogicalNotKernel",
    "AscendLogicalOrKernel",
    "AscendLtKernel",
    "AscendMaskedFillScalarKernel",
    "AscendMaskedFillTensorKernel",
    "AscendMaximumKernel",
    "AscendMinimumKernel",
    "AscendMishKernel",
    "AscendMulKernel",
    "AscendNanToNumKernel",
    "AscendNeKernel",
    "AscendNegKernel",
    "AscendPowKernel",
    "AscendPreluKernel",
    "AscendReciprocalKernel",
    "AscendReluKernel",
    "AscendRemainderKernel",
    "AscendRoundKernel",
    "AscendRsqrtKernel",
    "AscendSeluKernel",
    "AscendSigmoidKernel",
    "AscendSignKernel",
    "AscendSiluAndMulKernel",
    "AscendSiluKernel",
    "AscendSinKernel",
    "AscendSinusoidalKernel",
    "AscendSoftplusKernel",
    "AscendSqrtKernel",
    "AscendSubKernel",
    "AscendTanhKernel",
    "AscendTruncKernel",
    "AscendWhereKernel",
]
