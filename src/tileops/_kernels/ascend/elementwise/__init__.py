"""Ascend elementwise implementations."""

from tileops._kernels.ascend.elementwise.abs import AscendAbsKernel
from tileops._kernels.ascend.elementwise.alibi import AscendAlibiKernel
from tileops._kernels.ascend.elementwise.arithmetic import AscendAddKernel
from tileops._kernels.ascend.elementwise.arithmetic_div import AscendDivKernel
from tileops._kernels.ascend.elementwise.arithmetic_extrema import (
    AscendExtremaKernel,
    AscendMaximumKernel,
    AscendMinimumKernel,
)
from tileops._kernels.ascend.elementwise.arithmetic_modulo import (
    AscendFloorDivideKernel,
    AscendRemainderKernel,
)
from tileops._kernels.ascend.elementwise.arithmetic_mul import AscendMulKernel
from tileops._kernels.ascend.elementwise.arithmetic_power import AscendPowKernel
from tileops._kernels.ascend.elementwise.arithmetic_sub import AscendSubKernel
from tileops._kernels.ascend.elementwise.bitwise_binary import AscendBitwiseAndKernel
from tileops._kernels.ascend.elementwise.bitwise_not import AscendBitwiseNotKernel
from tileops._kernels.ascend.elementwise.bitwise_or import AscendBitwiseOrKernel
from tileops._kernels.ascend.elementwise.bitwise_xor import AscendBitwiseXorKernel
from tileops._kernels.ascend.elementwise.ceil import AscendCeilKernel
from tileops._kernels.ascend.elementwise.clamp_scalar import AscendClampScalarKernel
from tileops._kernels.ascend.elementwise.clamp_tensor import AscendClampTensorKernel
from tileops._kernels.ascend.elementwise.comparison import (
    AscendEqKernel,
    AscendGeKernel,
    AscendGtKernel,
    AscendLeKernel,
    AscendLtKernel,
    AscendNeKernel,
)
from tileops._kernels.ascend.elementwise.cos import AscendCosKernel
from tileops._kernels.ascend.elementwise.dropout import AscendDropoutKernel
from tileops._kernels.ascend.elementwise.elu import AscendELUKernel
from tileops._kernels.ascend.elementwise.erf import AscendErfKernel
from tileops._kernels.ascend.elementwise.exp import AscendExpKernel
from tileops._kernels.ascend.elementwise.expm1 import AscendExpm1Kernel
from tileops._kernels.ascend.elementwise.floor import AscendFloorKernel
from tileops._kernels.ascend.elementwise.gelu import AscendGELUKernel
from tileops._kernels.ascend.elementwise.gelu_and_mul import AscendGELUAndMulKernel
from tileops._kernels.ascend.elementwise.gelu_tanh_and_mul import AscendGELUTanhAndMulKernel
from tileops._kernels.ascend.elementwise.hardsigmoid import AscendHardsigmoidKernel
from tileops._kernels.ascend.elementwise.hardswish import AscendHardswishKernel
from tileops._kernels.ascend.elementwise.hardtanh import AscendHardtanhKernel
from tileops._kernels.ascend.elementwise.isfinite import AscendIsfiniteKernel
from tileops._kernels.ascend.elementwise.isinf import AscendIsinfKernel
from tileops._kernels.ascend.elementwise.isnan import AscendIsnanKernel
from tileops._kernels.ascend.elementwise.leaky_relu import AscendLeakyReLUKernel
from tileops._kernels.ascend.elementwise.lerp import AscendLerpScalarKernel, AscendLerpTensorKernel
from tileops._kernels.ascend.elementwise.log import AscendLogKernel
from tileops._kernels.ascend.elementwise.log1p import AscendLog1pKernel
from tileops._kernels.ascend.elementwise.logical_and import AscendLogicalAndKernel
from tileops._kernels.ascend.elementwise.logical_not import AscendLogicalNotKernel
from tileops._kernels.ascend.elementwise.logical_or import AscendLogicalOrKernel
from tileops._kernels.ascend.elementwise.masked_fill_scalar import AscendMaskedFillScalarKernel
from tileops._kernels.ascend.elementwise.masked_fill_tensor import AscendMaskedFillTensorKernel
from tileops._kernels.ascend.elementwise.mish import AscendMishKernel
from tileops._kernels.ascend.elementwise.nan_to_num import AscendNanToNumKernel
from tileops._kernels.ascend.elementwise.neg import AscendNegKernel
from tileops._kernels.ascend.elementwise.prelu import AscendPReLUKernel
from tileops._kernels.ascend.elementwise.reciprocal import AscendReciprocalKernel
from tileops._kernels.ascend.elementwise.relu import AscendReLUKernel
from tileops._kernels.ascend.elementwise.round import AscendRoundKernel
from tileops._kernels.ascend.elementwise.rsqrt import AscendRsqrtKernel
from tileops._kernels.ascend.elementwise.selu import AscendSELUKernel
from tileops._kernels.ascend.elementwise.sigmoid import AscendSigmoidKernel
from tileops._kernels.ascend.elementwise.sign import AscendSignKernel
from tileops._kernels.ascend.elementwise.silu import AscendSiLUKernel
from tileops._kernels.ascend.elementwise.silu_and_mul import AscendSiLUAndMulKernel
from tileops._kernels.ascend.elementwise.sin import AscendSinKernel
from tileops._kernels.ascend.elementwise.sinusoidal import AscendSinusoidalKernel
from tileops._kernels.ascend.elementwise.softplus import AscendSoftplusKernel
from tileops._kernels.ascend.elementwise.sqrt import AscendSqrtKernel
from tileops._kernels.ascend.elementwise.tanh import AscendTanhKernel
from tileops._kernels.ascend.elementwise.trunc import AscendTruncKernel
from tileops._kernels.ascend.elementwise.where import AscendWhereKernel

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
    "AscendELUKernel",
    "AscendEqKernel",
    "AscendErfKernel",
    "AscendExpKernel",
    "AscendExpm1Kernel",
    "AscendExtremaKernel",
    "AscendFloorDivideKernel",
    "AscendFloorKernel",
    "AscendGELUAndMulKernel",
    "AscendGELUKernel",
    "AscendGELUTanhAndMulKernel",
    "AscendGeKernel",
    "AscendGtKernel",
    "AscendHardsigmoidKernel",
    "AscendHardswishKernel",
    "AscendHardtanhKernel",
    "AscendIsfiniteKernel",
    "AscendIsinfKernel",
    "AscendIsnanKernel",
    "AscendLeKernel",
    "AscendLeakyReLUKernel",
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
    "AscendPReLUKernel",
    "AscendPowKernel",
    "AscendReLUKernel",
    "AscendReciprocalKernel",
    "AscendRemainderKernel",
    "AscendRoundKernel",
    "AscendRsqrtKernel",
    "AscendSELUKernel",
    "AscendSiLUAndMulKernel",
    "AscendSiLUKernel",
    "AscendSigmoidKernel",
    "AscendSignKernel",
    "AscendSinKernel",
    "AscendSinusoidalKernel",
    "AscendSoftplusKernel",
    "AscendSqrtKernel",
    "AscendSubKernel",
    "AscendTanhKernel",
    "AscendTruncKernel",
    "AscendWhereKernel",
]
