"""Ascend reduction implementations."""

from tileops.kernels.ascend.reduction.all import AscendAllKernel
from tileops.kernels.ascend.reduction.amax import AscendAmaxKernel
from tileops.kernels.ascend.reduction.amin import AscendAminKernel
from tileops.kernels.ascend.reduction.any import AscendAnyKernel
from tileops.kernels.ascend.reduction.argmax import AscendArgmaxKernel
from tileops.kernels.ascend.reduction.argmin import AscendArgminKernel
from tileops.kernels.ascend.reduction.count_nonzero import AscendCountNonzeroKernel
from tileops.kernels.ascend.reduction.cumprod import AscendCumprodKernel
from tileops.kernels.ascend.reduction.cumsum import AscendCumsumKernel
from tileops.kernels.ascend.reduction.log_softmax import AscendLogSoftmaxKernel
from tileops.kernels.ascend.reduction.logsumexp import AscendLogSumExpKernel
from tileops.kernels.ascend.reduction.mean import AscendMeanKernel
from tileops.kernels.ascend.reduction.prod import AscendProdKernel
from tileops.kernels.ascend.reduction.softmax import AscendSoftmaxKernel
from tileops.kernels.ascend.reduction.std import AscendStdKernel
from tileops.kernels.ascend.reduction.sum import AscendSumKernel
from tileops.kernels.ascend.reduction.var import AscendVarKernel
from tileops.kernels.ascend.reduction.var_mean import AscendVarMeanKernel
from tileops.kernels.ascend.reduction.vector_norm import AscendVectorNormKernel

__all__ = [
    "AscendAllKernel",
    "AscendAmaxKernel",
    "AscendAminKernel",
    "AscendAnyKernel",
    "AscendArgmaxKernel",
    "AscendArgminKernel",
    "AscendCountNonzeroKernel",
    "AscendCumprodKernel",
    "AscendCumsumKernel",
    "AscendLogSoftmaxKernel",
    "AscendLogSumExpKernel",
    "AscendMeanKernel",
    "AscendProdKernel",
    "AscendSoftmaxKernel",
    "AscendStdKernel",
    "AscendSumKernel",
    "AscendVarKernel",
    "AscendVarMeanKernel",
    "AscendVectorNormKernel",
]
