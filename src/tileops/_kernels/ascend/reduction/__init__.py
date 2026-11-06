"""Ascend reduction implementations."""

from tileops._kernels.ascend.reduction.all import AscendAllKernel
from tileops._kernels.ascend.reduction.amax import AscendAmaxKernel
from tileops._kernels.ascend.reduction.amin import AscendAminKernel
from tileops._kernels.ascend.reduction.any import AscendAnyKernel
from tileops._kernels.ascend.reduction.argmax import AscendArgmaxKernel
from tileops._kernels.ascend.reduction.argmin import AscendArgminKernel
from tileops._kernels.ascend.reduction.count_nonzero import AscendCountNonzeroKernel
from tileops._kernels.ascend.reduction.cumprod import AscendCumprodKernel
from tileops._kernels.ascend.reduction.cumsum import AscendCumsumKernel
from tileops._kernels.ascend.reduction.log_softmax import AscendLogSoftmaxKernel
from tileops._kernels.ascend.reduction.logsumexp import AscendLogSumExpKernel
from tileops._kernels.ascend.reduction.mean import AscendMeanKernel
from tileops._kernels.ascend.reduction.prod import AscendProdKernel
from tileops._kernels.ascend.reduction.softmax import AscendSoftmaxKernel
from tileops._kernels.ascend.reduction.std import AscendStdKernel
from tileops._kernels.ascend.reduction.sum import AscendSumKernel
from tileops._kernels.ascend.reduction.var import AscendVarKernel
from tileops._kernels.ascend.reduction.var_mean import AscendVarMeanKernel
from tileops._kernels.ascend.reduction.vector_norm import AscendVectorNormKernel

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
