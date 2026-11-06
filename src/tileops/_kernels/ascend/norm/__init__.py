"""Accepted Ascend normalization kernels."""

from tileops._kernels.ascend.norm.ada_layer_norm import (
    AscendAdaLayerNormKernel,
    AscendAdaLayerNormZeroKernel,
)
from tileops._kernels.ascend.norm.batch_norm_backward import AscendBatchNormBwdKernel
from tileops._kernels.ascend.norm.batch_norm_forward import (
    AscendBatchNormInferKernel,
    AscendBatchNormTrainKernel,
)
from tileops._kernels.ascend.norm.fused_add_layer_norm import AscendFusedAddLayerNormKernel
from tileops._kernels.ascend.norm.fused_add_rms_norm import AscendFusedAddRMSNormKernel
from tileops._kernels.ascend.norm.fused_rms_quant import AscendFusedRMSQuantKernel
from tileops._kernels.ascend.norm.group_norm import AscendGroupNormKernel
from tileops._kernels.ascend.norm.instance_norm import (
    AscendInstanceNormInferKernel,
    AscendInstanceNormKernel,
    AscendInstanceNormTrainKernel,
)
from tileops._kernels.ascend.norm.layer_norm import AscendLayerNormKernel
from tileops._kernels.ascend.norm.qk_norm_rope import AscendQKNormRoPEKernel
from tileops._kernels.ascend.norm.rms_norm import AscendRMSNormKernel

__all__ = [
    "AscendAdaLayerNormKernel",
    "AscendAdaLayerNormZeroKernel",
    "AscendBatchNormBwdKernel",
    "AscendBatchNormInferKernel",
    "AscendBatchNormTrainKernel",
    "AscendFusedAddLayerNormKernel",
    "AscendFusedAddRMSNormKernel",
    "AscendFusedRMSQuantKernel",
    "AscendGroupNormKernel",
    "AscendInstanceNormInferKernel",
    "AscendInstanceNormKernel",
    "AscendInstanceNormTrainKernel",
    "AscendLayerNormKernel",
    "AscendQKNormRoPEKernel",
    "AscendRMSNormKernel",
]
