"""Accepted Ascend normalization kernels."""

from tileops.kernels.ascend.norm.ada_layer_norm import (
    AscendAdaLayerNormKernel,
    AscendAdaLayerNormZeroKernel,
)
from tileops.kernels.ascend.norm.batch_norm_backward import AscendBatchNormBwdKernel
from tileops.kernels.ascend.norm.batch_norm_forward import (
    AscendBatchNormInferKernel,
    AscendBatchNormTrainKernel,
)
from tileops.kernels.ascend.norm.fused_add_layer_norm import AscendFusedAddLayerNormKernel
from tileops.kernels.ascend.norm.fused_add_rms_norm import AscendFusedAddRMSNormKernel
from tileops.kernels.ascend.norm.fused_rms_quant import AscendFusedRMSQuantKernel
from tileops.kernels.ascend.norm.group_norm import AscendGroupNormKernel
from tileops.kernels.ascend.norm.instance_norm import (
    AscendInstanceNormInferKernel,
    AscendInstanceNormKernel,
    AscendInstanceNormTrainKernel,
)
from tileops.kernels.ascend.norm.layer_norm import AscendLayerNormKernel
from tileops.kernels.ascend.norm.qk_norm_rope import AscendQKNormRoPEKernel
from tileops.kernels.ascend.norm.rms_norm import AscendRMSNormKernel

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
