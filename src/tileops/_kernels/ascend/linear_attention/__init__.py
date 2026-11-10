"""Ascend linear attention implementations."""

from tileops._kernels.ascend.linear_attention import (
    deltanet,
    deltanet_recurrent,
    deltanet_training_bwd,
    deltanet_training_fwd,
    gdn,
    gla,
    gla_recurrent,
    gla_training_bwd,
    gla_training_fwd,
    kda,
)
