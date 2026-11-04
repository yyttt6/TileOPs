"""Ascend exp: vector FP32 evaluation, pipelined copies and arbitrary tails."""

from tileops.kernels.ascend.elementwise.exponential import ExponentialKernel, T


class AscendExpKernel(ExponentialKernel):
    minus_one = False
    fp32_tile = 12288
    fp32_stages = 2


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("ExpFwdOp", "ascend", AscendExpKernel)
