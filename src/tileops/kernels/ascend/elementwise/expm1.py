"""Ascend expm1: vector FP32 evaluation, pipelined copies and arbitrary tails."""

from tileops.kernels.ascend.elementwise.exponential import ExponentialKernel, T


class AscendExpm1Kernel(ExponentialKernel):
    minus_one = True


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("Expm1FwdOp", "ascend", AscendExpm1Kernel)
