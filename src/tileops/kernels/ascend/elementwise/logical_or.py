"""Ascend logical OR, preserving nonzero truthiness and broadcasting."""

from tileops.kernels.ascend.elementwise._logical_binary import LogicalBinaryBase, T


class AscendLogicalOrKernel(LogicalBinaryBase):
    kind = "or"


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("LogicalOrFwdOp", "ascend", AscendLogicalOrKernel)
