"""Ascend logical AND, preserving nonzero truthiness and broadcasting."""

from tileops.kernels.ascend.elementwise._logical_binary import LogicalBinaryBase, T


class AscendLogicalAndKernel(LogicalBinaryBase):
    kind = "and"


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("LogicalAndFwdOp", "ascend", AscendLogicalAndKernel)
