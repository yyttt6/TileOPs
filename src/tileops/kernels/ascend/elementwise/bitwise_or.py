"""Ascend bitwise OR using the validated word representation and tiled tails."""

from tileops.kernels.ascend.elementwise.bitwise_binary import BitwiseBinaryBase, T


if T is not None:
    from tileops.kernels.ascend.elementwise._bitwise_fast import (
        FastBitwiseBinaryBase as BitwiseBinaryBase,
    )


class AscendBitwiseOrKernel(BitwiseBinaryBase):
    kind = "or"


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("BitwiseOrFwdOp", "ascend", AscendBitwiseOrKernel)
