"""Ascend bitwise XOR using the validated word representation and tiled tails."""

from tileops.kernels.ascend.elementwise.bitwise_binary import BitwiseBinaryBase, T


if T is not None:
    from tileops.kernels.ascend.elementwise._bitwise_fast import (
        FastBitwiseBinaryBase as BitwiseBinaryBase,
    )


class AscendBitwiseXorKernel(BitwiseBinaryBase):
    kind = "xor"


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("BitwiseXorFwdOp", "ascend", AscendBitwiseXorKernel)
