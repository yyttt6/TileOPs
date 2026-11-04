"""Ascend sqrt with shape-independent numerical semantics and tiled tails."""

from tileops.kernels.ascend.elementwise._logroots import LogRootsKernel, T
from tileops.kernels.elementwise.call_spec import UnaryElementwiseFwdInterface

__all__ = ["AscendSqrtKernel"]


class AscendSqrtKernel(LogRootsKernel, UnaryElementwiseFwdInterface):
    kind = "sqrt"
    configs = {
        (16777216, "float16"): (8192, 64, 3),
        "float16": (8192, 64, 3),
        (16777216, "bfloat16"): (16384, 64, 2),
        "bfloat16": (16384, 64, 2),
        (16777216, "float32"): (4096, 64, 4),
        "float32": (8192, 64, 3),
        (268435456, "float16"): (8192, 64, 3),
        (268435456, "bfloat16"): (8192, 64, 3),
    }


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("SqrtFwdOp", "ascend", AscendSqrtKernel)
