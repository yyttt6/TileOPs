"""Ascend rsqrt with shape-independent numerical semantics and tiled tails."""

from tileops.kernels.ascend.elementwise._logroots import LogRootsKernel, T
from tileops.kernels.elementwise.call_spec import UnaryElementwiseFwdInterface

__all__ = ["AscendRsqrtKernel"]


class AscendRsqrtKernel(LogRootsKernel, UnaryElementwiseFwdInterface):
    kind = "rsqrt"
    configs = {
        (16777216, "float16"): (8192, 64, 3),
        "float16": (8192, 64, 3),
        (16777216, "bfloat16"): (8192, 64, 3),
        "bfloat16": (8192, 64, 3),
        (16777216, "float32"): (4096, 64, 6),
        "float32": (8192, 64, 3),
        (268435456, "float16"): (8192, 64, 3),
        (268435456, "bfloat16"): (8192, 64, 3),
    }


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("RsqrtFwdOp", "ascend", AscendRsqrtKernel)
