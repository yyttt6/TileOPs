"""R355 / T355 #142 -- ``Conv2dFp8FwdOp``.

``conv2d_fp8`` is the only row of ``docs/reports/R267-data/oplist150.json`` with no
operator at all (idx 142, ``"op": null``).  This class, the manifest entry in
``manifest/convolution.yaml``, the builder in ``kernels/families/convolution.py`` and
the harness adapter ``adapters/conv2d_fp8.py`` exist to give that row a measured
number.

A NEW MODULE on purpose: ``ops/convolution.py`` and ``ops/conv_variants/`` are shared
with other rounds, and F067 measured that a shared file taken by two parallel rounds
gets both discarded.  The harness adapter imports this module by name.
"""

from typing import ClassVar, Optional, Tuple

import torch

from .op_base import Op, Target
from tileops.backend import Kernel

__all__ = ["Conv2dFp8FwdOp"]


def _pair(name: str, value) -> Tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value), int(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        if all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            return int(value[0]), int(value[1])
    raise TypeError(f"Conv2dFp8FwdOp {name} must be an int or a length-2 tuple")


class Conv2dFp8FwdOp(Op):
    """float8_e4m3fn NCHW convolution with a float16 result.

    🚨 NO DEQUANTIZATION SCALE, and that is a declared choice, not an omission: the
    operands are e4m3fn containers and this op convolves the values they DECODE TO.
    A per-tensor or per-channel scale folds into the output and is the caller's; the
    150-row source names ``conv2d_fp8`` and specifies nothing further, so inventing a
    scale would have invented a signature.  R355.md section 6.1.
    """

    compile_op_names: ClassVar[Tuple[str, ...]] = ()

    def __init__(
        self,
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        groups: int = 1,
        *,
        target: Target = None,
        tune: bool = False,
    ) -> None:
        """Build the op.  Shapes are taken from the first call.

        Args:
            stride: Manifest ``params.stride``, ``int | tuple[int, int]``, default ``1``.
            padding: Manifest ``params.padding``, ``int | tuple[int, int]``, default ``0``.
            dilation: Manifest ``params.dilation``, ``int | tuple[int, int]``, default ``1``.
            groups: Manifest ``params.groups``, ``int``, default ``1``.
            target: Backend target to serve this op, or ``None`` to decide from the device.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        if not isinstance(groups, int) or isinstance(groups, bool) or groups <= 0:
            raise ValueError("Conv2dFp8FwdOp groups must be a positive int")
        self.stride = _pair("stride", stride)
        self.padding = _pair("padding", padding)
        self.dilation = _pair("dilation", dilation)
        self.groups = groups
        self.n = None
        self.c_in = None
        self.h = None
        self.w = None
        self.c_out = None
        self.kernel_size = None
        self.dtype = None
        self.target = target
        self.tune = tune
        self.dispatch_kernel()
        self.kernel: Optional[Kernel] = None

    def _validate_dtypes(self, *tensors: "torch.Tensor | None") -> None:
        """Supplied rather than generated: ``float8_e4m3fn`` is the only dtype here and
        the generated validator's message would name a dtype list of one."""
        for tensor in tensors:
            if tensor is None:
                continue
            if tensor.dtype is not torch.float8_e4m3fn:
                raise TypeError(
                    f"Conv2dFp8FwdOp does not support dtype {tensor.dtype}; supported "
                    f"dtypes are [torch.float8_e4m3fn], but received {tensor.dtype}"
                )

    def _infer_output_shapes(
        self,
        input_shape: tuple[int, ...],
        weight_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        n, _c_in, h, w = input_shape
        c_out, _c_in_g, k_h, k_w = weight_shape
        out_h = (h + 2 * self.padding[0] - self.dilation[0] * (k_h - 1) - 1) // self.stride[0] + 1
        out_w = (w + 2 * self.padding[1] - self.dilation[1] * (k_w - 1) - 1) // self.stride[1] + 1
        return {"output": (n, c_out, out_h, out_w)}

    def eval_roofline(self) -> tuple[int, int]:
        """FLOPs and bytes for the shape the last call settled on.

        Supplied rather than generated because the operands are one byte each and the
        result is two -- the manifest's ``elem_bytes`` is a single width per entry and
        there is no single width here.
        """
        if self.n is None:
            raise RuntimeError("Conv2dFp8FwdOp.eval_roofline() before the first call")
        out = self._infer_output_shapes(
            (self.n, self.c_in, self.h, self.w),
            (self.c_out, self.c_in // self.groups, *self.kernel_size),
        )["output"]
        out_elems = out[0] * out[1] * out[2] * out[3]
        c_in_g = self.c_in // self.groups
        flops = 2 * out_elems * c_in_g * self.kernel_size[0] * self.kernel_size[1]
        nbytes = (
            self.n * self.c_in * self.h * self.w           # fp8 input, 1 byte
            + self.c_out * c_in_g * self.kernel_size[0] * self.kernel_size[1]
            + 2 * out_elems                                 # fp16 output
        )
        return int(flops), int(nbytes)

    def _get_kernel(self, inputs) -> Kernel:
        return self.get_or_build_kernel("conv2d_fp8_kernel", inputs)

    def forward(self, input, weight):
        """Run the op on the inputs the manifest declares.

        Args:
            input: ``[N, C_in, H, W]``, dtype ``float8_e4m3fn``.
            weight: ``[C_out, C_in_g, kH, kW]``, dtype ``float8_e4m3fn``.

        Returns:
            ``[N, C_out, out_H, out_W]``, dtype ``float16``.
        """
        if input.device.type != "npu" or weight.device.type != "npu":
            raise ValueError("Conv2dFp8FwdOp expects NPU tensors")
        if input.device != weight.device:
            raise ValueError("Conv2dFp8FwdOp tensors must share a device")
        if input.ndim != 4 or weight.ndim != 4:
            raise ValueError("Conv2dFp8FwdOp expects 4D NCHW input and OIHW weight")
        self._validate_dtypes(input, weight)
        self.n, self.c_in, self.h, self.w = (int(v) for v in input.shape)
        self.c_out = int(weight.shape[0])
        self.kernel_size = (int(weight.shape[2]), int(weight.shape[3]))
        self.dtype = input.dtype
        if self.c_in % self.groups or self.c_out % self.groups:
            raise ValueError("Conv2dFp8FwdOp channels must be divisible by groups")
        if int(weight.shape[1]) != self.c_in // self.groups:
            raise ValueError(
                f"Conv2dFp8FwdOp expected weight.shape[1]={self.c_in // self.groups}, "
                f"got {int(weight.shape[1])}"
            )
        self.kernel = self._get_kernel((input, weight))
        return self.kernel(input, weight)
