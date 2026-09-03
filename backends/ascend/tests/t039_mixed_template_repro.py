"""Minimal reproductions for the same-dtype binary-template boundary."""

import torch

from tileops_ascend.families.elementwise import _batch_builder
from tileops_ascend.kernels.elementwise_binary_batch import build_batch_binary


def show(label, fn):
    try:
        fn()
    except Exception as exc:
        print(label, type(exc).__name__, str(exc))
    else:
        print(label, "unexpectedly accepted")


def main():
    data = torch.zeros((2, 3, 4, 4), dtype=torch.float16).npu()
    mask = torch.zeros((2, 3, 4, 4), dtype=torch.bool).npu()
    weight = torch.zeros((3,), dtype=torch.float16).npu()
    integers = torch.zeros((16,), dtype=torch.int32).npu()
    show(
        "masked_fill_mixed_dtype",
        lambda: _batch_builder("MaskedFillScalarFwdOp", "mul")(data, mask),
    )
    show(
        "prelu_channel_mapping",
        lambda: build_batch_binary(
            data.shape,
            weight.shape,
            data.dtype,
            op_kind="mul",
            op_name="PreluFwdOp",
        ),
    )
    show(
        "bitwise_integer_domain",
        lambda: build_batch_binary(
            integers.shape,
            integers.shape,
            integers.dtype,
            op_kind="mul",
            op_name="BitwiseAndFwdOp",
        ),
    )


if __name__ == "__main__":
    main()
