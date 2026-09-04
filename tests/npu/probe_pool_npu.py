"""Deterministic NPU correctness matrix for the Ascend max-pool pilot."""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

import tileops.kernels.families.pool  # noqa: F401 - registration side effect
from tileops.ops.pool import MaxPool2dFwdOp

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

MANIFEST_CASES = [
    ((32, 64, 112, 112), (3, 3), (2, 2), (1, 1), (1, 1), False, "resnet-stem"),
    ((16, 128, 56, 56), (2, 2), (2, 2), (0, 0), (1, 1), False, "vgg-block"),
    ((32, 64, 55, 55), (3, 3), (2, 2), (0, 0), (1, 1), True, "alexnet-ceil"),
]


def _run_case(shape, kernel, stride, padding, dilation, ceil_mode, dtype_name, label):
    dtype = DTYPES[dtype_name]
    seed = 20260825 + sum(shape) + len(label) + dtype.itemsize
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cpu_input = torch.randn(shape, dtype=dtype, generator=generator)
    reference = F.max_pool2d(
        cpu_input,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    npu_input = cpu_input.npu()
    op = MaxPool2dFwdOp(
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        target="ascend",
    )
    output = op(npu_input)
    torch.npu.synchronize()
    actual = output.cpu()
    finite = bool(torch.isfinite(actual).all())
    max_abs = float((actual.float() - reference.float()).abs().max())
    atol = rtol = 1.6e-2 if dtype_name == "bfloat16" else 1e-3
    passed = finite and torch.allclose(actual, reference, atol=atol, rtol=rtol)
    print(
        f"[{'PRECISION_PASS' if passed else 'PRECISION_FAIL'}] {label} "
        f"shape={shape} dtype={dtype_name} output={tuple(actual.shape)} "
        f"path={getattr(op.kernel, 'path_kind', None)} finite={finite} "
        f"max_abs={max_abs} contiguous={output.is_contiguous()}",
        flush=True,
    )
    return bool(passed)


def _run_nan_case():
    cpu_input = torch.full((1, 1, 9, 11), -3.0, dtype=torch.float16)
    cpu_input[0, 0, 4, 5] = float("nan")
    cpu_input[0, 0, 2, 3] = 7.0
    reference = F.max_pool2d(cpu_input, 3, 2, 1)
    op = MaxPool2dFwdOp(kernel_size=3, stride=2, padding=1, target="ascend")
    output = op(cpu_input.npu())
    torch.npu.synchronize()
    actual = output.cpu()
    nan_match = torch.equal(torch.isnan(actual), torch.isnan(reference))
    finite = torch.isfinite(reference)
    values_match = torch.equal(actual[finite], reference[finite])
    passed = nan_match and values_match
    print(
        f"[{'PRECISION_PASS' if passed else 'PRECISION_FAIL'}] sparse-nan "
        f"path={getattr(op.kernel, 'path_kind', None)} nan_match={nan_match} "
        f"finite_values_match={values_match}",
        flush=True,
    )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=("quick", "full"), default="full")
    args = parser.parse_args()

    cases = MANIFEST_CASES if args.level == "full" else [MANIFEST_CASES[0]]
    passed = True
    for shape, kernel, stride, padding, dilation, ceil_mode, label in cases:
        for dtype_name in DTYPES:
            passed &= _run_case(
                shape,
                kernel,
                stride,
                padding,
                dilation,
                ceil_mode,
                dtype_name,
                label,
            )

    if args.level == "full":
        passed &= _run_case(
            (3, 5, 17, 19),
            (3, 3),
            (2, 2),
            (1, 1),
            (2, 1),
            True,
            "float16",
            "nondiv-dilated-ceil",
        )
        passed &= _run_case(
            (1, 1, 9, 257),
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            False,
            "float16",
            "generic-wide-output",
        )
        passed &= _run_nan_case()

    if passed:
        print("Test Passed!", flush=True)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
