"""Focused NPU probe for T031's batch binary template."""

from __future__ import annotations

import argparse

import torch

from tileops_ascend.families import elementwise as family


OPS = {
    "MulFwdOp": (family.build_mul, torch.mul),
    "DivFwdOp": (family.build_div, torch.div),
    "PowFwdOp": (family.build_pow, torch.pow),
    "LerpFwdOp": (family.build_lerp, lambda a, b: torch.lerp(a, b, 0.5)),
    "MaximumFwdOp": (family.build_maximum, torch.maximum),
    "MinimumFwdOp": (family.build_minimum, torch.minimum),
}


def _inputs(shape_a, shape_b, dtype):
    if dtype.is_floating_point:
        a = torch.rand(shape_a, dtype=dtype) * 2 + 0.25
        b = torch.rand(shape_b, dtype=dtype) * 2 + 0.25
    else:
        a = torch.randint(-31, 32, shape_a, dtype=dtype)
        b = torch.randint(-31, 32, shape_b, dtype=dtype)
    return a, b


def run_one(name):
    builder, ref_fn = OPS[name]
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for label, shape_a, shape_b in (
            ("nondiv", (17, 257), (17, 257)),
            ("broadcast", (4, 7, 17), (7, 1)),
        ):
            cpu_a, cpu_b = _inputs(shape_a, shape_b, dtype)
            ref = ref_fn(cpu_a, cpu_b)
            a, b = cpu_a.npu(), cpu_b.npu()
            kernel = builder(a, b)
            out = kernel(a, b)
            torch.npu.synchronize()
            got = out.cpu()
            tol = 1.6e-2 if dtype == torch.bfloat16 else 1e-3
            equal = torch.allclose(got.float(), ref.float(), rtol=tol, atol=tol)
            finite = bool(torch.isfinite(got).all())
            print(
                f"{name} {label}: equal={equal} finite={finite} shape={tuple(got.shape)} dtype={got.dtype}"
            )
            if not equal or not finite:
                raise AssertionError(f"{name} {label} {dtype} failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ops", nargs="*", default=list(OPS))
    args = parser.parse_args()
    torch.manual_seed(31)
    for name in args.ops:
        run_one(name)


if __name__ == "__main__":
    main()
