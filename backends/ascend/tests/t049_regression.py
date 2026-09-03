"""Regression checks for existing archetype-2 operators after T049."""

from __future__ import annotations

import torch

import tileops_ascend  # noqa: F401
import tileops_ascend.families.reduction  # noqa: F401
import tileops_ascend.families.reduction_count_nonzero  # noqa: F401
from tileops.ops.reduction.logical_reduce import AllFwdOp, AnyFwdOp, CountNonzeroFwdOp
from tileops.ops.reduction.reduce import MeanFwdOp
from tileops_ascend.kernels.reduction import build_reduction_kernel


def _mean_case(shape, dtype, dim, label) -> bool:
    torch.manual_seed(490)
    host = torch.randn(shape, dtype=torch.float32).mul_(0.5).to(dtype)
    reference = host.float().mean(dim=dim).to(dtype)
    output_npu = MeanFwdOp(dim=dim, target="ascend")(host.npu())
    torch.npu.synchronize()
    output = output_npu.cpu()
    tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-3
    passed = bool(
        torch.isfinite(output).all()
        and torch.allclose(
            output.float(), reference.float(), atol=tolerance, rtol=tolerance
        )
    )
    print(label, "passed", passed, "shape", tuple(output.shape))
    return passed


def _count_case(shape, dtype, dim, label) -> bool:
    torch.manual_seed(491)
    host = torch.randint(-2, 3, shape, dtype=torch.int32).to(dtype)
    reference = torch.count_nonzero(host, dim=dim)
    output_npu = CountNonzeroFwdOp(dim=dim, target="ascend")(host.npu())
    torch.npu.synchronize()
    output = output_npu.cpu()
    passed = output.dtype == torch.int64 and torch.equal(output, reference)
    print(label, "passed", passed, "shape", tuple(output.shape))
    return passed


def main() -> None:
    checks = [
        _mean_case((129, 300), torch.float16, -1, "mean-nondiv"),
        _mean_case((2048, 4096), torch.bfloat16, -1, "mean-large-multicore"),
        _mean_case((4, 128, 4096), torch.float16, (0, 2), "mean-multiaxis"),
        _count_case((17, 257), torch.float16, -1, "count-nondiv"),
        _count_case((2048, 4096), torch.float16, -1, "count-large-multicore"),
        _count_case((4, 128, 4096), torch.float16, (0, 2), "count-multiaxis"),
    ]
    sample = torch.ones((8, 16), dtype=torch.float16)
    for op_cls in (AllFwdOp, AnyFwdOp):
        try:
            op_cls(dim=-1, target="ascend")(sample.npu())
        except TypeError as exc:
            print(op_cls.__name__, "float16-fail-closed", str(exc))
        else:
            print(op_cls.__name__, "float16-fail-closed", False)
            checks.append(False)
    for invalid_dim in (2, -3):
        try:
            build_reduction_kernel(
                (8, 16),
                torch.float16,
                invalid_dim,
                False,
                op_kind="sum",
                op_name="SumFwdOp",
            )
        except ValueError as exc:
            print("invalid-dim-fail-closed", invalid_dim, str(exc))
        else:
            print("invalid-dim-fail-closed", invalid_dim, False)
            checks.append(False)
    passed = all(checks)
    print("regression_passed", passed)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
