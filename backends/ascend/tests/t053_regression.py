"""Fail-closed tail regression for the T031 and R039 operator sets."""

from __future__ import annotations

import json

import tilelang
import torch

from tileops.ops.elementwise.arithmetic import (
    DivFwdOp,
    LerpFwdOp,
    MaximumFwdOp,
    MinimumFwdOp,
    MulFwdOp,
    PowFwdOp,
    SubFwdOp,
)
from tileops.ops.elementwise.comparison import (
    EqFwdOp,
    GeFwdOp,
    GtFwdOp,
    LeFwdOp,
    LtFwdOp,
    NeFwdOp,
)
from tileops.ops.elementwise.logical import LogicalAndFwdOp, LogicalOrFwdOp

import tileops_ascend  # noqa: F401
from tileops_ascend._registry import REGISTERED
from tileops_ascend.kernels import (
    elementwise_binary,
    elementwise_binary_batch,
    elementwise_predicate,
)


N = 4097
FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
NUMERIC = (
    ("SubFwdOp", SubFwdOp, lambda a, b: torch.sub(a, b, alpha=2.5), {"alpha": 2.5}),
    ("MulFwdOp", MulFwdOp, torch.mul, {}),
    ("DivFwdOp", DivFwdOp, torch.div, {}),
    ("PowFwdOp", PowFwdOp, torch.pow, {}),
    ("LerpFwdOp", LerpFwdOp, lambda a, b: torch.lerp(a, b, 0.25), {"weight": 0.25}),
    ("MaximumFwdOp", MaximumFwdOp, torch.maximum, {}),
    ("MinimumFwdOp", MinimumFwdOp, torch.minimum, {}),
)
PREDICATES = (
    ("EqFwdOp", EqFwdOp, torch.eq),
    ("NeFwdOp", NeFwdOp, torch.ne),
    ("GtFwdOp", GtFwdOp, torch.gt),
    ("GeFwdOp", GeFwdOp, torch.ge),
    ("LtFwdOp", LtFwdOp, torch.lt),
    ("LeFwdOp", LeFwdOp, torch.le),
    ("LogicalAndFwdOp", LogicalAndFwdOp, torch.logical_and),
    ("LogicalOrFwdOp", LogicalOrFwdOp, torch.logical_or),
)


def _clear() -> None:
    elementwise_binary._compile_binary.cache_clear()
    elementwise_binary_batch._compile_batch.cache_clear()
    elementwise_predicate._compile_predicate.cache_clear()
    tilelang.cache.clear_cache()


def _numeric_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.arange(N, dtype=torch.float32)
    a = (0.25 + (index.remainder(29) + 1) / 16).to(dtype)
    b = (0.5 + (index.remainder(17) + 1) / 16).to(dtype)
    return a, b


def _predicate_inputs(
    dtype: torch.dtype, logical: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.arange(N, dtype=torch.int32)
    if dtype == torch.bool:
        return (index.remainder(2) == 0), (index.remainder(3) != 0)
    a = index.remainder(7).sub(3).to(dtype)
    b = index.remainder(5).sub(2).to(dtype)
    if logical:
        a[::4] = 0
        b[::5] = 0
    return a, b


def _run(
    name: str,
    cls,
    reference,
    params: dict,
    dtype: torch.dtype,
    a_cpu: torch.Tensor,
    b_cpu: torch.Tensor,
) -> dict:
    _clear()
    ref = reference(a_cpu, b_cpu)
    a, b = a_cpu.npu(), b_cpu.npu()
    op = cls(target="ascend", **params)
    first = op(a, b)
    torch.npu.synchronize()
    first_cpu = first.cpu()
    second = op(a, b)
    torch.npu.synchronize()
    second_cpu = second.cpu()
    if ref.dtype == torch.bool:
        reference_passed = torch.equal(second_cpu, ref)
        tail_equal = bool(second_cpu[-1] == ref[-1])
        max_abs_err = 0.0 if reference_passed else 1.0
        finite = True
    else:
        tolerance = 1.6e-2 if dtype == torch.bfloat16 else 1e-3
        reference_passed = bool(
            torch.allclose(
                second_cpu.float(), ref.float(), atol=tolerance, rtol=tolerance
            )
        )
        tail_equal = bool(
            torch.isclose(
                second_cpu[-1].float(), ref[-1].float(), atol=tolerance, rtol=tolerance
            )
        )
        max_abs_err = float((second_cpu.float() - ref.float()).abs().max())
        finite = bool(
            torch.isfinite(first_cpu).all() and torch.isfinite(second_cpu).all()
        )
    row = {
        "op": name,
        "dtype": str(dtype).removeprefix("torch."),
        "shape": [N],
        "python_lru_cleared": True,
        "tilelang_disk_cleared": True,
        "returned_distinct_from_inputs": bool(
            first.data_ptr() not in {a.data_ptr(), b.data_ptr()}
            and second.data_ptr() not in {a.data_ptr(), b.data_ptr()}
        ),
        "double_run_equal": torch.equal(first_cpu, second_cpu),
        "tail_equal": tail_equal,
        "finite": finite,
        "reference_passed": reference_passed,
        "max_abs_err": max_abs_err,
    }
    row["passed"] = all(
        row[key]
        for key in (
            "returned_distinct_from_inputs",
            "double_run_equal",
            "tail_equal",
            "finite",
            "reference_passed",
        )
    )
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


def main() -> int:
    expected = {name for name, *_ in NUMERIC + PREDICATES}
    missing = sorted(expected - REGISTERED.keys())
    if missing:
        print(json.dumps({"missing_registrations": missing}), flush=True)
        return 2
    rows = []
    for name, cls, reference, params in NUMERIC:
        for dtype in FLOAT_DTYPES:
            a_cpu, b_cpu = _numeric_inputs(dtype)
            rows.append(_run(name, cls, reference, params, dtype, a_cpu, b_cpu))
    for name, cls, reference in PREDICATES:
        dtypes = (
            (torch.bool, *FLOAT_DTYPES) if name.startswith("Logical") else FLOAT_DTYPES
        )
        for dtype in dtypes:
            a_cpu, b_cpu = _predicate_inputs(dtype, name.startswith("Logical"))
            rows.append(_run(name, cls, reference, {}, dtype, a_cpu, b_cpu))
    summary = {
        "cases": len(rows),
        "ops": len({row["op"] for row in rows}),
        "passed": all(row["passed"] for row in rows),
        "failures": [row for row in rows if not row["passed"]],
    }
    print(json.dumps({"SUMMARY": summary}, sort_keys=True), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
