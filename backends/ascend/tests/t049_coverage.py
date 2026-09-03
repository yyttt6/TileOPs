"""Generate fresh-process correctness coverage for T049 reductions."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from harnesslib import (
    THRESHOLD,
    git_revision,
    npu_smi_info,
    validate_result,
    visible_devices,
)
from tileops.ops.reduction.logical_reduce import AllFwdOp, AnyFwdOp
from tileops.ops.reduction.reduce import AmaxFwdOp, AminFwdOp, SumFwdOp
from tileops.ops.reduction.vector_norm import InfNormFwdOp, L1NormFwdOp, L2NormFwdOp

import tileops_ascend  # noqa: F401
import tileops_ascend.families.reduction  # noqa: F401
from tileops_ascend._registry import REGISTERED


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CaseSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype
    label: str
    dim: int | tuple[int, ...] | None = None
    keepdim: bool = False


OP_CLASSES = {
    "SumFwdOp": SumFwdOp,
    "AmaxFwdOp": AmaxFwdOp,
    "AminFwdOp": AminFwdOp,
    "AllFwdOp": AllFwdOp,
    "AnyFwdOp": AnyFwdOp,
    "L1NormFwdOp": L1NormFwdOp,
    "L2NormFwdOp": L2NormFwdOp,
    "InfNormFwdOp": InfNormFwdOp,
}

COMMON_REDUCE = (
    CaseSpec((2048, 4096), torch.float16, "hidden-state-reduce-fp16"),
    CaseSpec((2048, 4096), torch.bfloat16, "hidden-state-reduce-bf16"),
    CaseSpec((64, 32768), torch.bfloat16, "long-seq-reduce"),
    CaseSpec((4, 128, 4096), torch.float16, "3d-multidim-reduce", (0, 2)),
    CaseSpec((129, 300), torch.float16, "nondiv-multicore", -1),
    CaseSpec((129, 300), torch.float32, "nondiv-multicore-fp32", -1),
)

CASES = {
    "SumFwdOp": COMMON_REDUCE
    + (
        CaseSpec((2048, 4096), torch.bfloat16, "hidden-state-reduce-dim0", 0),
        CaseSpec((2048, 4096), torch.bfloat16, "hidden-state-reduce-keepdim", -1, True),
    ),
    "AmaxFwdOp": COMMON_REDUCE,
    "AminFwdOp": COMMON_REDUCE,
    "L1NormFwdOp": COMMON_REDUCE,
    "L2NormFwdOp": COMMON_REDUCE,
    "InfNormFwdOp": COMMON_REDUCE,
    "AllFwdOp": (
        CaseSpec((32, 4096), torch.bool, "mask-validation-4k"),
        CaseSpec((32, 32768), torch.bool, "mask-validation-32k"),
        CaseSpec((4, 128, 4096), torch.bool, "3d-multidim-reduce", (0, 2)),
        CaseSpec((129, 300), torch.bool, "nondiv-multicore", -1),
    ),
    "AnyFwdOp": (
        CaseSpec((32, 4096), torch.bool, "mask-validation-4k"),
        CaseSpec((32, 32768), torch.bool, "mask-validation-32k"),
        CaseSpec((4, 128, 4096), torch.bool, "3d-multidim-reduce", (0, 2)),
        CaseSpec((129, 300), torch.bool, "nondiv-multicore", -1),
    ),
}


def _revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip() or "unversioned"


def _input(name: str, spec: CaseSpec, index: int) -> torch.Tensor:
    torch.manual_seed(4900 + index)
    if spec.dtype == torch.bool:
        value = torch.randint(0, 2, spec.shape, dtype=torch.bool)
        if spec.dim == -1 and spec.shape[0] >= 2:
            value[0].fill_(True)
            value[1].fill_(False)
        return value
    scale = 0.5
    if name == "L1NormFwdOp" and spec.dtype == torch.float16 and spec.dim is None:
        scale = min(scale, 30_000.0 / float(torch.tensor(spec.shape).prod()))
    return torch.randn(spec.shape, dtype=torch.float32).mul_(scale).to(spec.dtype)


def _reference(name: str, x: torch.Tensor, spec: CaseSpec):
    if name == "AllFwdOp":
        return (
            torch.all(x)
            if spec.dim is None
            else torch.all(x, dim=spec.dim, keepdim=spec.keepdim)
        )
    if name == "AnyFwdOp":
        return (
            torch.any(x)
            if spec.dim is None
            else torch.any(x, dim=spec.dim, keepdim=spec.keepdim)
        )
    value = x.float()
    if name == "SumFwdOp":
        result = torch.sum(value, dim=spec.dim, keepdim=spec.keepdim)
    elif name == "AmaxFwdOp":
        result = torch.amax(value, dim=spec.dim, keepdim=spec.keepdim)
    elif name == "AminFwdOp":
        result = torch.amin(value, dim=spec.dim, keepdim=spec.keepdim)
    else:
        order = {"L1NormFwdOp": 1, "L2NormFwdOp": 2, "InfNormFwdOp": float("inf")}[name]
        result = torch.linalg.vector_norm(
            value,
            ord=order,
            dim=spec.dim,
            keepdim=spec.keepdim,
        )
    return result.to(x.dtype)


def _tolerance(name: str, dtype: torch.dtype) -> dict[str, float]:
    if dtype == torch.float32:
        value = 1e-5 if name.endswith("NormFwdOp") else 1e-4
    elif dtype == torch.bfloat16:
        value = 1e-2
    else:
        value = 1e-3
    return {"atol": value, "rtol": value}


def _run_case(name: str, spec: CaseSpec, index: int) -> dict:
    host = _input(name, spec, index)
    reference = _reference(name, host, spec)
    x = host.npu()
    op = OP_CLASSES[name](dim=spec.dim, keepdim=spec.keepdim, target="ascend")
    output_npu = op(x)
    torch.npu.synchronize()
    output = output_npu.cpu()
    distinct_output = output_npu.data_ptr() != x.data_ptr()
    if output.dtype == torch.bool:
        tolerance = {"atol": 0.0, "rtol": 0.0}
        passed = torch.equal(output, reference)
        max_abs = 0.0 if passed else 1.0
        finite = True
    else:
        tolerance = _tolerance(name, spec.dtype)
        delta = (output.float() - reference.float()).abs()
        finite_delta = delta[torch.isfinite(delta)]
        max_abs = float(finite_delta.max()) if finite_delta.numel() else 0.0
        finite = bool(torch.isfinite(output).all())
        passed = finite and torch.allclose(
            output.float(), reference.float(), **tolerance
        )
    params = {"dim": spec.dim, "keepdim": spec.keepdim}
    case = {
        "shape": {"input": list(spec.shape), "out": list(output.shape)},
        "dtype": str(spec.dtype).replace("torch.", ""),
        "label": spec.label,
        "params": params,
        "max_abs_err": max_abs,
        "tolerance": tolerance,
        "finite": finite,
        "distinct_output_storage": distinct_output,
        "passed": bool(passed and distinct_output),
    }
    print(
        f"{name} {spec.label}: passed={case['passed']} finite={finite} "
        f"max_abs={max_abs} out={tuple(output.shape)}",
        flush=True,
    )
    return case


def run_op(name: str, out_dir: Path, report_dir: Path) -> None:
    if name not in OP_CLASSES:
        raise ValueError(f"unknown T049 op {name!r}")
    if name not in REGISTERED:
        raise RuntimeError(f"{name} is not registered for target ascend")
    cases = [_run_case(name, spec, index) for index, spec in enumerate(CASES[name])]
    passed = all(case["passed"] for case in cases)
    command = (
        f"python -u tileops-ascend/tests/t049_coverage.py {name} "
        "--out tileops-ascend-harness/coverage "
        "--report-out docs/reports/R049-data/coverage"
    )
    result = {
        "schema_version": "1.0.0",
        "op": name,
        "target": "ascend",
        "status": "correct" if passed else "blocked",
        "device": {
            "soc": "ascend910b1",
            "visible_devices": visible_devices(),
            "npu_smi": npu_smi_info(),
        },
        "correctness": {"passed": passed, "cmd": command, "per_case": cases},
        "perf": {
            "timing": "not measured: T049 correctness batch",
            "warmup": 0,
            "repeats": 0,
            "statistic": "none",
            "l2_flushed": False,
            "cases": [],
            "ratio_min": None,
        },
        "baseline": {
            "tier": "vendor",
            "library": "CPU PyTorch reference (correctness only)",
            "repo_commit": _revision(ROOT / "TileOPs"),
            "source_path": str(Path(torch.__file__).resolve()),
            "build_cmd": "not applicable",
            "semantic_match": "exact",
            "fusion_match": "exact",
        },
        "verdict": {
            "perf_ok": False,
            "threshold": THRESHOLD,
            "reason": "correctness batch; performance not measured",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "harness_commit": git_revision(ROOT / "tileops-ascend-harness"),
        "implementation_commit": _revision(ROOT / "tileops-ascend"),
    }
    validate_result(result)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(payload, encoding="utf-8")
    (report_dir / f"{name}.json").write_text(payload, encoding="utf-8")
    print(f"{name}: correctness={passed} cases={len(cases)}")
    if not passed:
        raise AssertionError(f"{name} correctness failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("op", choices=tuple(OP_CLASSES))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run_op(args.op, args.out, args.report_out)


if __name__ == "__main__":
    main()
