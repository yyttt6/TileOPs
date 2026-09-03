"""T105 correctness coverage for the three binary bitwise ops and PReLU."""

from __future__ import annotations

import json
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.modules.setdefault(
    "tileops_ascend.families.two_pass", types.ModuleType("tileops_ascend.families.two_pass")
)

from tileops.ops.elementwise import (  # noqa: E402
    BitwiseAndFwdOp,
    BitwiseOrFwdOp,
    BitwiseXorFwdOp,
    PreluFwdOp,
)
from tileops_ascend.kernels import elementwise_mixed as mixed  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tileops-ascend-harness/coverage"


def _device_info() -> dict[str, str]:
    import subprocess

    smi = subprocess.run(
        ["npu-smi", "info"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False,
    ).stdout
    return {
        "soc": "ascend910b1",
        "visible_devices": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        "npu_smi": smi,
    }


def _skeleton(op: str, cases: list[dict], passed: bool) -> dict:
    return {
        "schema_version": "1.0.0",
        "op": op,
        "target": "ascend",
        "status": "correct" if passed else "blocked",
        "device": _device_info(),
        "correctness": {
            "passed": passed,
            "cmd": "python -u tileops-ascend/tests/t105_coverage.py",
            "per_case": cases,
        },
        "perf": {
            "timing": "not measured (T105 correctness coverage)",
            "warmup": 0,
            "repeats": 0,
            "statistic": "none",
            "l2_flushed": False,
            "cases": [],
            "ratio_min": None,
        },
        "baseline": {
            "tier": "vendor",
            "library": "not measured",
            "repo_commit": "unmeasured",
            "source_path": "not measured",
            "build_cmd": "not measured",
            "semantic_match": "exact",
            "fusion_match": "exact",
        },
        "verdict": {
            "perf_ok": False,
            "threshold": 0.70,
            "reason": "T105 correctness-only coverage; performance not measured",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "harness_commit": "uncommitted",
    }


def _run_bitwise(op_name: str, cls: type, fn) -> dict:
    cases = []
    for dtype in (torch.bool, torch.int32, torch.int64):
        for shape, other_shape, label in (
            ((4097,), (4097,), "tail-nondivisible"),
            ((2, 3, 683), (1, 3, 1), "broadcast-tail-nondivisible"),
        ):
            torch.manual_seed(105 + len(cases))
            if dtype == torch.bool:
                a = torch.randint(0, 2, shape, device="npu", dtype=torch.int8).bool()
                b = torch.randint(0, 2, other_shape, device="npu", dtype=torch.int8).bool()
            else:
                a = torch.randint(-17, 18, shape, device="npu", dtype=dtype)
                b = torch.randint(-17, 18, other_shape, device="npu", dtype=dtype)
            op = cls(target="ascend")
            try:
                got = op(a, b)
                torch.npu.synchronize()
                ref = fn(a.cpu(), b.cpu())
                delta = (got.cpu().to(torch.int64) - ref.to(torch.int64)).abs()
                max_abs = int(delta.max()) if delta.numel() else 0
                ok = bool(torch.equal(got.cpu(), ref))
                error = None
            except Exception as exc:  # noqa: BLE001 - preserve per-case evidence
                ok, max_abs, error = False, None, f"{type(exc).__name__}: {exc}"
            cases.append({
                "shape": {"input": list(shape), "other": list(other_shape),
                          "out": list(torch.broadcast_shapes(shape, other_shape))},
                "dtype": str(dtype).replace("torch.", ""),
                "label": label,
                "passed": ok,
                "max_abs_err": max_abs,
                "non_divisible": True,
                "error": error,
            })
    passed = all(case["passed"] for case in cases)
    return _skeleton(op_name, cases, passed)


def _run_prelu() -> dict:
    cases = []
    for dtype in (torch.float16, torch.bfloat16):
        for shape, weight_shape, label in (
            ((1, 4, 1025), (4,), "channel-tail-nondivisible"),
            ((2, 3, 700), (3,), "channel-broadcast-tail-nondivisible"),
        ):
            torch.manual_seed(105 + len(cases))
            x = torch.randn(shape, device="npu", dtype=dtype)
            w = torch.rand(weight_shape, device="npu", dtype=dtype) * 0.5
            op = PreluFwdOp(target="ascend")
            try:
                got = op(x, w)
                torch.npu.synchronize()
                ref = torch.nn.functional.prelu(x.float().cpu(), w.float().cpu()).to(dtype)
                delta = (got.cpu().float() - ref.float()).abs()
                atol = rtol = 1.6e-2 if dtype == torch.bfloat16 else 1.0e-3
                ok = bool(torch.allclose(got.cpu().float(), ref.float(), atol=atol, rtol=rtol))
                max_abs = float(delta.max())
                error = None
            except Exception as exc:  # noqa: BLE001 - preserve per-case evidence
                ok, max_abs, error = False, None, f"{type(exc).__name__}: {exc}"
            cases.append({
                "shape": {"input": list(shape), "weight": list(weight_shape), "out": list(shape)},
                "dtype": str(dtype).replace("torch.", ""),
                "label": label,
                "passed": ok,
                "max_abs_err": max_abs,
                "tolerance": {"atol": atol, "rtol": rtol} if error is None else None,
                "non_divisible": True,
                "error": error,
            })
    passed = all(case["passed"] for case in cases)
    return _skeleton("PreluFwdOp", cases, passed)


def main() -> None:
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES must be set")
    specs = (
        ("BitwiseAndFwdOp", BitwiseAndFwdOp, torch.bitwise_and),
        ("BitwiseOrFwdOp", BitwiseOrFwdOp, torch.bitwise_or),
        ("BitwiseXorFwdOp", BitwiseXorFwdOp, torch.bitwise_xor),
    )
    results = [_run_bitwise(name, cls, fn) for name, cls, fn in specs]
    results.append(_run_prelu())
    artifact_dir = ROOT / "docs/reports/R105-data/artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("and", "or", "xor"):
        for a_shape, b_shape, tag in (
            ((4097,), (4097,), "4097"),
            ((2, 3, 683), (1, 3, 1), "2x3x683-broadcast"),
        ):
            _, _, out_shape, a_strides, b_strides = mixed._shape_info(a_shape, b_shape)
            a_numel = int(torch.tensor(a_shape).prod())
            b_numel = int(torch.tensor(b_shape).prod())
            n = int(torch.tensor(out_shape).prod())
            kernel = mixed._compile_bitwise(
                a_numel, b_numel, n, out_shape, a_strides, b_strides, "int32", kind
            )
            (artifact_dir / f"bitwise-{kind}-{tag}-int32.cu").write_text(
                kernel.get_kernel_source(), encoding="utf-8"
            )
    for shape, channels, inner, tag in (
        ((1, 4, 1025), 4, 1025, "1x4x1025"),
        ((2, 3, 700), 3, 700, "2x3x700"),
    ):
        n = int(torch.tensor(shape).prod())
        kernel = mixed._compile_prelu(
            n, channels, shape, "float16", "float16", "float16", channels, inner
        )
        (artifact_dir / f"prelu-{tag}-fp16.cu").write_text(
            kernel.get_kernel_source(), encoding="utf-8"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    for result in results:
        (OUT / f"{result['op']}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps({"op": result["op"], "status": result["status"],
                          "passed": result["correctness"]["passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
