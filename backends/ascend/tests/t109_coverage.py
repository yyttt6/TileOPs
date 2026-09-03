#!/usr/bin/env python3
"""T109 fused row-normalization correctness, sentinel, and SOL coverage."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import torch_npu


WORKSPACE = Path(__file__).resolve().parents[2]
HARNESS = WORKSPACE / "tileops-ascend-harness"
sys.path.insert(0, str(HARNESS))

from harnesslib import (  # noqa: E402
    REPEATS,
    THRESHOLD,
    WARMUP,
    gate_case,
    get_op_roofline,
    git_revision,
    load_profile,
    npu_smi_info,
    paired_samples_us,
    sample_stats,
    sol_us,
    visible_devices,
    write_json,
)
import tileops_ascend  # noqa: E402,F401
from tileops.ops.norm.ada_layer_norm import AdaLayerNormFwdOp  # noqa: E402
from tileops.ops.norm.ada_layer_norm_zero import AdaLayerNormZeroFwdOp  # noqa: E402
from tileops.ops.norm.fused_add_layer_norm import FusedAddLayerNormFwdOp  # noqa: E402
from tileops.ops.norm.fused_add_rms_norm import FusedAddRMSNormFwdOp  # noqa: E402


OPS = {
    "FusedAddRMSNormFwdOp": FusedAddRMSNormFwdOp,
    "FusedAddLayerNormFwdOp": FusedAddLayerNormFwdOp,
    "AdaLayerNormFwdOp": AdaLayerNormFwdOp,
    "AdaLayerNormZeroFwdOp": AdaLayerNormZeroFwdOp,
}
CASES = (
    ((3, 129), torch.float16, "r104-minimal-tail"),
    ((129, 197), torch.float16, "double-tail"),
    ((257, 300), torch.bfloat16, "bf16-nondiv"),
    ((1024, 1152), torch.float16, "dit-xl-2"),
)


def _tolerance(name: str, dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 1.6e-2, 1.6e-2
    if name == "FusedAddRMSNormFwdOp":
        return 1.0e-2, 1.0e-2
    return 1.0e-3, 1.0e-3


def _make(name: str, shape: tuple[int, int], dtype: torch.dtype, seed: int):
    torch.manual_seed(seed)
    tensors = [torch.randn(shape, dtype=dtype) for _ in range(4)]
    x, first, second, third = tensors
    if name == "FusedAddRMSNormFwdOp":
        weight = torch.randn(shape[-1], dtype=dtype)
        return (x, first, weight)
    if name == "FusedAddLayerNormFwdOp":
        weight = torch.randn(shape[-1], dtype=dtype)
        bias = torch.randn(shape[-1], dtype=dtype)
        return (x, first, weight, bias)
    if name == "AdaLayerNormFwdOp":
        return (x, first, second)
    return (x, first, second, third)


def _reference(name: str, args, eps: float):
    if name.startswith("FusedAdd"):
        x, residual, weight, *bias = args
        add = (x.float() + residual.float()).to(x.dtype)
        if name == "FusedAddRMSNormFwdOp":
            denom = torch.sqrt(add.float().square().mean(-1, keepdim=True) + eps)
            y = (add.float() / denom * weight.float()).to(x.dtype)
        else:
            y = F.layer_norm(
                add.float(), (add.shape[-1],), weight.float(), bias[0].float(), eps
            ).to(x.dtype)
        return y, add
    x, scale, shift, *gate = args
    norm = F.layer_norm(x.float(), (x.shape[-1],), None, None, eps)
    y = scale.float() * norm + shift.float()
    if gate:
        y = gate[0].float() * y
    return y.to(x.dtype)


def correctness_case(name: str, shape, dtype, label, seed: int):
    eps = 1.0e-6 if name == "FusedAddRMSNormFwdOp" else 1.0e-5
    cpu_args = _make(name, shape, dtype, seed)
    npu_args = tuple(t.npu() for t in cpu_args)
    op = OPS[name](target="ascend", eps=eps)
    # Compile once, then prime the adapter's out_idx allocations for the second
    # launch.  This checks actual returned buffers without changing the kernel ABI.
    actual1 = op(*npu_args)
    sentinel = float(torch.tensor(12345.0, dtype=dtype).item())
    original_empty = torch.empty

    def sentinel_empty(*args, **kwargs):
        tensor = original_empty(*args, **kwargs)
        if tensor.device.type == "npu" and tensor.dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            tensor.fill_(sentinel)
            torch.npu.synchronize()
        return tensor

    torch.empty = sentinel_empty
    try:
        actual2 = op(*npu_args)
    finally:
        torch.empty = original_empty
    torch.npu.synchronize()
    expected = _reference(name, cpu_args, eps)
    got_seq = actual1 if isinstance(actual1, tuple) else (actual1,)
    got2_seq = actual2 if isinstance(actual2, tuple) else (actual2,)
    ref_seq = expected if isinstance(expected, tuple) else (expected,)
    output_names = ("output", "residual_out") if len(got_seq) == 2 else ("output",)
    atol, rtol = _tolerance(name, dtype)
    checks = {}
    passed = True
    max_abs = 0.0
    for output_name, got_npu, got2_npu, ref in zip(output_names, got_seq, got2_seq, ref_seq):
        got, got2 = got_npu.cpu(), got2_npu.cpu()
        delta = (got.float() - ref.float()).abs()
        error = float(delta.max())
        finite = bool(torch.isfinite(got).all() and torch.isfinite(got2).all())
        stable = bool(torch.equal(got, got2))
        correct = bool(torch.allclose(got, ref, atol=atol, rtol=rtol))
        sentinel_count = int((got2 == sentinel).sum().item())
        written = sentinel_count == 0
        checks[output_name] = {
            "max_abs_err": error,
            "finite": finite,
            "double_launch_stable": stable,
            "sentinel_value": sentinel,
            "sentinel_count": sentinel_count,
            "sentinel_written": written,
            "passed": correct and finite and stable and written,
        }
        max_abs = max(max_abs, error)
        passed = passed and checks[output_name]["passed"]
    result = {
        "shape": {"input": list(shape), "out": list(shape)},
        "dtype": str(dtype).removeprefix("torch."),
        "label": label,
        "params": {"eps": eps},
        "max_abs_err": max_abs,
        "tolerance": {"atol": atol, "rtol": rtol},
        "output_sentinel_checks": checks,
        "finite": all(v["finite"] for v in checks.values()),
        "passed": passed,
    }
    print(f"correctness {name} {label}: passed={passed} max_abs={max_abs} checks={checks}", flush=True)
    if not passed:
        raise AssertionError(result)
    return result


def perf_case(name: str):
    shape, dtype = (129, 197), torch.float16
    eps = 1.0e-6 if name == "FusedAddRMSNormFwdOp" else 1.0e-5
    cpu_args = _make(name, shape, dtype, 11900)
    args = tuple(t.npu() for t in cpu_args)
    op = OPS[name](target="ascend", eps=eps)
    tested = lambda: op(*args)
    if name == "FusedAddRMSNormFwdOp":
        x, residual, weight = args
        baseline = lambda: (
            F.rms_norm((x + residual), (shape[-1],), weight, eps),
            x + residual,
        )
    elif name == "FusedAddLayerNormFwdOp":
        x, residual, weight, bias = args
        baseline = lambda: (
            F.layer_norm(x + residual, (shape[-1],), weight, bias, eps),
            x + residual,
        )
    elif name == "AdaLayerNormFwdOp":
        x, scale, shift = args
        baseline = lambda: scale * F.layer_norm(x, (shape[-1],), None, None, eps) + shift
    else:
        x, scale, shift, gate = args
        baseline = lambda: gate * (scale * F.layer_norm(x, (shape[-1],), None, None, eps) + shift)
    tested()
    baseline()
    torch.npu.synchronize()
    flops, nbytes, source, roofline_error = get_op_roofline(op)
    tested_samples, baseline_samples = paired_samples_us(
        tested, baseline, torch.npu.synchronize, warmup=WARMUP, repeats=REPEATS
    )
    tested_stats = sample_stats(tested_samples)
    baseline_stats = sample_stats(baseline_samples)
    tested_us, baseline_us = tested_stats["median"], baseline_stats["median"]
    profile = load_profile()
    sol_hw, bound_hw = sol_us(flops, nbytes, "float16", profile, "peak")
    sol_achieved, bound_achieved = sol_us(flops, nbytes, "float16", profile, "achieved")
    tested_gate, baseline_gate = gate_case(tested_us, sol_hw), gate_case(baseline_us, sol_hw)
    gate = "pass" if tested_gate == baseline_gate == "pass" else "VIOLATION"
    result = {
        "shape": {"input": list(shape), "out": list(shape)},
        "dtype": "float16",
        "params": {"eps": eps},
        "tileops_us": tested_us,
        "baseline_us": baseline_us,
        "ratio": baseline_us / tested_us,
        "flops": flops,
        "bytes": nbytes,
        "roofline_source": source,
        "roofline_error": roofline_error,
        "sol_us_hw": sol_hw,
        "sol_bound_hw": bound_hw,
        "sol_us_achieved": sol_achieved,
        "sol_bound_achieved": bound_achieved,
        "efficiency_vs_hw_peak": sol_hw / tested_us,
        "efficiency_vs_vendor_achieved": sol_achieved / tested_us,
        "baseline_efficiency_vs_hw_peak": sol_hw / baseline_us,
        "baseline_efficiency_vs_vendor_achieved": sol_achieved / baseline_us,
        "sol_gate": gate,
        "sol_gate_detail": {"tileops": tested_gate, "baseline": baseline_gate},
        "timing_samples_us": {"tileops": tested_samples, "baseline": baseline_samples},
        "timing_stats_us": {"tileops": tested_stats, "baseline": baseline_stats},
    }
    print(f"perf {name}: tileops_us={tested_us} baseline_us={baseline_us} sol_gate={gate}", flush=True)
    return result


def run(name: str, out_dir: Path, report_dir: Path):
    correctness = [
        correctness_case(name, shape, dtype, label, 10900 + index)
        for index, (shape, dtype, label) in enumerate(CASES)
    ]
    perf = [perf_case(name)]
    all_correct = all(case["passed"] for case in correctness)
    all_sol = all(case["sol_gate"] == "pass" for case in perf)
    ratio_min = min(case["ratio"] for case in perf)
    perf_ok = all_correct and all_sol and ratio_min >= THRESHOLD
    payload = {
        "schema_version": "1.0.0",
        "op": name,
        "target": "ascend",
        "status": "perf_ok" if perf_ok else ("perf_measured" if all_correct and all_sol else "blocked"),
        "device": {"soc": "ascend910b1", "visible_devices": visible_devices(), "npu_smi": npu_smi_info()},
        "correctness": {
            "passed": all_correct,
            "cmd": f"cd {WORKSPACE} && export ASCEND_RT_VISIBLE_DEVICES=3 && python -u tileops-ascend/tests/t109_coverage.py {name}",
            "per_case": correctness,
        },
        "perf": {
            "timing": "host monotonic clock with pre/post NPU synchronization and paired alternating order",
            "warmup": WARMUP,
            "repeats": REPEATS,
            "statistic": "median",
            "l2_flushed": False,
            "cases": perf,
            "ratio_min": ratio_min,
        },
        "baseline": {
            "tier": "vendor",
            "library": "Torch-NPU eager normalization expression",
            "repo_commit": torch.__version__,
            "source_path": str(Path(torch.__file__).resolve()),
            "build_cmd": "system-installed torch/torch_npu; no build in T109",
            "semantic_match": "exact",
            "fusion_match": "baseline expression is not single-launch fused",
        },
        "verdict": {
            "perf_ok": perf_ok,
            "threshold": THRESHOLD,
            "reason": "ratio and SOL gates passed" if perf_ok else "correctness/SOL passed; ratio below threshold",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "harness_commit": git_revision(HARNESS),
    }
    write_json(out_dir / f"{name}.json", payload)
    write_json(report_dir / f"{name}.json", payload)
    if not all_correct or not all_sol:
        raise AssertionError(f"{name}: correctness={all_correct} sol={all_sol}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("op", choices=tuple(OPS))
    parser.add_argument("--out", type=Path, default=HARNESS / "coverage")
    parser.add_argument(
        "--report-out",
        type=Path,
        default=WORKSPACE / "docs/reports/R109-data/coverage",
    )
    args = parser.parse_args()
    run(args.op, args.out, args.report_out)
