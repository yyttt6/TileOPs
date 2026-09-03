#!/usr/bin/env python3
"""T075 Conv2d correctness, SOL, and canonical coverage on Ascend card 0."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import torch_npu


WORKSPACE = Path(__file__).resolve().parents[2]
HARNESS = WORKSPACE / "tileops-ascend-harness"
sys.path.insert(0, str(WORKSPACE / "TileOPs"))
sys.path.insert(0, str(HARNESS))

from harnesslib import (  # noqa: E402
    PROFILE_PATH,
    REPEATS,
    THRESHOLD,
    WARMUP,
    gate_case,
    git_revision,
    load_profile,
    npu_smi_info,
    paired_samples_us,
    repo_revision,
    sample_stats,
    sol_us,
    validate_result,
    visible_devices,
    write_json,
)
import tileops_ascend.families.convolution  # noqa: E402,F401
from tileops.ops.convolution import Conv2dFwdOp  # noqa: E402


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

CASES = [
    {
        "label": "default-dense-cube-fp16",
        "input_shape": (1, 16, 8, 8),
        "c_out": 32,
        "kernel": (3, 3),
        "dtype": "float16",
        "params": {},
        "bias": False,
        "tags": ["default", "padding", "cube", "fp16"],
    },
    {
        "label": "stride-padding-nondiv-cube-fp16",
        "input_shape": (1, 3, 10, 11),
        "c_out": 16,
        "kernel": (3, 3),
        "dtype": "float16",
        "params": {"stride": (2, 2), "padding": (1, 1)},
        "bias": False,
        "tags": ["stride", "padding", "nondiv-output", "cube", "k-tail"],
    },
    {
        "label": "dilation-cube-fp16",
        "input_shape": (1, 16, 9, 10),
        "c_out": 16,
        "kernel": (3, 3),
        "dtype": "float16",
        "params": {"padding": (2, 2), "dilation": (2, 2)},
        "bias": False,
        "tags": ["dilation", "padding", "cube"],
    },
    {
        "label": "grouped-direct-fp16",
        "input_shape": (1, 8, 9, 10),
        "c_out": 8,
        "kernel": (3, 3),
        "dtype": "float16",
        "params": {"padding": (1, 1), "groups": 4},
        "bias": False,
        "tags": ["groups", "direct"],
    },
    {
        "label": "depthwise-bias-direct-fp16",
        "input_shape": (1, 4, 8, 9),
        "c_out": 4,
        "kernel": (3, 3),
        "dtype": "float16",
        "params": {"padding": (1, 1), "groups": 4},
        "bias": True,
        "tags": ["groups", "depthwise", "bias", "direct"],
    },
    {
        "label": "dense-cube-bf16",
        "input_shape": (1, 16, 8, 8),
        "c_out": 16,
        "kernel": (3, 3),
        "dtype": "bfloat16",
        "params": {"padding": (1, 1)},
        "bias": False,
        "tags": ["bf16", "cube"],
    },
    {
        "label": "fp32-direct",
        "input_shape": (1, 2, 5, 6),
        "c_out": 3,
        "kernel": (3, 3),
        "dtype": "float32",
        "params": {"padding": (1, 1)},
        "bias": False,
        "tags": ["fp32", "direct"],
    },
    {
        "label": "large-grid-finite-depthwise-fp16",
        "input_shape": (16, 256, 56, 56),
        "c_out": 256,
        "kernel": (1, 1),
        "dtype": "float16",
        "params": {"groups": 256},
        "bias": False,
        "tags": ["large", "depthwise", "direct", "grid-stride", "finite"],
    },
]


def _normalized_params(case):
    params = {"stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1}
    params.update(case["params"])
    return params


def _cpu_data(case):
    dtype = DTYPES[case["dtype"]]
    n, c_in, h_in, w_in = case["input_shape"]
    groups = _normalized_params(case)["groups"]
    kernel_h, kernel_w = case["kernel"]
    generator = torch.Generator().manual_seed(1000 + CASES.index(case))
    x = torch.randn((n, c_in, h_in, w_in), generator=generator).mul_(0.25).to(dtype)
    weight = (
        torch.randn(
            (case["c_out"], c_in // groups, kernel_h, kernel_w),
            generator=generator,
        )
        .mul_(0.25)
        .to(dtype)
    )
    bias = (
        torch.randn((case["c_out"],), generator=generator).mul_(0.25).to(dtype)
        if case["bias"]
        else None
    )
    return x, weight, bias


def _reference(x, weight, bias, params, dtype):
    ref = F.conv2d(
        x.float(),
        weight.float(),
        None if bias is None else bias.float(),
        stride=params["stride"],
        padding=params["padding"],
        dilation=params["dilation"],
        groups=params["groups"],
    )
    return ref.to(dtype)


def _tolerance(dtype_name):
    if dtype_name == "float16":
        return 1e-3, 1e-3
    return 1.6e-2, 1.6e-2


def _correctness(case):
    params = _normalized_params(case)
    dtype = DTYPES[case["dtype"]]
    x_cpu, weight_cpu, bias_cpu = _cpu_data(case)
    ref = _reference(x_cpu, weight_cpu, bias_cpu, params, dtype)
    x = x_cpu.npu()
    weight = weight_cpu.npu()
    bias = None if bias_cpu is None else bias_cpu.npu()
    op = Conv2dFwdOp(target="ascend", **params)
    output = op(x, weight, bias)
    torch.npu.synchronize()
    actual = output.cpu()
    delta = (actual.float() - ref.float()).abs()
    denominator = ref.float().abs().clamp_min(1e-12)
    atol, rtol = _tolerance(case["dtype"])
    finite = bool(torch.isfinite(actual).all())
    grid_stride_passed = "grid-stride" not in case["tags"] or (
        op.kernel.logical_blocks > 65535
        and op.kernel.launch_blocks == 65535
        and op.kernel.grid_repeats > 1
    )
    passed = (
        finite
        and grid_stride_passed
        and bool(torch.allclose(actual, ref, atol=atol, rtol=rtol))
    )
    row = {
        "shape": {
            "input": list(case["input_shape"]),
            "weight": list(weight.shape),
            "out": list(output.shape),
            "params": params,
            "label": case["label"],
            "tags": case["tags"],
            "path_kind": op.kernel.path_kind,
            "logical_blocks": op.kernel.logical_blocks,
            "launch_blocks": op.kernel.launch_blocks,
            "grid_repeats": op.kernel.grid_repeats,
        },
        "dtype": case["dtype"],
        "max_abs_err": float(delta.max()) if delta.numel() else 0.0,
        "max_rel_err": float((delta / denominator).max()) if delta.numel() else 0.0,
        "tolerance": {"atol": atol, "rtol": rtol},
        "finite": finite,
        "grid_stride_passed": grid_stride_passed,
        "passed": passed,
    }
    print(
        f"[CORRECTNESS] {case['label']} path={op.kernel.path_kind} "
        f"finite={finite} max_abs={row['max_abs_err']:.8g} passed={passed}",
        flush=True,
    )
    return row, op, x, weight, bias


def _performance(case, op, x, weight, bias, profile):
    params = _normalized_params(case)

    def tested():
        return op(x, weight, bias)

    def baseline():
        return F.conv2d(x, weight, bias, **params)

    tested_samples, baseline_samples = paired_samples_us(
        tested, baseline, torch.npu.synchronize
    )
    tested_stats = sample_stats(tested_samples)
    baseline_stats = sample_stats(baseline_samples)
    tested_us = tested_stats["median"]
    baseline_us = baseline_stats["median"]
    flops, nbytes = op.eval_roofline()
    sol_hw, bound_hw = sol_us(flops, nbytes, case["dtype"], profile, "peak")
    sol_achieved, bound_achieved = sol_us(
        flops, nbytes, case["dtype"], profile, "achieved"
    )
    tested_gate = gate_case(tested_us, sol_hw)
    baseline_gate = gate_case(baseline_us, sol_hw)
    overall_gate = (
        "VIOLATION" if "VIOLATION" in {tested_gate, baseline_gate} else "pass"
    )
    row = {
        "shape": {
            "input": list(case["input_shape"]),
            "weight": list(weight.shape),
            "out": list(op.kernel.output_shape),
            "params": params,
            "label": case["label"],
            "path_kind": op.kernel.path_kind,
        },
        "dtype": case["dtype"],
        "tileops_us": tested_us,
        "baseline_us": baseline_us,
        "ratio": baseline_us / tested_us,
        "flops": flops,
        "bytes": nbytes,
        "roofline_source": "Conv2dFwdOp.eval_roofline()",
        "roofline_error": None,
        "sol_us_hw": sol_hw,
        "sol_bound_hw": bound_hw,
        "sol_us_achieved": sol_achieved,
        "sol_bound_achieved": bound_achieved,
        "efficiency_vs_hw_peak": sol_hw / tested_us,
        "efficiency_vs_vendor_achieved": sol_achieved / tested_us,
        "baseline_efficiency_vs_hw_peak": sol_hw / baseline_us,
        "baseline_efficiency_vs_vendor_achieved": sol_achieved / baseline_us,
        "sol_gate": overall_gate,
        "sol_gate_detail": {"tileops": tested_gate, "baseline": baseline_gate},
        "timing_samples_us": {
            "tileops": tested_samples,
            "baseline": baseline_samples,
        },
        "timing_stats_us": {
            "tileops": tested_stats,
            "baseline": baseline_stats,
        },
    }
    print(
        f"[MEASURED] {case['label']} tileops={tested_us:.3f}us "
        f"baseline={baseline_us:.3f}us ratio={row['ratio']:.6f} "
        f"sol_gate={overall_gate}",
        flush=True,
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0":
        raise RuntimeError("T075 requires ASCEND_RT_VISIBLE_DEVICES=0")

    profile = load_profile(PROFILE_PATH)
    smi = npu_smi_info()
    correctness = []
    perf = []
    for index, case in enumerate(CASES):
        row, op, x, weight, bias = _correctness(case)
        correctness.append(row)
        if not args.correctness_only and index in {0, 3, 5, 7}:
            perf.append(_performance(case, op, x, weight, bias, profile))
        del op, x, weight, bias
        torch.npu.empty_cache()

    correctness_passed = all(row["passed"] for row in correctness)
    all_sol_pass = bool(perf) and all(row["sol_gate"] == "pass" for row in perf)
    ratio_min = min((row["ratio"] for row in perf), default=None)
    perf_ok = bool(
        correctness_passed
        and all_sol_pass
        and ratio_min is not None
        and ratio_min >= THRESHOLD
    )
    if not correctness_passed or (perf and not all_sol_pass):
        status = "blocked"
        reason = "correctness failure or non-pass SOL gate"
    elif perf_ok:
        status = "perf_ok"
        reason = f"ratio_min={ratio_min:.6f} >= {THRESHOLD:.2f}"
    elif perf:
        status = "perf_measured"
        reason = f"ratio_min={ratio_min:.6f}; T075 does not gate on ratio"
    else:
        status = "correct"
        reason = "correctness-only run"

    command = (
        "cd /home/dyq/workspace && "
        "export ASCEND_RT_VISIBLE_DEVICES=0 && "
        "python -u tileops-ascend/tests/t075_convolution_coverage.py"
    )
    payload = {
        "schema_version": "1.0.0",
        "op": "Conv2dFwdOp",
        "target": "ascend",
        "status": status,
        "device": {
            "soc": "ascend910b1",
            "visible_devices": visible_devices(),
            "npu_smi": smi,
        },
        "correctness": {
            "passed": correctness_passed,
            "cmd": command,
            "per_case": correctness,
        },
        "perf": {
            "timing": "host_perf_counter_with_pre_and_post_torch_npu_synchronize",
            "warmup": WARMUP,
            "repeats": REPEATS,
            "statistic": "median",
            "l2_flushed": False,
            "cases": perf,
            "ratio_min": ratio_min,
        },
        "baseline": {
            "tier": "vendor",
            "library": "Torch-NPU eager torch.nn.functional.conv2d",
            "repo_commit": repo_revision(WORKSPACE / "ascend_baselines" / "ops-nn"),
            "source_path": str(Path(torch_npu.__file__).resolve()),
            "build_cmd": "N/A: exact Torch-NPU eager vendor baseline",
            "semantic_match": "exact",
            "fusion_match": "exact",
        },
        "verdict": {
            "perf_ok": perf_ok,
            "threshold": THRESHOLD,
            "reason": reason,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "harness_commit": git_revision(HARNESS),
    }
    validate_result(payload)
    data_dir = WORKSPACE / "docs" / "reports" / "R075-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "Conv2dFwdOp.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    if not args.correctness_only:
        write_json(HARNESS / "coverage" / "Conv2dFwdOp.json", payload)
    print(
        f"[SUMMARY] correctness={correctness_passed} all_sol_pass={all_sol_pass} "
        f"ratio_min={ratio_min} status={status}",
        flush=True,
    )
    if not correctness_passed or (perf and not all_sol_pass):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
