"""Standalone harness adapter for the T036 MaxPool2dFwdOp pilot."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu

WORKSPACE = Path(__file__).resolve().parents[3]
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
    visible_devices,
    write_json,
)

import tileops.kernels.families.pool  # noqa: E402,F401 - registration side effect
from tileops.ops.pool import MaxPool2dFwdOp  # noqa: E402

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

WORKLOADS = [
    ((32, 64, 112, 112), (3, 3), (2, 2), (1, 1), (1, 1), False, "resnet-stem"),
    ((16, 128, 56, 56), (2, 2), (2, 2), (0, 0), (1, 1), False, "vgg-block"),
    ((32, 64, 55, 55), (3, 3), (2, 2), (0, 0), (1, 1), True, "alexnet-ceil"),
]


def _shape_info(shape, output_shape, kernel, stride, padding, dilation, ceil_mode):
    return {
        "input": list(shape),
        "output": list(output_shape),
        "kernel_size": list(kernel),
        "stride": list(stride),
        "padding": list(padding),
        "dilation": list(dilation),
        "ceil_mode": ceil_mode,
    }


def _tested_call(op, npu_input):
    return op(npu_input)


def _baseline_call(npu_input, kernel, stride, padding, dilation, ceil_mode):
    return F.max_pool2d(
        npu_input,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


def _correctness_case(
    shape,
    kernel,
    stride,
    padding,
    dilation,
    ceil_mode,
    dtype_name,
    label,
):
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
    delta = (actual.float() - reference.float()).abs()
    max_abs = float(delta.max()) if delta.numel() else 0.0
    denominator = reference.float().abs().clamp_min(1e-12)
    max_rel = float((delta / denominator).max()) if delta.numel() else 0.0
    atol = rtol = 1.6e-2 if dtype_name == "bfloat16" else 1e-3
    finite = bool(torch.isfinite(actual).all())
    passed = finite and torch.allclose(actual, reference, atol=atol, rtol=rtol)
    shape_info = _shape_info(
        shape, reference.shape, kernel, stride, padding, dilation, ceil_mode
    )
    result = {
        "shape": shape_info,
        "dtype": dtype_name,
        "label": label,
        "finite": finite,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "tolerance": {"atol": atol, "rtol": rtol},
        "passed": bool(passed),
    }
    return result, op, npu_input


def run(out_dir: Path) -> dict:
    if visible_devices() != "5":
        raise RuntimeError(
            "T036 measurements require ASCEND_RT_VISIBLE_DEVICES=5, "
            f"got {visible_devices()!r}"
        )
    profile = load_profile(PROFILE_PATH)
    correctness_cases = []
    perf_cases = []

    for shape, kernel, stride, padding, dilation, ceil_mode, label in WORKLOADS:
        for dtype_name in DTYPES:
            correctness, op, npu_input = _correctness_case(
                shape,
                kernel,
                stride,
                padding,
                dilation,
                ceil_mode,
                dtype_name,
                label,
            )
            correctness_cases.append(correctness)

            tested = partial(_tested_call, op, npu_input)
            baseline = partial(
                _baseline_call,
                npu_input,
                kernel,
                stride,
                padding,
                dilation,
                ceil_mode,
            )

            tested_samples, baseline_samples = paired_samples_us(
                tested, baseline, torch.npu.synchronize
            )
            tested_stats = sample_stats(tested_samples)
            baseline_stats = sample_stats(baseline_samples)
            tested_us = tested_stats["median"]
            baseline_us = baseline_stats["median"]
            flops, nbytes = op.eval_roofline()
            sol_hw, bound_hw = sol_us(flops, nbytes, dtype_name, profile, "peak")
            sol_achieved, bound_achieved = sol_us(
                flops, nbytes, dtype_name, profile, "achieved"
            )
            tested_gate = gate_case(tested_us, sol_hw)
            baseline_gate = gate_case(baseline_us, sol_hw)
            overall_gate = (
                "VIOLATION" if "VIOLATION" in {tested_gate, baseline_gate} else "pass"
            )
            perf_cases.append(
                {
                    "shape": correctness["shape"],
                    "dtype": dtype_name,
                    "label": label,
                    "tileops_us": tested_us,
                    "baseline_us": baseline_us,
                    "ratio": baseline_us / tested_us,
                    "flops": flops,
                    "bytes": nbytes,
                    "roofline_source": "MaxPool2dFwdOp.eval_roofline()",
                    "roofline_error": None,
                    "sol_us_hw": sol_hw,
                    "sol_bound_hw": bound_hw,
                    "sol_us_achieved": sol_achieved,
                    "sol_bound_achieved": bound_achieved,
                    "efficiency_vs_hw_peak": sol_hw / tested_us,
                    "efficiency_vs_vendor_achieved": sol_achieved / tested_us,
                    "baseline_efficiency_vs_hw_peak": sol_hw / baseline_us,
                    "baseline_efficiency_vs_vendor_achieved": sol_achieved
                    / baseline_us,
                    "sol_gate": overall_gate,
                    "sol_gate_detail": {
                        "tileops": tested_gate,
                        "baseline": baseline_gate,
                    },
                    "timing_samples_us": {
                        "tileops": tested_samples,
                        "baseline": baseline_samples,
                    },
                    "timing_stats_us": {
                        "tileops": tested_stats,
                        "baseline": baseline_stats,
                    },
                }
            )
            print(
                f"[MEASURED] {label} {dtype_name}: "
                f"tileops={tested_us:.3f}us baseline={baseline_us:.3f}us "
                f"ratio={baseline_us / tested_us:.6f} gate={overall_gate}",
                flush=True,
            )
            del op, npu_input
            torch.npu.empty_cache()

    nondiv, _, _ = _correctness_case(
        (3, 5, 17, 19),
        (3, 3),
        (2, 2),
        (1, 1),
        (2, 1),
        True,
        "float16",
        "nondiv-dilated-ceil",
    )
    correctness_cases.append(nondiv)
    print(
        f"[CORRECTNESS] nondiv-dilated-ceil float16: passed={nondiv['passed']}",
        flush=True,
    )

    ratio_min = min(case["ratio"] for case in perf_cases)
    correctness_passed = all(case["passed"] for case in correctness_cases)
    violation = any(case["sol_gate"] == "VIOLATION" for case in perf_cases)
    perf_ok = correctness_passed and not violation and ratio_min >= THRESHOLD
    if violation:
        status = "blocked"
        reason = "reported timing is below the hardware SOL"
    elif not correctness_passed:
        status = "blocked"
        reason = "correctness failed"
    elif perf_ok:
        status = "perf_ok"
        reason = f"all cases pass; ratio_min={ratio_min:.6f} >= {THRESHOLD:.2f}"
    else:
        status = "perf_measured"
        reason = f"ratio_min={ratio_min:.6f} < {THRESHOLD:.2f}"

    command = (
        f"cd {WORKSPACE / 'tileops-ascend'} && "
        "export ASCEND_RT_VISIBLE_DEVICES=5 && "
        f"python tests/bench_pool_harness.py --out {out_dir}"
    )
    return {
        "schema_version": "1.0.0",
        "op": "MaxPool2dFwdOp",
        "target": "ascend",
        "status": status,
        "device": {
            "soc": "ascend910b1",
            "visible_devices": visible_devices(),
            "npu_smi": npu_smi_info(),
        },
        "correctness": {
            "passed": correctness_passed,
            "cmd": command,
            "per_case": correctness_cases,
        },
        "perf": {
            "timing": "host_perf_counter_with_pre_and_post_torch_npu_synchronize",
            "warmup": WARMUP,
            "repeats": REPEATS,
            "statistic": "median",
            "order": "paired; tested-first on even repeats, baseline-first on odd repeats",
            "l2_flushed": False,
            "cases": perf_cases,
            "ratio_min": ratio_min,
        },
        "baseline": {
            "tier": "vendor",
            "library": "Torch-NPU eager torch.nn.functional.max_pool2d",
            "repo_commit": repo_revision(WORKSPACE / "ascend_baselines" / "ops-nn"),
            "source_path": str(Path(torch_npu.__file__).resolve()),
            "build_cmd": (
                "N/A: ops-nn MaxPoolV2 declares only ascend950 and ATVC has no "
                "semantically equivalent runnable MaxPool2d example"
            ),
            "semantic_match": "exact",
            "fusion_match": "exact",
            "compile_error": (
                "No 910B handwritten candidate: max_pool_v2 README marks Atlas A2 "
                "unsupported and CMake SUPPORT_COMPUTE_UNIT=ascend950"
            ),
        },
        "verdict": {"perf_ok": perf_ok, "threshold": THRESHOLD, "reason": reason},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "harness_commit": git_revision(HARNESS),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HARNESS / "coverage")
    args = parser.parse_args()
    result = run(args.out)
    destination = args.out / "MaxPool2dFwdOp.json"
    write_json(destination, result)
    print(
        f"output={destination} status={result['status']} "
        f"ratio_min={result['perf']['ratio_min']}"
    )
    return 0 if result["status"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
