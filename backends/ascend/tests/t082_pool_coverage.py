#!/usr/bin/env python3
"""T082 correctness, SOL, and canonical coverage generation on Ascend card 4."""

from datetime import datetime, timezone
from functools import partial
import argparse
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
import tileops_ascend.families.pool  # noqa: E402,F401
from tileops.ops.pool import (  # noqa: E402
    AvgPool1dFwdOp,
    AvgPool2dFwdOp,
    AvgPool3dFwdOp,
    MaxPool3dFwdOp,
)


DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}

MANIFEST_CASES = {
    "AvgPool1dFwdOp": [
        ((4, 128, 4096), "float16", dict(kernel_size=3, stride=2, padding=1), "audio-downsample"),
        ((2, 256, 32000), "float16", dict(kernel_size=5, stride=4, padding=2), "long-temporal"),
        (
            (2, 128, 2048),
            "bfloat16",
            dict(
                kernel_size=4,
                stride=2,
                padding=1,
                ceil_mode=True,
                count_include_pad=False,
            ),
            "ceil",
        ),
    ],
    "AvgPool2dFwdOp": [
        (
            (2, 64, 112, 112),
            "float16",
            dict(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            "vision-3x3-s2",
        ),
        (
            (2, 128, 56, 56),
            "float16",
            dict(kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            "vision-5x5-s2",
        ),
        (
            (3, 96, 55, 57),
            "bfloat16",
            dict(
                kernel_size=(3, 5),
                stride=(2, 2),
                padding=(1, 2),
                ceil_mode=True,
                count_include_pad=False,
                divisor_override=7,
            ),
            "ceil-divisor",
        ),
    ],
    "AvgPool3dFwdOp": [
        (
            (1, 32, 16, 56, 56),
            "float16",
            dict(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
            "video-2x2x2",
        ),
        (
            (2, 64, 8, 28, 28),
            "float16",
            dict(
                kernel_size=(2, 3, 3),
                stride=(2, 2, 2),
                padding=(1, 1, 1),
                ceil_mode=True,
                count_include_pad=False,
            ),
            "ceil-video",
        ),
        (
            (2, 24, 10, 20, 22),
            "bfloat16",
            dict(
                kernel_size=(2, 2, 3),
                stride=(2, 2, 2),
                padding=(0, 1, 1),
                divisor_override=7,
            ),
            "divisor",
        ),
    ],
    "MaxPool3dFwdOp": [
        (
            (8, 64, 16, 112, 112),
            "float16",
            dict(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),
            "c3d-pool1",
        ),
        (
            (4, 128, 16, 56, 56),
            "float16",
            dict(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
            "c3d-pool2",
        ),
        (
            (2, 64, 32, 112, 112),
            "bfloat16",
            dict(
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                padding=(1, 1, 1),
                ceil_mode=True,
            ),
            "medicalnet-stem",
        ),
    ],
}

EXTRA_CASES = {
    "AvgPool1dFwdOp": [
        ((1, 2, 17), "float16", dict(kernel_size=3, stride=2, padding=1), "include-pad-true"),
        (
            (1, 2, 17),
            "float16",
            dict(kernel_size=3, stride=2, padding=1, count_include_pad=False),
            "include-pad-false",
        ),
        (
            (1, 2, 4),
            "float16",
            dict(kernel_size=3, stride=2, padding=1, ceil_mode=True),
            "ceil-implicit-right-pad",
        ),
    ],
    "AvgPool2dFwdOp": [
        (
            (1, 2, 17, 19),
            "float16",
            dict(
                kernel_size=(3, 5),
                stride=(2, 3),
                padding=(1, 2),
                ceil_mode=True,
                divisor_override=7,
            ),
            "nondiv-divisor-override",
        )
    ],
    "AvgPool3dFwdOp": [
        (
            (1, 2, 5, 7, 9),
            "float16",
            dict(
                kernel_size=(2, 3, 3),
                stride=(2, 2, 2),
                padding=(0, 1, 1),
                ceil_mode=True,
                count_include_pad=False,
            ),
            "nondiv-ncdhw",
        )
    ],
    "MaxPool3dFwdOp": [
        (
            (1, 2, 5, 7, 9),
            "float16",
            dict(
                kernel_size=(2, 3, 3),
                stride=(2, 2, 2),
                padding=(0, 1, 1),
                dilation=(1, 2, 1),
                ceil_mode=True,
            ),
            "nondiv-dilated-ncdhw",
        ),
        (
            (1, 1, 258, 256, 128),
            "float16",
            dict(kernel_size=1, stride=1, padding=0),
            "grid-stride-over-65535",
        ),
    ],
}

OP_CLASSES = {
    "AvgPool1dFwdOp": AvgPool1dFwdOp,
    "AvgPool2dFwdOp": AvgPool2dFwdOp,
    "AvgPool3dFwdOp": AvgPool3dFwdOp,
    "MaxPool3dFwdOp": MaxPool3dFwdOp,
}

REFERENCE = {
    "AvgPool1dFwdOp": F.avg_pool1d,
    "AvgPool2dFwdOp": F.avg_pool2d,
    "AvgPool3dFwdOp": F.avg_pool3d,
    "MaxPool3dFwdOp": F.max_pool3d,
}


def _json_params(params):
    return {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in params.items()
    }


def _make_input(shape, dtype, sentinel):
    if sentinel:
        return torch.full(shape, 2.0, dtype=dtype, device="npu")
    return torch.randn(shape, dtype=dtype, device="npu")


def _correctness_case(op_name, shape, dtype_name, params, label):
    dtype = DTYPES[dtype_name]
    sentinel = label == "grid-stride-over-65535"
    x = _make_input(shape, dtype, sentinel)
    op = OP_CLASSES[op_name](target="ascend", **params)
    actual = op(x)
    reference_device = "npu"
    try:
        reference = REFERENCE[op_name](x, **params)
    except RuntimeError:
        dilation = params.get("dilation", 1)
        if op_name != "MaxPool3dFwdOp" or dilation == 1:
            raise
        reference = REFERENCE[op_name](x.cpu(), **params)
        actual = actual.cpu()
        reference_device = "cpu"
    torch.npu.synchronize()
    delta = (actual.float() - reference.float()).abs()
    max_abs = float(delta.max().cpu()) if delta.numel() else 0.0
    denominator = reference.float().abs().clamp_min(1e-12)
    max_rel = float((delta / denominator).max().cpu()) if delta.numel() else 0.0
    is_max = op_name == "MaxPool3dFwdOp"
    atol = rtol = 0.0 if is_max else (1.6e-2 if dtype_name == "bfloat16" else 1e-3)
    passed = bool(
        torch.equal(actual, reference)
        if is_max
        else torch.allclose(actual, reference, atol=atol, rtol=rtol)
    )
    zero_count = int((actual == 0).sum().cpu()) if sentinel else None
    if sentinel:
        passed = passed and zero_count == 0
    row = {
        "shape": {
            "input": list(shape),
            "output": list(actual.shape),
            "params": _json_params(params),
            "label": label,
            "path": op.kernel.path_kind,
            "logical_blocks": op.kernel.logical_blocks,
            "launch_blocks": op.kernel.launch_blocks,
            "reference_device": reference_device,
        },
        "dtype": dtype_name,
        "finite": bool(torch.isfinite(actual).all().cpu()),
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "tolerance": {"atol": atol, "rtol": rtol},
        "passed": passed,
    }
    if sentinel:
        row["sentinel_zero_count"] = zero_count
    print(
        f"[CORRECTNESS] {op_name} {label} {dtype_name} "
        f"max_abs={max_abs} logical_blocks={op.kernel.logical_blocks} "
        f"launch_blocks={op.kernel.launch_blocks} passed={passed}",
        flush=True,
    )
    return row, op, x


def _perf_case(op_name, shape, dtype_name, params, label, op, x, profile):
    tested = partial(op, x)
    baseline = partial(REFERENCE[op_name], x, **params)
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
        "VIOLATION"
        if "VIOLATION" in {tested_gate, baseline_gate}
        else "pass"
    )
    shape_info = {
        "input": list(shape),
        "output": list(op.kernel.output_shape),
        "params": _json_params(params),
        "label": label,
    }
    row = {
        "shape": shape_info,
        "dtype": dtype_name,
        "label": label,
        "tileops_us": tested_us,
        "baseline_us": baseline_us,
        "ratio": baseline_us / tested_us,
        "flops": flops,
        "bytes": nbytes,
        "roofline_source": f"{op_name}.eval_roofline()",
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
        f"[MEASURED] {op_name} {label} {dtype_name} "
        f"tileops={tested_us:.3f}us baseline={baseline_us:.3f}us "
        f"ratio={row['ratio']:.6f} sol_gate={overall_gate}",
        flush=True,
    )
    return row


def _payload(op_name, correctness, perf, npu_smi):
    correctness_passed = all(row["passed"] for row in correctness)
    ratio_min = min(row["ratio"] for row in perf)
    violation = any(row["sol_gate"] != "pass" for row in perf)
    perf_ok = correctness_passed and not violation and ratio_min >= THRESHOLD
    if violation or not correctness_passed:
        status = "blocked"
        reason = "correctness failure or non-pass SOL gate"
    elif perf_ok:
        status = "perf_ok"
        reason = f"ratio_min={ratio_min:.6f} >= {THRESHOLD:.2f}"
    else:
        status = "perf_measured"
        reason = f"ratio_min={ratio_min:.6f} < {THRESHOLD:.2f}"
    command = (
        "cd /home/dyq/workspace/tileops-ascend && "
        "export ASCEND_RT_VISIBLE_DEVICES=4 && "
        "python tests/t082_pool_coverage.py"
    )
    return {
        "schema_version": "1.0.0",
        "op": op_name,
        "target": "ascend",
        "status": status,
        "device": {
            "soc": "ascend910b1",
            "visible_devices": visible_devices(),
            "npu_smi": npu_smi,
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
            "library": f"Torch-NPU eager torch.nn.functional.{op_name[:-5].lower()}",
            "repo_commit": repo_revision(WORKSPACE / "ascend_baselines" / "ops-nn"),
            "source_path": str(Path(torch_npu.__file__).resolve()),
            "build_cmd": "N/A: T082 uses exact Torch-NPU eager vendor baseline",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=tuple(OP_CLASSES))
    args = parser.parse_args()
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "4":
        raise RuntimeError("T082 requires ASCEND_RT_VISIBLE_DEVICES=4")
    profile = load_profile(PROFILE_PATH)
    npu_smi = npu_smi_info()
    output_dir = HARNESS / "coverage"
    data_dir = WORKSPACE / "docs" / "reports" / "R082-data"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    failed = False
    op_names = [args.op] if args.op else OP_CLASSES
    for op_name in op_names:
        correctness = []
        perf = []
        for shape, dtype_name, params, label in MANIFEST_CASES[op_name]:
            row, op, x = _correctness_case(
                op_name, shape, dtype_name, params, label
            )
            correctness.append(row)
            perf.append(
                _perf_case(
                    op_name,
                    shape,
                    dtype_name,
                    params,
                    label,
                    op,
                    x,
                    profile,
                )
            )
            del op, x
            torch.npu.empty_cache()
        for shape, dtype_name, params, label in EXTRA_CASES[op_name]:
            row, op, x = _correctness_case(
                op_name, shape, dtype_name, params, label
            )
            correctness.append(row)
            del op, x
            torch.npu.empty_cache()

        payload = _payload(op_name, correctness, perf, npu_smi)
        validate_result(payload)
        write_json(output_dir / f"{op_name}.json", payload)
        write_json(data_dir / f"{op_name}.json", payload)
        failed |= payload["status"] == "blocked"
        print(
            f"[COVERAGE] {op_name} status={payload['status']} "
            f"correctness={payload['correctness']['passed']} "
            f"ratio_min={payload['perf']['ratio_min']:.6f}",
            flush=True,
        )
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
