#!/usr/bin/env python3
"""The local and GitHub Ascend 950 correctness CI entry point.

Performance acceptance is a separate benchmark gate requiring an aligned baseline
and clock policy. This entry checks source contracts and every registered NPU op's
manifest workload, and retains failed or interrupted runs.
"""

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cpu", "device", "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    owner = f"ci/ascend950/{uuid.uuid4().hex}"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT))))
    steps = []
    if args.phase in ("cpu", "all"):
        steps.extend(
            [
                (
                    "manifest",
                    [
                        sys.executable,
                        "scripts/validate_manifest.py",
                        "--levels",
                        "schema,signature,bench",
                    ],
                ),
                (
                    "contracts",
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "--tileops-device=cpu",
                        "tests/test_validate_manifest.py",
                        "tests/test_check_bench_coverage.py",
                        "tests/test_target_conformance.py",
                        "tests/test_ascend_toolchain.py",
                        "tests/test_kernel_dispatch.py",
                        f"--junitxml={output / 'contracts.xml'}",
                    ],
                ),
                (
                    "benchmark-contracts",
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "benchmarks/tests",
                        f"--junitxml={output / 'benchmark-contracts.xml'}",
                    ],
                ),
                ("packaging-coverage", [sys.executable, "scripts/ci/check_packaging_coverage.py"]),
            ]
        )
    if args.phase in ("device", "all"):
        steps.append(
            (
                "manifest-device",
                [
                    sys.executable,
                    "scripts/benchmark_ascend.py",
                    "--owner",
                    f"{owner}/manifest-device",
                    "--timeout",
                    "5400",
                    "--run-dir",
                    str(output / "manifest-device"),
                    "--verify-only",
                    "--op",
                    "all",
                ],
            )
        )
    results = []
    for name, command in steps:
        if (
            name != "manifest-device"
            and env.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0"
            and Path("/usr/local/Ascend/driver/lib64/driver/libdrvdsmi_host.so").exists()
        ):
            # Imports on an NPU host also belong to the host reservation.
            command = [
                sys.executable,
                str(
                    Path(
                        os.environ.get("TILEOPS_NPU_GUARD", ROOT / "scripts/npu_guard.py")
                    ).resolve(strict=True)
                ),
                "run",
                "--owner",
                f"{owner}/{name}",
                "--wait",
                "600",
                "--timeout",
                "900",
                "--",
                *command,
            ]
        with (output / f"{name}.log").open("x") as log:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        step = {"name": name, "command": command, "exit_code": result.returncode}
        if name in {"contracts", "benchmark-contracts"} and result.returncode == 0:
            import xml.etree.ElementTree as ET

            cases = ET.parse(output / f"{name}.xml").findall(".//testcase")
            passed = sum(
                all(c.find(tag) is None for tag in ("failure", "error", "skipped")) for c in cases
            )
            step["passed"] = passed
            if passed == 0:
                step["exit_code"] = 1
                step["error"] = "JUnit contains no passed tests"
        results.append(step)
        receipt = {
            "status": "passed"
            if len(results) == len(steps) and all(r["exit_code"] == 0 for r in results)
            else "incomplete",
            "phase": args.phase,
            "steps": results,
        }
        if step["exit_code"]:
            receipt["status"] = "failed"
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"{name}: exit {step['exit_code']}", flush=True)
        if step["exit_code"]:
            return step["exit_code"]
    return 0


if __name__ == "__main__":
    sys.exit(main())
