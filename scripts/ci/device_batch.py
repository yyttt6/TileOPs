"""Bounded process reuse inside one host reservation, with immutable guard receipts."""

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def groups(operators, size=8):
    if not 1 <= size <= 8:
        raise ValueError("Batch size must be between 1 and 8")
    if len(operators) != len(set(operators)):
        raise ValueError("Duplicate batch operator")
    return [operators[i : i + size] for i in range(0, len(operators), size)]


def run(source, jobs, root, label, owner, *, timer="auto", case_plan=None):
    root = Path(root).resolve()
    if not label.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Unsafe batch label")
    plan, receipt, log, summary = [
        root / f"{label}.{suffix}" for suffix in ("plan.json", "guard.json", "log", "summary.json")
    ]
    for j in jobs:
        if not Path(j["output"]).resolve().is_relative_to(root):
            raise ValueError("Batch output escapes evidence directory")
    write(plan, dict(jobs=jobs, timer=timer, case_plan=case_plan))
    command = [
        "npu-guard",
        "run",
        "--owner",
        owner,
        "--wait",
        "1800",
        "--timeout",
        "7200",
        "--receipt",
        str(receipt),
        "--",
        sys.executable,
        str(HERE / "device_batch.py"),
        "--source",
        str(Path(source).resolve()),
        "--plan",
        str(plan),
        "--summary",
        str(summary),
    ]
    started = time.monotonic()
    with log.open("x") as stream:
        completed = subprocess.run(command, cwd=source, stdout=stream, stderr=subprocess.STDOUT)
    meta = json.loads(receipt.read_text()) if receipt.exists() else {}
    code = completed.returncode
    if not code and not (
        meta.get("owner") == owner
        and meta.get("exit_code") == 0
        and meta.get("performance_valid") is True
        and meta.get("workload_completed") is True
    ):
        code = 78
    return dict(
        code=code,
        guard=meta,
        guard_receipt=str(receipt),
        guard_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest() if receipt.exists() else None,
        seconds=time.monotonic() - started,
        worker=json.loads(summary.read_text()) if summary.exists() else {},
    )


def worker(source, plan, summary):
    import nightly_device

    records = []
    started = time.monotonic()
    for job in plan["jobs"]:
        arguments = [
            "--source",
            str(source),
            "--output",
            job["output"],
            "--op",
            job["op"],
            "--verify-measured-output",
            "--timer-mode",
            plan["timer"],
        ]
        if plan.get("case_plan"):
            arguments += ["--case-plan", plan["case_plan"]]
        begin = time.monotonic()
        fatal = None
        try:
            nightly_device.main(arguments)
            code = 0
        except SystemExit as exc:
            code = exc.code or 0
        except Exception:
            code, fatal = 1, traceback.format_exc()
        records.append(
            dict(op=job["op"], exit_code=code, seconds=time.monotonic() - begin, fatal=fatal)
        )
        write(summary, dict(records=records, seconds=time.monotonic() - started))
        if fatal:
            return 1
        if code:
            # The controller can start untouched operators in a fresh process.
            break
        import torch

        torch.npu.synchronize()
        gc.collect()
        torch.npu.empty_cache()
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    a = p.parse_args()
    return worker(a.source, json.loads(a.plan.read_text()), a.summary)


if __name__ == "__main__":
    raise SystemExit(main())
