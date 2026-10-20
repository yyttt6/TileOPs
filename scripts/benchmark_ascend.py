#!/usr/bin/env python3
"""Team benchmark entry point: reserve the host NPU before importing its runtime."""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="new directory; existing output is never reused"
    )
    parser.add_argument("--owner", required=True)
    parser.add_argument("--wait", type=int, default=600)
    parser.add_argument("--timeout", type=int, default=1800)
    args, rest = parser.parse_known_args()
    if "--help" in rest:
        return 0
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root)))
    guard = root / "scripts/npu_guard.py"
    command = [
        sys.executable,
        str(guard),
        "run",
        "--owner",
        args.owner,
        "--wait",
        str(args.wait),
        "--timeout",
        str(args.timeout),
        "--",
        sys.executable,
        "-m",
        "benchmarks.ascend.runner",
        "--run-dir",
        str(run),
        *rest,
    ]
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (run / "run.log").open("x") as log:
        result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    child = run / "result.json"
    summary = json.loads(child.read_text()) if child.exists() else {}
    success = result.returncode == 0 and summary.get("status") == "passed"
    receipt = {
        "status": "passed" if success else "failed",
        "guard_exit_code": result.returncode,
        "started_at": started,
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": command,
        "result": summary,
    }
    (run / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"run_dir": str(run), **receipt}, indent=2))
    return 0 if success else (result.returncode or 1)


if __name__ == "__main__":
    sys.exit(main())
