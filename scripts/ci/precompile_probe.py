"""Diagnostic entry for the pre-compilation pass: does the ordinary call path reuse it?

The pass is an optimisation and may not fail a job, so a green measurement proves nothing
about it: a run whose pass never executed and a run whose pass worked both finish green.
This entry is the controlled diagnostic that separates the two. It runs as a *step of its
own*, on the candidate's own code, and then hands the operator back to the ordinary
measurement step, which is untouched.

The experiment is three processes and one counter:

1. **enumerate** -- the operator's cases, in-process, and the compiler invocations that
   enumeration alone starts. This has to be zero for step 2 to mean anything.
2. **pre-compile** -- ``precompile.py`` against a cache directory of its own; the receipt
   records how many compiler invocations the pass spent.
3. **fresh call** -- a new process builds the operator through its public constructor and
   runs one ``forward``, counting compiler invocations. Zero here, with a nonzero count in
   step 2, is the claim: the programs the pass built were the ones the call needed.

Step 3 runs a real ``forward``, so it is placed inside the device reservation, and the
whole entry is executed by the same ``npu-guard`` the measurement step uses. That is the
answer to "does the pass touch the device": it does not have to be argued from
``is_initialized()`` -- which cannot answer it, since importing ``torch_npu`` opens
``/dev/davinci_manager`` while leaving the flag false -- because the pass is not outside
the reservation in the first place.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CHILD = r'''
"""One process, one job: enumerate, pre-compile, or make the real call."""
import json, os, sys, time
sys.path[:0] = [os.environ["PROBE_SOURCE"] + "/src", os.environ["PROBE_SOURCE"]]
sys.path.insert(0, os.environ["PROBE_SCRIPTS"])
import torch, torch_npu
from precompile import CompilerCounter, plan_for
import importlib
from tileops.backend import _registry
from tileops.manifest import load_manifest

op_name = os.environ["PROBE_OP"]
stage = os.environ["PROBE_STAGE"]
manifest = load_manifest()
_registry.ensure_loaded()
if op_name not in _registry.REGISTERED_KERNEL_TYPES:
    importlib.import_module(f"tileops._kernels.ascend.{manifest[op_name]['family']}")
op_class = getattr(importlib.import_module(f"tileops.{manifest[op_name]['family']}"), op_name)

report = dict(stage=stage, device_initialized=torch.npu.is_initialized())
CompilerCounter.install()
began = time.monotonic()
try:
    if stage == "call":
        # The workload draws its inputs on the run's device, so set it the way the
        # measurement process does. This is the only stage that touches the device.
        from workloads.device import set_run_device
        torch.npu.set_device(0)
        set_run_device("npu")
        from benchmarks.api import cases
        from benchmarks._cases import entry
        case = cases(op_name)[0]
        op = op_class(**case.arguments)
        impl = entry(op_name).binder(op, case)
        args = case.inputs if impl.args is None else impl.args
        out = impl.run(*args)
        report["built"] = 1
        report["ran"] = True
        report["output"] = [list(o.shape) for o in (out if isinstance(out, (list, tuple)) else [out])]
    else:
        programs, skipped, unsupported = plan_for(op_name, __import__("pathlib").Path(os.environ["PROBE_SOURCE"]))
        report["programs"] = len(programs)
        report["skipped_cases"] = skipped
        report["unsupported"] = unsupported
except Exception:
    report["error"] = __import__("traceback").format_exc()
finally:
    CompilerCounter.remove()
    report["compilers"] = CompilerCounter.count
    report["seconds"] = round(time.monotonic() - began, 3)
    report["device_initialized_after"] = torch.npu.is_initialized()
print("PROBE " + json.dumps(report))
'''


def run_child(stage, source, op, cache_dir, extra_env=None):
    env = dict(
        os.environ,
        PROBE_SOURCE=str(source),
        PROBE_OP=op,
        PROBE_STAGE=stage,
        PROBE_SCRIPTS=str(HERE),
        TILELANG_CACHE_DIR=str(cache_dir),
    )
    if extra_env:
        env.update(extra_env)
    began = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", CHILD], capture_output=True, text=True, env=env,
        cwd=str(source),
    )
    wall = round(time.monotonic() - began, 3)
    report = None
    for line in proc.stdout.splitlines():
        if line.startswith("PROBE "):
            report = json.loads(line[6:])
    return dict(
        exit=proc.returncode,
        seconds=wall,
        report=report,
        stdout=proc.stdout[-3000:],
        stderr=proc.stderr[-3000:],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--op", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="cache directory the run owns")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("TILEOPS_PRECOMPILE_WORKERS", "16")))
    args = parser.parse_args()

    source = args.source.resolve()
    args.cache.mkdir(parents=True, exist_ok=True)
    # A run that starts from a warm shared cache cannot show a compile at all, so this
    # experiment uses a directory of its own rather than the run's persistent namespace.
    evidence = dict(op=args.op, source=str(source), cache=str(args.cache), stages={})

    evidence["stages"]["enumerate"] = run_child("enumerate", source, args.op, args.cache)

    pass_receipt = args.output.with_name(args.output.stem + "-pass.json")
    began = time.monotonic()
    code = subprocess.run(
        [sys.executable, str(HERE / "precompile.py"),
         "--source", str(source), "--op", args.op,
         "--output", str(pass_receipt), "--workers", str(args.workers)],
        capture_output=True, text=True,
        env=dict(os.environ, TILELANG_CACHE_DIR=str(args.cache)),
        cwd=str(source),
    )
    evidence["stages"]["precompile"] = dict(
        exit=code.returncode,
        seconds=round(time.monotonic() - began, 3),
        receipt=json.loads(pass_receipt.read_text()) if pass_receipt.exists() else None,
        stderr=code.stderr[-3000:],
    )

    evidence["stages"]["fresh_call"] = run_child("call", source, args.op, args.cache)

    enumeration = evidence["stages"]["enumerate"].get("report") or {}
    receipt = (evidence["stages"]["precompile"].get("receipt") or {})
    fresh = evidence["stages"]["fresh_call"].get("report") or {}
    checks = {
        # Enumeration is compile-only: if it compiled, the later counts prove nothing.
        "enumeration_compiled_nothing": enumeration.get("compilers") == 0,
        # The pass did the building, from a cold directory.
        "pass_compiled": (receipt.get("compilers") or 0) > 0,
        # The ordinary call path built the operator and needed no compiler of its own.
        "fresh_call_ran": bool(fresh.get("ran")),
        "fresh_call_compiled_nothing": fresh.get("compilers") == 0,
        "fresh_call_clean": "error" not in fresh,
    }
    evidence["checks"] = checks
    evidence["status"] = "passed" if all(checks.values()) else "failed"
    evidence["claim"] = (
        f"pass spent {receipt.get('compilers')} compiler invocations on "
        f"{receipt.get('programs')} programs; the next process ran the op with "
        f"{fresh.get('compilers')}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(dict(status=evidence["status"], checks=checks,
                          claim=evidence["claim"]), indent=2))
    return 0 if evidence["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
