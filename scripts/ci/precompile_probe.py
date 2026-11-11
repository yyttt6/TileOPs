"""Opt-in, guarded cold-cache/reuse diagnostic for the merged ReLU pilot.

This does not replace the trusted measurement harness or change its defaults.
Each stage runs in a fresh process. Compiler counts cover direct bisheng/ld.lld
launches through Python subprocess.Popen, the supported TileLang Ascend path;
they are not a general audit of native or descendant process creation.
"""

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


@contextlib.contextmanager
def compiler_counter():
    original = subprocess.Popen
    launches = []
    lock = threading.Lock()

    def counted(args, *pos, **kw):
        command = args if isinstance(args, (list, tuple)) else [args]
        name = Path(os.fsdecode(command[0])).name if command else ""
        child = original(args, *pos, **kw)
        if name in {"bisheng", "ld.lld"}:
            with lock:
                launches.append([os.fsdecode(x) for x in command])
        return child

    subprocess.Popen = counted
    try:
        yield launches
    finally:
        subprocess.Popen = original


def require_guard(source):
    sys.path[:0] = [str(source / "src"), str(source)]
    from benchmarks.ascend.guard import require_reservation

    return require_reservation()


def stage(source, name, output, workers):
    report = {"stage": name, "status": "failed"}
    started = time.monotonic()
    try:
        # Enforced in EVERY child before Torch/TileLang or operator imports.
        report["guard_owner"] = require_guard(source).get("owner")
        with compiler_counter() as launches:
            try:
                import precompile

                if name == "enumerate":
                    programs, skipped, reason = precompile.plan_for("ReLUFwdOp", source)
                    if reason or skipped or not programs:
                        raise RuntimeError(f"Incomplete pilot enumeration: {reason}, {skipped}")
                    report["programs"] = len(programs)
                elif name == "compile":
                    receipt_path = output.with_name("precompile-pass.json")
                    code = precompile.run(source, ["ReLUFwdOp"], receipt_path, workers)
                    report["receipt"] = json.loads(receipt_path.read_text())
                    if code:
                        raise RuntimeError("Merged precompile driver failed")
                else:
                    import torch
                    import torch_npu  # noqa: F401

                    from benchmarks._cases import entry
                    from benchmarks.api import cases
                    from benchmarks.ascend.runner import benchmark_case
                    from tileops.backend import _registry
                    from tileops.elementwise import ReLUFwdOp
                    from workloads.device import set_run_device

                    torch.npu.set_device(0)
                    set_run_device("npu")
                    _registry.ensure_loaded()
                    if _registry.LOAD_FAILURES:
                        raise RuntimeError(_registry.LOAD_FAILURES)
                    import tileops._kernels.ascend.elementwise  # noqa: F401

                    report["cases"] = []
                    for case in cases("ReLUFwdOp"):
                        try:
                            torch.manual_seed(0)
                            op = ReLUFwdOp(**case.arguments)
                            impl = entry("ReLUFwdOp").binder(op, case)
                            checked = benchmark_case(
                                case, {"tileops": impl}, None, name=case.id, verify_only=True
                            )
                            torch.npu.synchronize()
                            report["cases"].append({"id": case.id, **checked["tileops"]})
                        finally:
                            case._release()
                    if not report["cases"]:
                        raise RuntimeError("No pilot cases executed")
                report["status"] = "passed"
            finally:
                report["compiler_launches"] = launches
                report["compilers"] = len(launches)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["seconds"] = time.monotonic() - started
        output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "passed" else 1


def evaluate(stages):
    enum = stages.get("enumerate", {})
    compile_stage = stages.get("compile", {})
    fresh = stages.get("call", {})
    compiled = compile_stage.get("receipt", {}).get("operators", {}).get("ReLUFwdOp", {})
    return {
        "all_stages_passed": all(
            stages.get(name, {}).get("status") == "passed" and stages[name].get("exit_code") == 0
            for name in ("enumerate", "compile", "call")
        ),
        "all_programs_compiled": enum.get("programs", 0) > 0
        and compiled.get("status") == "compiled"
        and compiled.get("compiled") == enum.get("programs")
        and not compiled.get("skipped_cases"),
        "enumeration_did_not_compile": enum.get("compilers") == 0,
        "cold_compile_launched_compiler": compile_stage.get("compilers", 0) > 0,
        "fresh_call_did_not_compile": fresh.get("compilers") == 0,
        "all_pilot_cases_verified": len(fresh.get("cases", [])) == 3
        and len({row["id"] for row in fresh.get("cases", [])}) == 3
        and all(
            row.get("correctness", {}).get("checked_outputs", 0) > 0
            and row["correctness"]["checked_outputs"] == row["correctness"].get("total_outputs")
            and not row["correctness"].get("unchecked_reason")
            for row in fresh.get("cases", [])
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new diagnostic directory")
    parser.add_argument("--toolchain", type=Path, help="passed canonical toolchain check receipt")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stage", choices=("enumerate", "compile", "call"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    source = args.source.resolve()
    if args.stage:
        return stage(source, args.stage, args.output, args.workers)
    if args.toolchain is None:
        parser.error("--toolchain is required")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "source": str(source), "stages": {}}
    try:
        report["guard_owner"] = require_guard(source).get("owner")
        raw = args.toolchain.read_bytes()
        toolchain = json.loads(raw)
        if toolchain.get("status") != "passed" or toolchain.get("errors"):
            raise ValueError("Toolchain check did not pass")
        (args.output / "toolchain.json").write_bytes(raw)
        report["toolchain_sha256"] = hashlib.sha256(raw).hexdigest()
        report["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        report["tree"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=source, text=True
        ).strip()
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=source):
            raise ValueError("Diagnostic source must be clean")
        report["probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        cache = args.output / "cache"
        cache.mkdir()  # A new directory, never the shared persistent cache.
        report["cache"] = str(cache)
        for name in ("enumerate", "compile", "call"):
            output = args.output / f"{name}.json"
            with (args.output / f"{name}.log").open("w") as log:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--stage",
                        name,
                        "--source",
                        str(source),
                        "--output",
                        str(output),
                        "--workers",
                        str(args.workers),
                    ],
                    env=dict(os.environ, TILELANG_CACHE_DIR=str(cache)),
                    cwd=source,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=900,
                )
            row = json.loads(output.read_text()) if output.exists() else {"status": "failed"}
            row["exit_code"] = proc.returncode
            report["stages"][name] = row
            if proc.returncode or row.get("status") != "passed":
                break
        report["checks"] = evaluate(report["stages"])
        report["status"] = "passed" if all(report["checks"].values()) else "failed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
