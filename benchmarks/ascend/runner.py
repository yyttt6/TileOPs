"""Run manifest cases using the common Ascend correctness and timing protocol."""

import argparse
import dataclasses
import hashlib
import importlib
import json
import re
import signal
import sys
import traceback
from importlib.metadata import version
from pathlib import Path

from benchmarks.ascend.guard import require_reservation

ROOT = Path(__file__).resolve().parents[2]


def source_fingerprint():
    """Fingerprint code and contracts without caches, results or other developers' files."""
    paths = []
    for name in ("src/tileops", "workloads", "benchmarks", "scripts"):
        paths.extend(
            p
            for p in (ROOT / name).rglob("*")
            if p.suffix in {".py", ".yaml", ".sh", ".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh"}
            and "__pycache__" not in p.parts
        )
    paths.extend(
        ROOT / name for name in ("pyproject.toml", "constraints.txt", "constraints-runner-lock.txt")
    )
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)
    }


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, ensure_ascii=False) + "\n")


def runtime_versions(get_version):
    """Keep unavailable component queries visible in the environment record."""
    result = {}
    for module in ("CANN", "RUNTIME", "COMPILER", "DRIVER", "TOOLKIT", "OPP"):
        try:
            result[module] = get_version(module=module)
        except Exception as exc:
            result[module] = {"error": str(exc)}
    return result


def benchmark_case(case, implementations, timer, *, name, verify_only=False):
    """Verify every implementation fully, then measure through the same timer.

    Baseline adapters provide the existing benchmarks.api.Implementation contract.
    Tolerances, references, structure and mutation rules come from the workload.
    """
    from workloads.numerics import Request, check_inputs_preserved, verify

    requests = {}
    for tag, impl in implementations.items():
        if not re.fullmatch("[a-z][a-z0-9-]*", tag):
            raise ValueError(f"Invalid implementation name: {tag}")
        if impl.noncomparable_reason:
            raise ValueError(f"{tag}: not comparable: {impl.noncomparable_reason}")
        args = case.inputs if impl.args is None else impl.args
        requests[tag] = Request(impl.run, args, impl.reset, preserve_inputs=True)
    checks = verify(case.reference, case.inputs, evidence=case.verification, requests=requests)
    for tag, result in checks.items():
        if (
            result.checked_outputs <= 0
            or result.checked_outputs != result.total_outputs
            or result.unchecked_reason
        ):
            raise AssertionError(f"{tag}: incomplete correctness coverage: {result}")
    results = {}
    for tag, impl in implementations.items():
        result = {"correctness": dataclasses.asdict(checks[tag])}
        if not verify_only:
            args = case.inputs if impl.args is None else impl.args
            result["measurement"] = timer.measure(
                impl.run, args, reset=impl.reset, name=f"{name}/{tag}"
            )
            # A complete preservation check, including changes outside a digest prefix.
            check_inputs_preserved(requests[tag], case.inputs)
        results[tag] = result
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--op", action="append", required=True)
    parser.add_argument("--case-regex")
    parser.add_argument("--clock-config", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument(
        "--implementation-factory",
        action="append",
        default=[],
        help="module:function returning named Implementation objects for (op_name, case)",
    )
    args = parser.parse_args(argv)

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Benchmark interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    reservation = require_reservation()
    if not args.verify_only and args.clock_config is None:
        parser.error("formal benchmark requires --clock-config; no unlocked fallback")
    if version("tilelang").split("+", 1)[0] != "0.1.15":
        raise RuntimeError("This protocol requires the official TileLang 0.1.15 release")
    before = source_fingerprint()
    _write(args.run_dir / "sources.json", before)

    import torch
    import torch_npu

    from benchmarks._cases import entry
    from benchmarks.api import Implementation, cases
    from tileops.backend import registry
    from tileops.manifest import load_manifest
    from workloads.device import set_run_device

    registry.ensure_loaded()
    if registry.LOAD_FAILURES:
        raise RuntimeError(registry.LOAD_FAILURES)
    set_run_device("npu")
    device_name = torch.npu.get_device_name(0)
    if "Ascend950" not in device_name:
        raise RuntimeError(f"This protocol requires Ascend 950, got {device_name}")
    manifest = load_manifest()
    supported = {op for op, kinds in registry.REGISTERED_KERNEL_TYPES.items() if "ascend" in kinds}
    selected = sorted(supported) if args.op == ["all"] else args.op
    if not selected:
        raise RuntimeError("No Ascend operators registered; check the installed backend entry point")
    missing = set(selected) - supported
    if missing:
        raise ValueError(f"No Ascend implementation registered: {sorted(missing)}")
    factories = []
    for spec in args.implementation_factory:
        module, name = spec.split(":", 1)
        factories.append(getattr(importlib.import_module(module), name))
    metadata = {
        "tilelang": version("tilelang"),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "cann_components": runtime_versions(torch_npu.utils.get_cann_version),
        "python": sys.version,
        "device": device_name,
        "reservation": reservation,
        "ops": selected,
        "verify_only": args.verify_only,
        "clock_policy": json.loads(args.clock_config.read_text()) if args.clock_config else None,
    }
    _write(args.run_dir / "environment.json", metadata)
    from contextlib import nullcontext

    from benchmarks.ascend.clocks import ClockPolicy
    from benchmarks.ascend.timing import ProfilerTimer

    from benchmarks.ascend.support import unsupported_reason

    passed = failed = unsupported = 0
    completed = False
    clock_context = nullcontext() if args.verify_only else ClockPolicy(args.clock_config)
    try:
        with clock_context as clocks:
            timer = None if args.verify_only else ProfilerTimer(args.run_dir / "profiles", clocks)
            with (args.run_dir / "results.jsonl").open("x") as output:
                for op_name in selected:
                    op_class = getattr(
                        importlib.import_module(f"tileops.{manifest[op_name]['family']}"), op_name
                    )
                    op_cases = [
                        c
                        for c in cases(op_name)
                        if not args.case_regex or re.search(args.case_regex, c.id)
                    ]
                    if not op_cases:
                        raise ValueError(f"{op_name}: selection matched no cases")
                    for index, case in enumerate(op_cases):
                        row = {"op": op_name, "case_id": case.id, "params": case.params}
                        seed = int.from_bytes(
                            hashlib.sha256(f"{op_name}::{case.id}".encode()).digest()[:4], "little"
                        )
                        torch.manual_seed(seed)
                        row["seed"] = seed
                        try:
                            reason = unsupported_reason(
                                op_name, case, registry.REGISTERED_KERNEL_TYPES
                            )
                            if reason is not None:
                                row.update(status="unsupported", reason=reason)
                                unsupported += 1
                                output.write(json.dumps(row, default=str) + "\n")
                                output.flush()
                                print(f"{op_name} {case.id}: unsupported ({reason})", flush=True)
                                continue
                            op = op_class(**case.arguments)
                            implementations = {"tileops": entry(op_name).binder(op, case)}
                            if args.include_reference:
                                implementations["reference"] = Implementation(case.reference)
                            for factory in factories:
                                extra = factory(op_name, case)
                                if set(extra) & set(implementations):
                                    raise ValueError("Duplicate implementation tags")
                                implementations.update(extra)
                            row["implementations"] = benchmark_case(
                                case,
                                implementations,
                                timer,
                                name=f"{op_name}/{index:04d}",
                                verify_only=args.verify_only,
                            )
                            row["status"] = "verified" if args.verify_only else "measured"
                            passed += 1
                        except Exception:
                            row.update(status="failed", error=traceback.format_exc())
                            failed += 1
                        output.write(json.dumps(row, default=str) + "\n")
                        output.flush()
                        print(f"{op_name} {case.id}: {row['status']}", flush=True)
                        case._release()
                torch.npu.synchronize()
        completed = True
    finally:
        if not args.verify_only:
            _write(
                args.run_dir / "clock-readbacks.json",
                {
                    "before": getattr(clock_context, "before", None),
                    "samples": clock_context.samples,
                    "after_restore": getattr(clock_context, "after", None),
                },
            )
        after = source_fingerprint()
        changed = before != after
        _write(
            args.run_dir / "result.json",
            {
                "status": "passed"
                if completed and passed and not failed and not changed
                else "failed",
                "passed": passed,
                "failed": failed,
                "unsupported": unsupported,
                "source_changed": changed,
                "verify_only": args.verify_only,
            },
        )
    return 0 if completed and passed and not failed and not changed else 1


if __name__ == "__main__":
    sys.exit(main())
