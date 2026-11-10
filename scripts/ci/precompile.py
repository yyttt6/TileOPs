"""Bounded, guarded precompilation pilot; device measurement remains unchanged.

The compile-description/ReLU split is adapted from PR97. Until complete process
and child-process device-access auditing exists, even imports run under the
existing host reservation. Only reviewed pilot operators opt in.
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

PILOT_OPERATORS = frozenset({"ReLUFwdOp"})


def plan_for(op_name, source):
    sys.path[:0] = [str(source / "src"), str(source)]
    from benchmarks.api import cases
    from tileops.backend import _registry
    from tileops.manifest import load_manifest

    _registry.ensure_loaded()
    if _registry.LOAD_FAILURES:
        raise RuntimeError(_registry.LOAD_FAILURES)
    family = load_manifest()[op_name]["family"]
    importlib.import_module(f"tileops._kernels.ascend.{family}")
    op = getattr(importlib.import_module(f"tileops.{family}"), op_name)
    interfaces = tuple(op.interfaces.values())
    if len(interfaces) != 1:
        return [], [], "multiple interfaces require explicit support"
    implementations = [
        cls
        for key, cls in _registry.REGISTERED_KERNEL_TYPES.get(op_name, {}).items()
        if (key == "ascend" or key.startswith("ascend_")) and issubclass(cls, interfaces[0])
    ]
    if len(implementations) != 1:
        return [], [], "ambiguous or absent Ascend implementation"
    factory = getattr(implementations[0], "compilations", None)
    if factory is None:
        return [], [], "source revision has no compile-only contract"
    programs, skipped = {}, []
    for case in cases(op_name):
        found = tuple(factory(case._call.tensors, case._call.params))
        if not found:
            skipped.append(case.id)
        for item in found:
            programs.setdefault(item, case.id)
    return list(programs), skipped, None


def compile_programs(programs, workers):
    from tilelang.jit import par_compile

    groups = {}
    for item in programs:
        groups.setdefault(item.identity()[1:], []).append(item)
    compiled = 0
    effective = []
    for group in groups.values():
        n = min(workers, len(group))
        effective.append(n)
        item = group[0]
        built = par_compile(
            [x.program for x in group],
            out_idx=item.out_idx,
            target=item.target,
            target_host=item.target_host,
            execution_backend=item.execution_backend,
            pass_configs=item.pass_configs,
            compile_flags=item.compile_flags,
            num_workers=n,
            ignore_error=False,
        )
        if len(built) != len(group) or any(x is None for x in built):
            raise RuntimeError("Parallel compiler returned incomplete results")
        compiled += len(built)
    return compiled, effective


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--operators", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if not 1 <= a.workers <= 16:
        p.error("workers must be between 1 and 16")
    return run(a.source, a.operators.split(","), a.output, a.workers)


def run(source, operators, output, workers=4):
    if not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    source = source.resolve()
    sys.path[:0] = [str(source / "src"), str(source)]
    receipt = {
        "source": str(source),
        "requested_workers": workers,
        "cache_directory": os.environ.get("TILELANG_CACHE_DIR"),
        "operators": {},
        "status": "failed",
        "stage": "reservation",
    }
    start = time.monotonic()
    try:
        # This check precedes torch, torch_npu, TileLang and candidate imports.
        from benchmarks.ascend.guard import require_reservation

        meta = require_reservation()
        receipt["guard_owner"] = meta.get("owner")
        receipt["stage"] = "enumeration"
        for op in operators:
            if op not in PILOT_OPERATORS:
                receipt["operators"][op] = {"status": "unsupported"}
                continue
            began = time.monotonic()
            programs, skipped, reason = plan_for(op, source)
            row = {
                "programs": len(programs),
                "skipped_cases": skipped,
                "enumeration_seconds": time.monotonic() - began,
            }
            receipt["operators"][op] = row
            if reason or not programs:
                row.update(status="unsupported", reason=reason or "no compile descriptions")
                continue
            receipt["stage"] = "compilation"
            began = time.monotonic()
            compiled, effective = compile_programs(programs, workers)
            row.update(
                status="compiled",
                compiled=compiled,
                effective_workers=effective,
                compile_seconds=time.monotonic() - began,
            )
        receipt.update(status="passed", stage="done")
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        receipt["seconds"] = time.monotonic() - start
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
