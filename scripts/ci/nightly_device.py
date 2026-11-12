"""Measure one frozen registered operator under the shared physical32 reservation."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "nightly_support"))


def registered_ascend_ops(registered):
    """Include all named Ascend implementations, including train/infer variants."""
    return sorted(
        op
        for op, types in registered.items()
        if any(name == "ascend" or name.startswith("ascend_") for name in types)
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--op", required=True)
    p.add_argument("--verify-measured-output", action="store_true")
    p.add_argument("--timer-mode", choices=("auto", "session", "isolated"), default="auto")
    p.add_argument(
        "--case-plan", type=Path, help="Explicit diagnostic sample; never a full-coverage receipt"
    )
    a = p.parse_args(argv)
    sys.path[:0] = [str(a.source / "src"), str(a.source)]
    from benchmarks.ascend.guard import require_reservation

    require_reservation()
    import torch
    import torch_npu
    from complete_profile import make_timer

    from benchmarks._cases import entry
    from benchmarks.api import cases
    from benchmarks.ascend.coverage import registered_coverage
    from benchmarks.ascend.runner import benchmark_case, source_fingerprint
    from tileops.backend import _registry as registry
    from tileops.manifest import load_manifest
    from workloads.device import set_run_device
    from workloads.numerics import Request, check_inputs_preserved

    torch.npu.set_device(0)
    set_run_device("npu")
    props = torch.npu.get_device_properties(0)
    hardware = dict(
        device=torch.npu.get_device_name(0),
        cube=props.cube_core_num,
        vector=props.vector_core_num,
        clock_policy="platform-default",
    )
    assert "Ascend950" in hardware["device"] and (hardware["cube"], hardware["vector"]) == (32, 64)
    registry.ensure_loaded()
    assert not registry.LOAD_FAILURES, registry.LOAD_FAILURES
    coverage = registered_coverage()
    selected = coverage["operators"]
    if a.op == "__list__":
        a.output.write_text(
            json.dumps(
                {
                    **coverage,
                    "hardware": hardware,
                    "cases": {op: [c.id for c in cases(op)] for op in selected},
                }
            )
            + "\n"
        )
        return
    assert a.op in selected
    from reference_runtime import initialize_reference_runtime

    initialize_reference_runtime(a.op)
    a.output.mkdir(parents=True, exist_ok=False)
    before = source_fingerprint()
    (a.output / "sources.json").write_text(json.dumps(before, sort_keys=True))
    (a.output / "hardware.json").write_text(json.dumps(hardware))
    # Compile only reviewed opt-in kernels, under this existing reservation.
    # Reuse this process to avoid a second import/registry startup.
    from precompile import PILOT_OPERATORS
    from precompile import run as precompile

    if a.op in PILOT_OPERATORS:
        assert precompile(a.source, [a.op], a.output / "precompile.json") == 0
    manifest = load_manifest()
    module = __import__("importlib").import_module(f"tileops.{manifest[a.op]['family']}")
    op_class = getattr(module, a.op)
    golden_identity = None
    if os.environ.get("ASCEND950_GOLDEN_CACHE_DIR"):
        from golden_cache import contract_identity

        try:
            golden_identity = contract_identity(a.source)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Golden cache disabled: {exc}", flush=True)
    selected_cases = cases(a.op)
    if a.case_plan:
        ids = json.loads(a.case_plan.read_text())[a.op]
        found = {c.id: c for c in selected_cases}
        assert ids and len(ids) == len(set(ids)) and set(ids) <= set(found)
        selected_cases = [found[i] for i in ids]
    count = failed = 0
    started = time.monotonic()
    with (a.output / "results.jsonl").open("x") as stream:
        for index, case in enumerate(selected_cases):
            row = dict(
                op=a.op, case_id=case.id, params=case.params, family=manifest[a.op]["family"]
            )
            begin = time.monotonic()
            golden = None
            try:
                seed = int.from_bytes(
                    hashlib.sha256(f"{a.op}::{case.id}".encode()).digest()[:4], "little"
                )
                torch.manual_seed(seed)
                op = op_class(**case.arguments)
                impl = entry(a.op).binder(op, case)
                args = case.inputs if impl.args is None else impl.args
                mode = "session" if a.timer_mode == "auto" else a.timer_mode
                if a.verify_measured_output:
                    from measured_correctness import measure_profile_and_verify

                    if golden_identity is not None:
                        from golden_cache import for_case

                        golden = for_case(a.source, a.op, case.id, golden_identity)
                    correctness, measurement = measure_profile_and_verify(
                        case,
                        impl,
                        torch,
                        torch_npu,
                        args,
                        a.output / "profiles" / f"{index:04d}",
                        mode=a.timer_mode,
                        golden_cache=golden,
                    )

                else:
                    checks = benchmark_case(
                        case, {"tileops": impl}, None, name=case.id, verify_only=True
                    )
                    timer = make_timer(a.output / "profiles" / f"{index:04d}-{mode}", mode=mode)
                    measurement = timer(torch, torch_npu, impl.run, args, 3, 15, reset=impl.reset)
                    check_inputs_preserved(
                        Request(impl.run, args, impl.reset, preserve_inputs=True), case.inputs
                    )
                    correctness = checks["tileops"]["correctness"]
                measurement.update(warmup=3, repeats=15, flush_bytes=1 << 30)
                row.update(
                    status="passed",
                    seed=seed,
                    correctness=correctness,
                    correctness_source=(
                        "final-profile-repeat" if a.verify_measured_output else "separate-run"
                    ),
                    measurement=measurement,
                    input_preserved=True,
                )
                count += 1
            except Exception:
                row.update(status="failed", error=__import__("traceback").format_exc())
                failed += 1
            if golden is not None:
                row["golden_cache"] = golden.events
            row["wall_seconds"] = time.monotonic() - begin
            stream.write(json.dumps(row, default=str) + "\n")
            stream.flush()
            print(a.op, case.id, row["status"], flush=True)
            case._release()
            if failed and a.verify_measured_output:
                break
    torch.npu.synchronize()
    changed = before != source_fingerprint()
    result = dict(
        status="passed" if count and not failed and not changed else "failed",
        passed=count,
        failed=failed,
        source_changed=changed,
        seconds=time.monotonic() - started,
        selection_scope="case-sample" if a.case_plan else "all-cases",
    )
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
