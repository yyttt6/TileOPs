"""Measure one frozen registered operator under the shared physical32 reservation."""

import argparse
import hashlib
import json
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--op", required=True)
    p.add_argument("--verify-measured-output", action="store_true")
    a = p.parse_args()
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
    from tileops.backend import registry
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
    a.output.mkdir(parents=True, exist_ok=False)
    before = source_fingerprint()
    (a.output / "sources.json").write_text(json.dumps(before, sort_keys=True))
    (a.output / "hardware.json").write_text(json.dumps(hardware))
    manifest = load_manifest()
    module = __import__("importlib").import_module(f"tileops.{manifest[a.op]['family']}")
    op_class = getattr(module, a.op)
    count = failed = 0
    started = time.monotonic()
    with (a.output / "results.jsonl").open("x") as stream:
        for index, case in enumerate(cases(a.op)):
            row = dict(
                op=a.op, case_id=case.id, params=case.params, family=manifest[a.op]["family"]
            )
            begin = time.monotonic()
            try:
                seed = int.from_bytes(
                    hashlib.sha256(f"{a.op}::{case.id}".encode()).digest()[:4], "little"
                )
                torch.manual_seed(seed)
                op = op_class(**case.arguments)
                impl = entry(a.op).binder(op, case)
                args = case.inputs if impl.args is None else impl.args
                timer = make_timer(a.output / "profiles" / f"{index:04d}")
                if a.verify_measured_output:
                    from measured_correctness import measure_and_verify

                    correctness, measurement = measure_and_verify(
                        case,
                        impl,
                        lambda run, timer=timer, args=args, reset=impl.reset: timer(
                            torch, torch_npu, run, args, 3, 15, reset=reset
                        ),
                        expected_calls=1 + 3 + 15,
                    )
                else:
                    checks = benchmark_case(
                        case, {"tileops": impl}, None, name=case.id, verify_only=True
                    )
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
    )
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
