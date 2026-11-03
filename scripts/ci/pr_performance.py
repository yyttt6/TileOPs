"""Changed-operator correctness and audited performance against a frozen baseline."""

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def select(root, base):
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", base, "HEAD"], cwd=root, text=True
    ).splitlines()
    kernels = root / "src/tileops/kernels/ascend"
    files = list(kernels.rglob("*.py"))
    selected = set()
    direct = set()
    reasons = {}
    for path in changed:
        p = root / path
        if not p.exists() or p.suffix != ".py" or p.name == "__init__.py":
            continue
        if path.startswith("src/tileops/kernels/ascend/"):
            # Include the kernel family for helper/template changes: dependencies may be dynamic.
            tree = ast.parse(p.read_text())
            direct.update(
                n.value
                for n in ast.walk(tree)
                if isinstance(n, ast.Constant)
                and isinstance(n.value, str)
                and n.value.endswith("Op")
                and n.value.isidentifier()
            )
            family = path.split("/")[4]
            candidates = list((kernels / family).rglob("*.py"))
        elif path.startswith(("src/tileops/ops/", "workloads/", "benchmarks/_cases/")):
            candidates = [p]
        elif path.startswith(("src/tileops/backend/", "src/tileops/manifest/")):
            candidates = files
        else:
            continue
        ops = set()
        for candidate in candidates:
            tree = ast.parse(candidate.read_text())
            ops.update(
                n.value
                for n in ast.walk(tree)
                if isinstance(n, ast.Constant)
                and isinstance(n.value, str)
                and n.value.endswith("Op")
                and n.value.isidentifier()
            )
        selected.update(ops)
        reasons[path] = sorted(ops)
    return dict(changed=changed, operators=sorted(direct or selected), reasons=reasons)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = Path.cwd()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = select(root, a.base)

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2, default=str) + "\n")

    save("selection.json", plan)
    if not plan["operators"]:
        (out / "summary.md").write_text("No affected Ascend operators detected.\n")
        return

    def guarded(op, target):
        cmd = [
            "npu-guard",
            "run",
            "--owner",
            f"pr/{os.environ.get('GITHUB_RUN_ID', 'local')}/{op}",
            "--wait",
            "1800",
            "--timeout",
            "7200",
            "--",
            sys.executable,
            str(HERE / "nightly_device.py"),
            "--source",
            str(root),
            "--output",
            str(target),
            "--op",
            op,
        ]
        with (out / f"{op}.log").open("w") as log:
            return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode

    assert guarded("__list__", out / "registered.json") == 0
    registered = json.loads((out / "registered.json").read_text())
    ops = sorted(set(plan["operators"]) & set(registered["operators"]))
    assert ops, "Changed operators have no registered Ascend implementation"
    plan["operators"] = ops
    save("selection.json", plan)
    baseline = json.loads(a.baseline.read_text())
    save(
        "provenance.json",
        dict(
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            base=a.base,
            baseline_sha256=hashlib.sha256(a.baseline.read_bytes()).hexdigest(),
        ),
    )
    # Complete correctness first. No performance is measured if this gate fails.
    for op in ops:
        cmd = [
            sys.executable,
            "scripts/benchmark_ascend.py",
            "--guard",
            "/usr/local/libexec/ascend-npu/npu_guard.py",
            "--owner",
            f"pr/correctness/{op}",
            "--wait",
            "1800",
            "--timeout",
            "7200",
            "--run-dir",
            str(out / f"correctness-{op}"),
            "--verify-only",
            "--op",
            op,
        ]
        subprocess.run(cmd, check=True)
    sys.path.insert(0, str(HERE / "nightly_support"))
    import math

    from complete_profile import complete_digest, decode, pure_aicpu_base
    from tools.measure_baselines import _parse_ffts_profile

    rows = []
    for op in ops:
        target = out / op
        code = guarded(op, target)
        save(f"{op}-guard.json", dict(exit_code=code))
        assert code == 0, f"{op}: invalid guard receipt {code}"
        result = json.loads((target / "result.json").read_text())
        assert (
            result["status"] == "passed" and not result["source_changed"] and not result["failed"]
        )
        entries = [json.loads(line) for line in (target / "results.jsonl").read_text().splitlines()]
        assert len(entries) == len(registered["cases"][op])
        assert {r["case_id"] for r in entries} == set(registered["cases"][op])
        for line in (target / "results.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert row["status"] == "passed"
            coverage = json.loads(Path(row["measurement"]["profile_coverage_evidence"]).read_text())
            assert len(coverage) >= 15
            for c in coverage[-15:]:
                assert complete_digest(c["path"]) == c["complete_sha256"]
                extra = decode(c["path"])
                try:
                    replay = _parse_ffts_profile(Path(c["path"]), 1)
                except ValueError as exc:
                    if str(exc) != "No valid FFTS activity":
                        raise
                    replay = pure_aicpu_base(extra)
                assert math.isclose(
                    replay["latency_ms"] + extra["added_us"] / 1000,
                    c["result"]["latency_ms"],
                    rel_tol=1e-10,
                )
            matches = [
                b
                for b in baseline["rows"]
                if b["op"] == op
                and b["case"] == row["case_id"]
                and b["workload"]["params"] == row["params"]
            ]
            us = row["measurement"]["latency_ms"] * 1000
            rows.append(
                dict(
                    op=op,
                    case=row["case_id"],
                    tileops_us=us,
                    comparisons=[
                        dict(
                            library=b["library"],
                            baseline_us=b["latency_us"],
                            ratio=b["latency_us"] / us,
                            protocol=b["protocol"],
                        )
                        for b in matches
                    ],
                )
            )
        save("comparisons.json", rows)
    lines = [
        "| Operator | Case | TileOPs µs | Baseline | Baseline µs | Baseline / TileOPs |",
        "|---|---|---:|---|---:|---:|",
    ]
    for r in rows:
        for b in r["comparisons"] or [
            dict(library="Missing matching baseline", baseline_us=0, ratio=0)
        ]:
            lines.append(
                f"| {r['op']} | {r['case']} | {r['tileops_us']:.3f} | {b['library']} | {b['baseline_us']:.3f} | {b['ratio']:.3f} |"
            )
    lines += [
        "",
        "Historical native v2 baselines are retained; new TileOPs measurements use v3. Missing baseline is not a performance pass. Ratios are reported, not a new merge threshold.",
    ]
    summary = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(summary)


if __name__ == "__main__":
    main()
