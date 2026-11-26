"""Changed-operator timing with output verification against a frozen baseline."""

import argparse
import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def canonical_contract(value):
    """Normalize JSON representations of non-finite workload parameters only."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, str) and value in ("inf", "-inf"):
        return "Infinity" if value == "inf" else "-Infinity"
    if isinstance(value, dict):
        return {key: canonical_contract(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_contract(item) for item in value]
    return value


def matching_baselines(baseline, op, case, params):
    """Match a saved definition, including preserved pre-migration contracts."""
    from baseline_contracts import comparison_identity

    target = comparison_identity(op, case, params)

    retired = set(baseline.get("retired_measurement_ids", []))

    def alias_rank(row):
        return (row["op"] != op or row["case"] != case, bool(row.get("requested_library")))

    candidates = sorted(baseline["rows"], key=alias_rank) + sorted(
        baseline.get("historical_rows", []), key=alias_rank
    )
    matches = [
        b
        for b in candidates
        if b.get("measurement_id") not in retired
        and comparison_identity(b["op"], b["case"], b["workload"]["params"]) == target
    ]
    # Current rows precede history; never compare two versions of the same provider.
    providers = {}
    for row in matches:
        providers.setdefault(row.get("actual_library", row["library"]), row)
    return list(providers.values())


# Reviewed additive runtime transitions. Bind both complete product trees rather
# than a PR number, branch, or a candidate-supplied list. Any further product
# change falls back to ordinary impact analysis, including transitive delegates.
# The upstream12 conformance audit compares all 2,267 legacy cases and the
# existing dispatch and workload definitions on this exact pair.
REVIEWED_ADDITIONS = {
    (
        "af323b414b4de7c99661a2316efd3cea5e80b1c6841257d27b1a1f91f1b48977",
        "f7c0f8da1715e4e5bcaa2d28e1417f41b92154b680c6180708b6a0d52f1b6efe",
    ),
}


def declared_ops(root, revision, paths):
    """Read top-level operator declarations in the reviewed block-style specs."""
    names = set()
    for path in paths:
        blob = subprocess.run(["git", "show", f"{revision}:{path}"], cwd=root, capture_output=True)
        if blob.returncode:
            continue
        names.update(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*Op):[ \t]*$", blob.stdout.decode(), re.M))
    return names


def product_tree(root, revision):
    entries = subprocess.check_output(["git", "ls-tree", "-rz", "--full-tree", revision], cwd=root)
    return hashlib.sha256(
        b"\0".join(
            entry
            for entry in entries.split(b"\0")
            if entry
            and not entry.split(b"\t", 1)[1].startswith((b"docs/", b".github/", b"scripts/ci/"))
        )
    ).hexdigest()


def registry_delta(root, base, path, current):
    """A literal registry edit with identical surrounding code affects just its changed entries."""
    from gemm_selection import _read, _registry, _signature

    old = _registry(_read(root, base, path))
    new = _registry(current)
    if old is None or new is None or _signature(old[1]) != _signature(new[1]):
        return None
    dynamic_names = {"ENTRIES", "globals", "locals", "eval", "exec", "__import__"}
    if any(
        isinstance(node, ast.Name) and node.id in dynamic_names
        for statement in new[1]
        for node in ast.walk(statement)
    ):
        return None
    before, after = old[0], new[0]
    if [k for k in before if k in after] != [k for k in after if k in before]:
        return None
    return (before.keys() ^ after.keys()) | {
        name
        for name in before.keys() & after.keys()
        if ast.dump(before[name]) != ast.dump(after[name])
    }


def manifest_entry_delta(root, base, path):
    """Resolve YAML aliases before comparing entries; global type changes take the broad path."""
    if Path(path).name == "types.yaml":
        return None
    try:
        import yaml
    except ImportError:
        return None
    entries = []
    for rev in (base, "HEAD"):
        result = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=root, capture_output=True)
        if result.returncode:
            entries.append({})
            continue
        try:
            data = yaml.safe_load(result.stdout)
        except yaml.YAMLError:
            return None
        if not isinstance(data, dict) or not all(
            isinstance(k, str) and k.isidentifier() and k.endswith("Op") for k in data
        ):
            return None
        entries.append(data)
    old, new = entries
    return (old.keys() ^ new.keys()) | {op for op in old.keys() & new.keys() if old[op] != new[op]}


def select(root, base):
    # An old PR tree can lag behind the target branch. A two-tree diff would
    # count target-only changes as PR changes and may select the entire library.
    target_base = subprocess.check_output(
        ["git", "rev-parse", f"{base}^{{commit}}"], cwd=root, text=True
    ).strip()
    bases = subprocess.check_output(
        ["git", "merge-base", "--all", target_base, "HEAD"], cwd=root, text=True
    ).splitlines()
    if len(bases) != 1:
        raise ValueError("Selection requires one unambiguous merge base")
    base = bases[0]
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", base, "HEAD"], cwd=root, text=True
    ).splitlines()
    shared = any(
        path.startswith(("src/tileops/backend/", "src/tileops/manifest/")) for path in changed
    )
    if shared:
        if (product_tree(root, base), product_tree(root, "HEAD")) in REVIEWED_ADDITIONS:
            specs = [
                p
                for p in changed
                if p.startswith("src/tileops/manifest/spec/") and p.endswith(".yaml")
            ]
            old_ops = declared_ops(root, base, specs)
            new_ops = declared_ops(root, "HEAD", specs)
            if not old_ops < new_ops:
                raise ValueError(
                    "Reviewed additive transition no longer describes a manifest addition"
                )
            added = sorted(new_ops - old_ops)
            return dict(
                changed=changed,
                operators=added,
                reasons={"reviewed-additive-transition": added},
                target_base=target_base,
                comparison_base=base,
                target_is_ancestor=base == target_base,
            )
    kernels = root / "src/tileops/_kernels/ascend"
    files = list(kernels.rglob("*.py"))
    selected = set()
    direct = {}
    families = {}
    shared_families = set()
    reasons = {}
    # These public builtins route to Ascend without register_kernel_type literals.
    # Keep the existing conservative helper-family expansion for every other path.
    builtin_sources = {
        "src/tileops/_kernels/ascend/convolution/conv1d.py": "Conv1dFwdOp",
        "src/tileops/_kernels/ascend/convolution/conv1d_shallow.py": "Conv1dFwdOp",
        "src/tileops/_kernels/ascend/transform/hadamard.py": "HadamardTransformFwdOp",
    }
    from gemm_selection import REGISTRY, WORKLOADS, select_gemm_harness

    gemm_paths = {REGISTRY, WORKLOADS} & set(changed)
    for path in changed:
        if path in gemm_paths:
            continue
        if path in builtin_sources:
            selected.add(builtin_sources[path])
        p = root / path
        if path.startswith("src/tileops/manifest/spec/") and path.endswith(".yaml"):
            delta = manifest_entry_delta(root, base, path)
            if delta is not None:
                selected.update(delta)
                reasons[path] = sorted(delta)
                continue
            selected.update(builtin_sources.values())
            candidates = files
        elif not p.exists() or p.suffix != ".py" or p.name == "__init__.py":
            continue
        elif path.startswith("src/tileops/_kernels/ascend/"):
            # Include the kernel family for helper/template changes: dependencies may be dynamic.
            tree = ast.parse(p.read_text())
            family = path.split("/")[4]
            direct.setdefault(family, set()).update(
                n.value
                for n in ast.walk(tree)
                if isinstance(n, ast.Constant)
                and isinstance(n.value, str)
                and n.value.endswith("Op")
                and n.value.isidentifier()
            )
            family = path.split("/")[4]
            candidates = list((kernels / family).rglob("*.py"))
        elif path.startswith("benchmarks/_cases/"):
            delta = registry_delta(root, base, path, ast.parse(p.read_text()))
            if delta is not None:
                selected.update(delta)
                reasons[path] = sorted(delta)
                continue
            candidates = [p]
        elif path.startswith(("src/tileops/_ops/", "workloads/")):
            candidates = [p]
        elif path.startswith(("src/tileops/backend/", "src/tileops/manifest/")):
            # Formatting and comments in a shared parser cannot affect device programs.
            old = subprocess.run(["git", "show", f"{base}:{path}"], cwd=root, capture_output=True)
            if old.returncode == 0 and ast.dump(ast.parse(old.stdout)) == ast.dump(
                ast.parse(p.read_text())
            ):
                continue
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
        if path.startswith("src/tileops/_kernels/ascend/"):
            families.setdefault(family, set()).update(ops)
            has_registration = any(
                isinstance(n, ast.Constant)
                and isinstance(n.value, str)
                and n.value.endswith("Op")
                and n.value.isidentifier()
                for n in ast.walk(ast.parse(p.read_text()))
            )
            exists_in_base = (
                subprocess.run(
                    ["git", "cat-file", "-e", f"{base}:{path}"],
                    cwd=root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode
                == 0
            )
            if not has_registration and exists_in_base:
                shared_families.add(family)
        else:
            selected.update(ops)
        reasons[path] = sorted(ops)
    if gemm_paths:
        gemm_operators = select_gemm_harness(root, base, gemm_paths)
        selected.update(gemm_operators)
        for path in gemm_paths:
            reasons[path] = sorted(gemm_operators)
    for family, ops in families.items():
        selected.update(ops if family in shared_families else (direct.get(family) or ops))
    return dict(
        changed=changed,
        operators=sorted(selected),
        reasons=reasons,
        target_base=target_base,
        comparison_base=base,
        target_is_ancestor=base == target_base,
    )


def affected_composites(selected, registered):
    """Retest supported composites when any of their transitive delegates changes."""
    affected = set(selected)
    while True:
        added = {
            name
            for name, children in registered.get("delegates", {}).items()
            if set(children) & affected
        }
        if added <= affected:
            return sorted(affected & set(registered["operators"]))
        affected.update(added)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--operators",
        default="",
        help="Explicit operator selection for manual performance dispatch",
    )
    a = p.parse_args()
    root = Path.cwd()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if a.operators:
        plan = dict(
            changed=[],
            operators=a.operators.split(","),
            target_base=a.base,
            selection_source="explicit-manual-operators",
        )
    else:
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
            "--verify-measured-output",
        ]
        with (out / f"{op}.log").open("w") as log:
            return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode

    assert guarded("__list__", out / "registered.json") == 0
    registered = json.loads((out / "registered.json").read_text())
    if a.operators:
        assert set(plan["operators"]) <= set(registered["operators"]), (
            "Explicit selection contains an unsupported operator"
        )
    ops = affected_composites(plan["operators"], registered)
    assert ops and set(ops) <= set(registered["operators"]), (
        "Changed operators have no registered Ascend implementation"
    )
    plan["operators"] = ops
    save("selection.json", plan)
    baseline = json.loads(a.baseline.read_text())
    save(
        "provenance.json",
        dict(
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            base=a.base,
            correctness_protocol="final-profile-output-v1",
            harness_sources={
                name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                for name in (
                    "pr_performance.py",
                    "gemm_selection.py",
                    "device_batch.py",
                    "nightly_support/session_profile.py",
                    "baseline_contracts.py",
                    "nightly_device.py",
                    "precompile.py",
                    "measured_correctness.py",
                    "golden_cache.py",
                    "reference_cache_compat.py",
                    "golden_cache_policy.json",
                    "nightly_support/complete_profile.py",
                )
            },
            baseline_sha256=hashlib.sha256(a.baseline.read_bytes()).hexdigest(),
        ),
    )
    # Each case checks its final timed output; the device helper stops on failure.
    sys.path.insert(0, str(HERE / "nightly_support"))

    from device_batch import groups
    from device_batch import run as run_batch

    rows = []
    live = {}
    for index, op in enumerate(ops):
        if op not in live:
            batch = groups(ops[index:], 8)[0]
            report = run_batch(
                root,
                [dict(op=name, output=str(out / name)) for name in batch],
                out,
                f"batch-{index:04d}",
                f"pr/{os.environ.get('GITHUB_RUN_ID', 'local')}/batch-{index:04d}",
                timer=os.environ.get("ASCEND950_TIMER_MODE", "auto"),
            )
            for name in batch:
                live[name] = report["code"]
                save(
                    f"{name}-guard.json",
                    dict(
                        exit_code=report["code"],
                        receipt=report["guard_receipt"],
                        sha256=report["guard_sha256"],
                    ),
                )
        target = out / op
        code = live[op]
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
            assert row["correctness_source"] == "final-profile-repeat"
            from complete_profile import replay_coverage

            replay_coverage(row["measurement"])
            assert result.get("selection_scope", "all-cases") == "all-cases"
            matches = matching_baselines(baseline, op, row["case_id"], row["params"])
            us = row["measurement"]["latency_ms"] * 1000
            rows.append(
                dict(
                    op=op,
                    case=row["case_id"],
                    tileops_us=us,
                    measurement_protocol=row["measurement"]["measurement_protocol"],
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
        "Historical native v2 baselines are retained; new TileOPs measurements record their v3 isolated or v4 bracketed-session protocol explicitly. Missing baseline is not a performance pass. Ratios are reported, not a new merge threshold.",
    ]
    summary = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(summary)


if __name__ == "__main__":
    main()
