"""Import or resume complete, guarded PR measurements. Device code always comes from the trusted harness."""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

MAX_CASES = 24
HARNESS_FILES = (
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
EXTENSIONS = {".py", ".yaml", ".sh", ".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(data):
    return digest(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())


def read(path):
    return json.loads(path.read_bytes())


def save(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=True) + "\n")
    temp.replace(path)


def source_hashes(root):
    paths = [
        p
        for directory in ("src/tileops", "workloads", "benchmarks", "scripts")
        for p in (root / directory).rglob("*")
        if p.is_file() and p.suffix in EXTENSIONS and "__pycache__" not in p.parts
    ]
    paths.extend(
        root / name for name in ("pyproject.toml", "constraints.txt", "constraints-runner-lock.txt")
    )
    return {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in paths}


def measurement_sources(sources):
    """CPU-only controller regression tests do not run in device workers."""
    return {name: sha for name, sha in sources.items() if not name.startswith("scripts/ci/tests/")}


def identity(provenance, registration, cache, toolchain, sources):
    if toolchain["status"] != "passed" or toolchain["errors"]:
        raise ValueError("Toolchain verification failed")
    return dict(
        base=provenance["base"],
        baseline_sha256=provenance["baseline_sha256"],
        correctness_protocol=provenance["correctness_protocol"],
        harness_sources=provenance["harness_sources"],
        registration=fingerprint(registration),
        cache=fingerprint(cache),
        reference=fingerprint(toolchain["reference_runtime"]),
        sources=fingerprint(measurement_sources(sources)),
    )


def safe_child(root, relative):
    path = root / relative
    if (
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError(f"Invalid evidence path: {relative}")
    return path


def archive_index(archive_path, expected_digest, destination):
    """Bind legacy evidence to the official artifact before consulting persistent runner storage."""
    if digest(archive_path.read_bytes()) != expected_digest.removeprefix("sha256:"):
        raise ValueError("Previous Actions artifact checksum mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        compact = [n for n in names if PurePosixPath(n).name == "checkpoint-index.json"]
        if compact:
            if len(compact) != 1:
                raise ValueError("Ambiguous checkpoint indexes")
            data = json.loads(archive.read(compact[0]))
            if data["schema"] != 1:
                raise ValueError("Unknown checkpoint index schema")
            save(destination, data)
            return

        def unique(name):
            matches = [n for n in names if PurePosixPath(n).name == name]
            if len(matches) != 1:
                raise ValueError(f"Missing or ambiguous artifact member: {name}")
            return matches[0]

        prov_name = unique("provenance.json")
        prefix = PurePosixPath(prov_name).parent
        origin = prefix.name
        if not re.fullmatch(r"[0-9]+-[0-9]+", origin):
            raise ValueError("Invalid originating run")

        def raw(rel):
            return archive.read(str(prefix / rel))

        def document(rel):
            return json.loads(raw(rel))

        prov = document("provenance.json")
        reg = document("registered.json")
        cache = json.loads(archive.read(unique("pr-cache.json")))
        toolchain = json.loads(archive.read(unique("pr-toolchain.json")))
        records = []
        sources = None
        for op in document("selection.json")["operators"]:
            pointer = prefix / (op + "-guard.json")
            if str(pointer) not in archive.NameToInfo:
                continue
            if not op.isidentifier():
                raise ValueError(f"Invalid operator {op}")
            g = document(pointer.name)
            if g["exit_code"] != 0:
                continue
            receipt = PurePosixPath(g["receipt"])
            if receipt.parent.name != origin or digest(raw(receipt.name)) != g["sha256"]:
                raise ValueError("Guard digest mismatch")
            guard = document(receipt.name)
            if not valid_guard(guard):
                continue
            result = document(op + "/result.json")
            if (
                result["status"] != "passed"
                or result["source_changed"]
                or result["selection_scope"] != "all-cases"
            ):
                continue
            measured_sources = document(op + "/sources.json")
            if sources is None:
                sources = measured_sources
            if sources != measured_sources:
                raise ValueError("Source changed between archived operators")
            records.append(
                dict(
                    op=op,
                    cases=reg["cases"][op],
                    origin=origin,
                    commit=prov["commit"],
                    directory=op,
                    guard=receipt.name,
                    guard_sha256=g["sha256"],
                    sources_sha256=fingerprint(measured_sources),
                    rows_sha256=digest(raw(op + "/results.jsonl")),
                    scope="all-cases",
                )
            )
        if sources is None:
            save(destination, dict(schema=1, identity=None, records=[]))
        else:
            save(
                destination,
                dict(
                    schema=1,
                    identity=identity(prov, reg, cache, toolchain, sources),
                    records=records,
                ),
            )


def valid_guard(g):
    return (
        g.get("phase") == "finished"
        and g.get("workload_completed") is True
        and g.get("workload_exit_code") == g.get("exit_code") == 0
        and g.get("performance_valid") is True
        and g.get("visibility_lost") is False
        and g.get("conflict_detected") is False
    )


def verify(record, evidence, cases, sources, hardware, replay, provenance):
    origin = record["origin"]
    if not re.fullmatch(r"[0-9]+-[0-9]+", origin):
        raise ValueError("Invalid run identity")
    root = safe_child(evidence, origin)
    target = safe_child(root, record["directory"])
    receipt = safe_child(root, record["guard"])
    raw = receipt.read_bytes()
    g = json.loads(raw)
    if digest(raw) != record["guard_sha256"] or not valid_guard(g):
        raise ValueError("Invalid guard")
    # The guard must protect the batch that explicitly listed this exact output path.
    label = receipt.name.removesuffix(".guard.json")
    assert g["owner"] == f"pr/{origin.split('-')[0]}/{label}"
    plan = read(receipt.with_name(label + ".plan.json"))
    assert any(
        j["op"] == record["op"] and Path(j["output"]).resolve() == target.resolve()
        for j in plan["jobs"]
    )
    summary = read(receipt.with_name(label + ".summary.json"))
    assert any(
        r["op"] == record["op"] and r["exit_code"] == 0 and r["fatal"] is None
        for r in summary["records"]
    )
    original = read(root / "provenance.json")
    assert original["commit"] == record["commit"] and plan["timer"] == "auto"
    assert all(
        original[field] == provenance[field]
        for field in ("base", "baseline_sha256", "correctness_protocol", "harness_sources")
    )
    recorded_sources = read(target / "sources.json")
    assert fingerprint(recorded_sources) == record["sources_sha256"]
    if record["commit"] == provenance["commit"]:
        assert recorded_sources == sources
    else:
        assert measurement_sources(recorded_sources) == measurement_sources(sources)
    assert read(target / "hardware.json") == hardware
    result = read(target / "result.json")
    assert result["status"] == "passed" and not result["failed"] and not result["source_changed"]
    assert result["selection_scope"] == record["scope"]
    ids = record["cases"]
    selected = set(ids)
    assert ids and len(ids) == len(selected) and ids == [c for c in cases if c in selected]
    if record["scope"] == "case-sample":
        case_plan = Path(plan["case_plan"])
        assert case_plan.resolve().is_relative_to(root) and read(case_plan)[record["op"]] == ids
    else:
        assert record["scope"] == "all-cases" and ids == cases
    rows_bytes = (target / "results.jsonl").read_bytes()
    assert digest(rows_bytes) == record["rows_sha256"]
    rows = [json.loads(line) for line in rows_bytes.splitlines()]
    assert len(rows) == result["passed"] and [r["case_id"] for r in rows] == ids
    for r in rows:
        c = r["correctness"]
        m = r["measurement"]
        assert (
            r["op"] == record["op"]
            and r["status"] == "passed"
            and r["input_preserved"]
            and r["correctness_source"] == "final-profile-repeat"
        )
        assert c["total_outputs"] == c["checked_outputs"] > 0 and not c.get("unchecked_reason")
        assert (
            m["warmup"] == 3
            and m["flush_bytes"] == 1 << 30
            and m["repeats"] == len(m["latency_samples_ms"]) == 15
        )
        assert math.isfinite(m["latency_ms"]) and m["latency_ms"] > 0
        assert Path(m["profile_coverage_evidence"]).resolve().is_relative_to(target.resolve())
        assert len(replay(m)) == 15
    return rows


def partitions(ops, registered_cases, satisfied):
    """Bound the work in one reservation; incomplete operators get explicit exhaustive case slices."""
    group = []
    count = 0
    for op in ops:
        all_cases = registered_cases[op]
        remaining = [c for c in all_cases if (op, c) not in satisfied]
        if not remaining:
            continue
        if group and (len(group) == 8 or count + len(remaining) > MAX_CASES):
            yield group
            group = []
            count = 0
        if remaining != all_cases or len(remaining) > MAX_CASES:
            for i in range(0, len(remaining), MAX_CASES):
                yield [(op, remaining[i : i + MAX_CASES], True)]
        else:
            group.append((op, remaining, False))
            count += len(remaining)
    if group:
        yield group


# This decoder-only recovery is qualified by replay of all original 900 windows.
# Bind the signed-off records, the original environment and the entire new
# measurement identity. Other changes continue to use the normal equality check.
TASK_ID_REPLAY_INDEX = "296d7157bc51971c4dc416435cec09273033aeaed92c9536fb6ecb8df1d2400d"
TASK_ID_REPLAY_IDENTITY = "68427ad22d759cb1ce17960f31c2e0359ffb16d4e3d53333a8cb4aa0472c9293"


def task_id_replay(prior, current):
    return (
        fingerprint(prior) == TASK_ID_REPLAY_INDEX
        and fingerprint(current) == TASK_ID_REPLAY_IDENTITY
    )


def measure(args):
    root = Path.cwd().resolve()
    harness = (root / "ci-harness/scripts/ci").resolve()
    sys.path[:0] = [str(harness), str(harness / "nightly_support")]
    import pr_performance as pr
    from baseline_contracts import comparison_identity
    from complete_profile import replay_coverage
    from device_batch import run as run_batch

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    origin = out.name
    run_id = os.environ["GITHUB_RUN_ID"]
    assert origin == run_id + "-" + os.environ["GITHUB_RUN_ATTEMPT"]
    plan = (
        pr.select(root, args.base)
        if not args.operators
        else dict(
            operators=args.operators.split(","),
            changed=[],
            target_base=args.base,
            selection_source="explicit-manual-operators",
        )
    )
    save(out / "selection.json", plan)
    if not plan["operators"]:
        (out / "summary.md").write_text("No affected Ascend operators detected.\n")
        return
    with (out / "__list__.log").open("x") as log:
        subprocess.run(
            [
                "npu-guard",
                "run",
                "--owner",
                f"pr/{run_id}/__list__",
                "--wait",
                "1800",
                "--timeout",
                "7200",
                "--",
                sys.executable,
                str(harness / "nightly_device.py"),
                "--source",
                str(root),
                "--op",
                "__list__",
                "--output",
                str(out / "registered.json"),
                "--verify-measured-output",
            ],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    reg = read(out / "registered.json")
    cases = reg["cases"]
    if args.operators:
        assert set(plan["operators"]) <= set(reg["operators"])
    ops = pr.affected_composites(plan["operators"], reg)
    assert ops and set(ops) <= set(reg["operators"])
    plan["operators"] = ops
    save(out / "selection.json", plan)
    prov = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        base=args.base,
        correctness_protocol="final-profile-output-v1",
        harness_sources={f: digest((harness / f).read_bytes()) for f in HARNESS_FILES},
        baseline_sha256=digest(args.baseline.read_bytes()),
    )
    save(out / "provenance.json", prov)
    sources = source_hashes(root)
    toolchain = read(args.toolchain)
    cache = read(args.cache)
    key = identity(prov, reg, cache, toolchain, sources)
    records = []
    satisfied = set()
    comparisons = []
    baseline = read(args.baseline)
    indexed = {}
    for section in ("rows", "historical_rows"):
        for row in baseline.get(section, []):
            k = comparison_identity(row["op"], row["case"], row["workload"]["params"])
            group = indexed.setdefault(
                k,
                {
                    "rows": [],
                    "historical_rows": [],
                    "retired_measurement_ids": baseline.get("retired_measurement_ids", []),
                },
            )
            group[section].append(row)

    def persist():
        save(out / "checkpoint-index.json", dict(schema=1, identity=key, records=records))
        save(out / "comparisons.json", comparisons)

    def admit(record, *, verified_decoder_transition=False):
        op = record["op"]
        record_sources, record_provenance = sources, prov
        if verified_decoder_transition:
            # The exact index binds guard/source/row hashes and original run.
            # Check the original source signature and the original receipt, but
            # always replay all 15 windows with the current, corrected decoder.
            original = safe_child(out.parent, record["origin"])
            record_sources = read(safe_child(original, record["directory"]) / "sources.json")
            record_provenance = read(original / "provenance.json")
            assert fingerprint(measurement_sources(record_sources)) == prior["identity"]["sources"]
            for field in ("base", "baseline_sha256", "correctness_protocol", "harness_sources"):
                assert record_provenance[field] == prior["identity"][field]
        rows = verify(
            record,
            out.parent,
            cases[op],
            record_sources,
            reg["hardware"],
            replay_coverage,
            record_provenance,
        )
        keys = {(op, row["case_id"]) for row in rows}
        assert not (keys & satisfied)
        additions = []
        for row in rows:
            cid = row["case_id"]
            k = comparison_identity(op, cid, row["params"])
            matches = pr.matching_baselines(indexed.get(k, {"rows": []}), op, cid, row["params"])
            m = row["measurement"]
            us = m["latency_ms"] * 1000
            additions.append(
                dict(
                    op=op,
                    case=cid,
                    tileops_us=us,
                    measurement_protocol=m["measurement_protocol"],
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
        satisfied.update(keys)
        comparisons.extend(additions)
        records.append(record)

    if args.prior and args.prior.is_file():
        prior = read(args.prior)
        transition = task_id_replay(prior, key)
        if prior["identity"] == key or transition:
            for record in prior["records"]:
                if record["op"] not in ops:
                    continue
                try:
                    admit(record, verified_decoder_transition=transition)
                except (AssertionError, OSError, ValueError, KeyError) as exc:
                    if transition:
                        raise
                    print(
                        f"Rejected checkpoint for {record['op']}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
        else:
            print("Prior measurement identity differs; measuring the selection.", flush=True)
    persist()
    print(
        f"Recovered {len(satisfied)} cases from {len(records)} valid reservations or operator records",
        flush=True,
    )
    for i, group in enumerate(partitions(ops, cases, satisfied)):
        label = f"batch-{i:04d}"
        jobs = []
        case_plan = None
        for op, ids, partial in group:
            directory = Path("parts") / label / op if partial else Path(op)
            (out / directory).parent.mkdir(parents=True, exist_ok=True)
            jobs.append(dict(op=op, output=str(out / directory)))
            if partial:
                case_plan = out / (label + ".cases.json")
                save(case_plan, {op: ids})
        report = run_batch(
            root,
            jobs,
            out,
            label,
            f"pr/{run_id}/{label}",
            timer="auto",
            case_plan=str(case_plan) if case_plan else None,
        )
        assert report["code"] == 0 and valid_guard(report["guard"]), (
            f"{label}: invalid host-guard result {report['code']}"
        )
        for (op, ids, partial), job in zip(group, jobs, strict=True):
            target = Path(job["output"])
            record = dict(
                op=op,
                cases=ids,
                origin=origin,
                commit=prov["commit"],
                directory=str(target.relative_to(out)),
                guard=Path(report["guard_receipt"]).name,
                guard_sha256=report["guard_sha256"],
                sources_sha256=fingerprint(sources),
                rows_sha256=digest((target / "results.jsonl").read_bytes()),
                scope="case-sample" if partial else "all-cases",
            )
            admit(record)
            persist()
    expected = {(op, c) for op in ops for c in cases[op]}
    assert satisfied == expected
    summary = f"Verified {len(satisfied)} cases across {len(ops)} selected operators. Full per-case ratios are in comparisons.json.\n"
    (out / "summary.md").write_text(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("index")
    a.add_argument("archive", type=Path)
    a.add_argument("sha256")
    a.add_argument("destination", type=Path)
    a = sub.add_parser("measure")
    a.add_argument("--base", required=True)
    a.add_argument("--baseline", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--cache", type=Path, required=True)
    a.add_argument("--toolchain", type=Path, required=True)
    a.add_argument("--prior", type=Path)
    a.add_argument("--operators", default="")
    opts = parser.parse_args()
    if opts.mode == "index":
        archive_index(opts.archive, opts.sha256, opts.destination)
    else:
        measure(opts)
