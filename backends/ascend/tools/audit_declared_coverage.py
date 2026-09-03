#!/usr/bin/env python3
"""Compare what each op *declares* it supports against what was actually measured.

``correctness.passed=true`` means "right on the cases that were run". It does not mean
"the cases that were run covered what the op claims to support". Those are different
questions, and this project has now conflated a coverage metric with a capability claim
three times (F015, §13.11, D023) -- this is the fourth place it can happen.

The manifest is the authority on what an op claims: each entry carries
``workloads[]``, and each workload carries its own ``dtypes``. The cross product of
(workload label, dtype) is what the op says it handles. This script diffs that against
the (label, dtype) pairs actually present in the op's coverage JSON and reports what
was never exercised.

An untested declared pair is not automatically a bug -- an op may legitimately refuse a
dtype, as GQA does for bf16 under D022. But a *silent* gap is: the op claims support,
nothing measured it, and the coverage number counted the op as done. Declared refusals
show up in the JSON (``status=unsupported_dtype``) and are reported separately from
pairs that are simply missing.

Usage:
    python audit_declared_coverage.py [--op NAME] [--json]
"""
import argparse
import glob
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "TileOPs", "src", "tileops", "manifest")
COVERAGE = os.path.join(ROOT, "tileops-ascend-harness", "coverage")


def declared():
    """op -> {(workload_label, dtype)} that the manifest claims."""
    claims = {}
    for path in glob.glob(os.path.join(MANIFEST, "*.yaml")):
        try:
            doc = yaml.safe_load(open(path)) or {}
        except yaml.YAMLError:
            continue
        for op, entry in doc.items():
            if not isinstance(entry, dict):
                continue
            pairs = set()
            for index, workload in enumerate(entry.get("workloads") or []):
                if not isinstance(workload, dict):
                    continue
                label = workload.get("label") or f"workload[{index}]"
                for dtype in workload.get("dtypes") or []:
                    pairs.add((label, dtype))
            if pairs:
                claims[op] = pairs
    return claims


def measured(op):
    """({(label, dtype)} exercised, {(label, dtype)} explicitly refused)."""
    path = os.path.join(COVERAGE, f"{op}.json")
    if not os.path.exists(path):
        return None, None
    try:
        data = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return None, None
    exercised, refused = set(), set()
    for case in (data.get("correctness") or {}).get("per_case") or []:
        if not isinstance(case, dict):
            continue
        label = case.get("label")
        dtype = case.get("dtype")
        if label is None or dtype is None:
            continue
        if case.get("status") == "unsupported_dtype" or case.get("passed") is None:
            refused.add((label, dtype))
        elif case.get("passed"):
            exercised.add((label, dtype))
    return exercised, refused


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--op")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    claims = declared()
    rows, no_json, no_labels = [], [], []
    for op in sorted(claims):
        if args.op and op != args.op:
            continue
        exercised, refused = measured(op)
        if exercised is None:
            no_json.append(op)
            continue
        if not exercised and not refused:
            # The JSON exists but its per_case entries carry no label/dtype, so the
            # comparison cannot be made. Say so instead of reporting a clean result.
            no_labels.append(op)
            continue
        gap = claims[op] - exercised - refused
        if gap:
            rows.append((op, sorted(gap), sorted(refused), len(claims[op])))

    if args.json:
        json.dump(
            {
                "untested": [
                    {"op": op, "missing": [list(p) for p in gap],
                     "refused": [list(p) for p in ref], "declared": total}
                    for op, gap, ref, total in rows
                ],
                "no_coverage_json": no_json,
                "unlabelled_cases": no_labels,
            },
            sys.stdout, indent=2,
        )
        print()
        return 0

    print(f"ops with declared workloads: {len(claims)}")
    print(f"  fully exercised or explicitly refused : {len(claims) - len(rows) - len(no_json) - len(no_labels)}")
    print(f"  with untested declared (label, dtype) : {len(rows)}")
    print(f"  no coverage JSON                      : {len(no_json)}")
    print(f"  JSON present but cases unlabelled     : {len(no_labels)}  <- cannot be checked")
    if rows:
        print()
        print("untested declared pairs (op claims it, nothing measured it):")
        for op, gap, refused, total in sorted(rows, key=lambda r: -len(r[1])):
            print(f"  {op}  ({len(gap)}/{total} declared pairs untested"
                  + (f", {len(refused)} explicitly refused" if refused else "") + ")")
            for label, dtype in gap[:6]:
                print(f"      {label}  {dtype}")
            if len(gap) > 6:
                print(f"      ... and {len(gap)-6} more")
    if no_labels:
        print()
        print("unlabelled (comparison impossible -- these are NOT known-good):")
        for op in no_labels:
            print(f"      {op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
