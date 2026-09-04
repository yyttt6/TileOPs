#!/usr/bin/env python3
"""Tiered coverage report. One number cannot answer "how far along are we?".

This project has now been bitten three times by a single-number coverage metric that
silently included ops that were not actually usable:

* F015 -- 68 ops "passing" while only 24 were reachable through the ``tileops.backends``
  entry point. Coverage measured "the kernel computes the right answer"; it could not
  see "the op layer can find the kernel".
* §13.11 -- coverage JSONs going stale against the code they measured, in both
  directions (``AbsFwdOp`` counted as failing for a day after it was fixed).
* §13.16 / §13.28 -- ``REGISTERED`` counting fail-closed stubs, and stale stubs
  blocking kernels that worked.

The third form, which this script exists for: an op can have
``correctness.passed=true`` while ``status=blocked``. Those two fields answer different
questions and the headline was quietly counting the optimistic one. Concretely, on
2026-08-26 twelve ops were in that state, and three of them (``EngramGateConvFwdOp``,
``EngramGateConvBwdOp``, ``MHCPreFwdOp``) could not be invoked through the TileOPs op
layer at all -- their kernels are correct, but a CUDA-only check intercepts the call
before the backend is reached. Counting them as covered overstated progress by 3.

So the report is tiered, and the tiers are not interchangeable:

  A  usable      correctness passed AND not blocked -- callable and gated
  B  ungated     correctness passed BUT blocked -- kernel is right, something else isn't
  C  remaining   no passing correctness result

Tier B is split by *why*, because the reasons have opposite implications: an op blocked
because bool has no hardware-peak entry is usable today and merely lacks a performance
gate; an op blocked because the op layer refuses the call is not usable at all.

Usage:
    python coverage_report.py [--json]
"""
import argparse
import glob
import json
import os
import re
import sys

# D018: archetype 9 is deferred to phase two, so phase one's denominator excludes it.
DEFERRED_FAMILIES = {"linear_attention.yaml", "mamba.yaml"}
DEFERRED_OPS = {"EngramDecodeFwdOp"}

# 910B1 has no native FP8; these go via CATLASS soft-FP8 in phase two. FFT is
# archetype 11, also phase two.
PHASE_TWO_ONLY = {
    "FP8LightningIndexerFwdOp",
    "BmmFp8KNFwdOp",
    "BmmFp8NKFwdOp",
    "GemmFp8FwdOp",
    "FP8QuantFwdOp",
    "FFTC2CFwdOp",
}

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPS_TABLE = os.path.join(ROOT, "docs", "tasks", "OPS-178.md")
COVERAGE = os.path.join(ROOT, "tileops-ascend-harness", "coverage")


def authoritative_ops():
    """op -> family, from the authoritative 178-op inventory."""
    ops = {}
    with open(OPS_TABLE) as handle:
        for line in handle:
            match = re.match(r"^\| *\d+ *\| *`([A-Za-z0-9_]+)` *\| *([a-z_.]+)", line)
            if match:
                ops[match.group(1)] = match.group(2)
    return ops


def load_coverage():
    """op -> parsed coverage JSON, skipping self-test fixtures."""
    results = {}
    for path in glob.glob(os.path.join(COVERAGE, "*.json")):
        if "_selftest" in path:
            continue
        try:
            data = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        results[data.get("op") or os.path.basename(path)[:-5]] = data
    return results


def block_reason(data):
    """Why a correctness-passing op is still blocked, as a coarse category.

    Distinguishes "usable but ungated" from "not usable", which is the whole point of
    splitting tier B.
    """
    text = json.dumps(data).lower()
    if "unavailable" in text and "sol" in text:
        return "sol-profile-gap"        # e.g. bool has no hardware peak entry: usable
    if "is_cuda" in text or "_manifest_params" in text or "op layer" in text:
        return "op-layer-refuses-call"  # not usable end to end
    if "baseline" in text and "violation" in text:
        return "baseline-impossible"    # usable; the reference is untrustworthy (D020)
    return "unclassified"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    families = authoritative_ops()
    coverage = load_coverage()

    deferred = {
        op for op, family in families.items() if family in DEFERRED_FAMILIES
    } | DEFERRED_OPS
    phase_one = set(families) - deferred

    usable, ungated, remaining = [], [], []
    for op in sorted(phase_one):
        data = coverage.get(op)
        passed = bool(((data or {}).get("correctness") or {}).get("passed"))
        if not passed:
            remaining.append(op)
        elif data.get("status") == "blocked":
            ungated.append((op, block_reason(data)))
        else:
            usable.append(op)

    addressable = [op for op in remaining if op not in PHASE_TWO_ONLY]
    total = len(phase_one)

    if args.json:
        json.dump(
            {
                "phase_one_total": total,
                "usable": usable,
                "ungated": [{"op": op, "reason": why} for op, why in ungated],
                "remaining": remaining,
                "remaining_addressable": addressable,
            },
            sys.stdout,
            indent=2,
        )
        print()
        return 0

    print(f"phase one denominator: {total}  (178 minus {len(deferred)} deferred by D018)")
    print()
    print(f"A  usable     {len(usable):>3}/{total}  = {len(usable)/total*100:4.1f}%   "
          f"correctness passed AND not blocked -- callable and gated")
    print(f"B  ungated    {len(ungated):>3}/{total}  = {len(ungated)/total*100:4.1f}%   "
          f"correctness passed BUT blocked -- NOT interchangeable with A")
    print(f"C  remaining  {len(remaining):>3}/{total}  = {len(remaining)/total*100:4.1f}%   "
          f"({len(addressable)} addressable, {len(remaining)-len(addressable)} phase-two-only)")
    print()
    print("tier B by reason -- these do not mean the same thing:")
    by_reason = {}
    for op, why in ungated:
        by_reason.setdefault(why, []).append(op)
    for why in sorted(by_reason):
        note = {
            "op-layer-refuses-call": "NOT usable: the call never reaches the backend",
            "sol-profile-gap": "usable today; only the performance gate is missing",
            "baseline-impossible": "usable; the reference is untrustworthy (D020)",
        }.get(why, "needs classifying")
        print(f"  {why:24} {len(by_reason[why]):>2}  -- {note}")
        for op in sorted(by_reason[why]):
            print(f"      {op}")
    if addressable:
        print()
        print(f"tier C addressable ({len(addressable)}):")
        for op in addressable:
            print(f"      {op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
