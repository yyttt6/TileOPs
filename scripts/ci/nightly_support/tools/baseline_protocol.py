"""CPU-only protocol, provenance, and supervisor progress for the 2026-10-18 rerun."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

PROTOCOL_ID = "ascend950-isolated-ffts-v3"
PROTOCOL = {
    "id": PROTOCOL_ID,
    "warmup": 3,
    "repeats": 15,
    "cache_flush_mib": 1024,
    "reset": "before flush and synchronize, outside profiling",
    "window": "one NPU profiler session per repeat",
    "metric": "mean of summed distinct kernel task spans; AIC/AIV roles counted once; excludes host launch gaps",
    "samples": "all 15 samples retained; no outlier removal",
}
TERMINAL_FAILURES = {
    "failed",
    "failed_verification",
    "failed_correctness",
    "unverified",
    "noncomparable",
    "timeout",
    "device_fault",
    "no_reference",
    "no_implementation",
    "skipped_policy",
    "interrupted",
}


def provenance(row):
    row["measurement_protocol"] = PROTOCOL_ID
    row["run_id"] = os.environ.get("BASELINE_RUN_ID", "standalone")
    row["toolchain_sha256"] = os.environ.get("BASELINE_TOOLCHAIN_SHA256")
    row["recorded_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return row


def begin(row, output, stage="measure"):
    path = os.environ.get("BASELINE_PROGRESS_FILE")
    if not path:
        return
    p = Path(path)
    data = {
        "pid": os.getpid(),
        "started_at": time.time(),
        "stage": stage,
        "output": str(output),
        "row": provenance(dict(row)),
    }
    temp = p.with_suffix(".tmp")
    temp.write_text(json.dumps(data, default=str), encoding="utf-8")
    temp.replace(p)


def seed_case(torch, op, case_id):
    seed = int.from_bytes(hashlib.sha256(f"{op}::{case_id}".encode()).digest()[:4], "big")
    torch.manual_seed(seed)
    return seed


def current_row(row):
    return (
        row.get("measurement_protocol") == PROTOCOL_ID
        and row.get("warmup") == 3
        and row.get("repeats") == 15
        and (
            not os.environ.get("BASELINE_RUN_ID")
            or row.get("run_id") == os.environ["BASELINE_RUN_ID"]
        )
    )


def eligible(row):
    if row.get("warmup") != 3 or row.get("repeats") != 15:
        return False
    if row.get("measurement_protocol") != PROTOCOL_ID:
        return False
    if row.get("status") not in {
        "measured_reference",
        "measured_candidate",
        "measured_tileops_ascend",
        "measured_candidate_input_mutated",
        "measured_tileops_ascend_input_mutated",
    }:
        return False
    latency = row.get("latency_ms")
    samples = row.get("latency_samples_ms", [])
    if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
        return False
    if len(samples) != 15 or not all(
        isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in samples
    ):
        return False
    if row.get("candidate") != "manifest_reference":
        return bool(row.get("checked_outputs", 0)) and not row.get("noncomparable_reason")
    return not row.get("input_mutated")
