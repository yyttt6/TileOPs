"""Check that the complete process belongs to the host NPU reservation."""

import fcntl
import json
import os
from pathlib import Path

LOCK_PATH = Path("/run/lock/ascend-npu/device-0.lock")


def require_reservation() -> dict:
    """Reject bare execution, stale metadata, or another process's reservation."""
    with LOCK_PATH.open("r") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(handle, fcntl.LOCK_UN)
            raise RuntimeError(
                "Run the whole command through scripts/benchmark_ascend.py or npu-guard"
            )
        meta = json.load(handle)
    ancestors = set()
    pid = os.getpid()
    while pid > 1:
        ancestors.add(pid)
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        pid = int(fields[1])
    guard_pid = meta.get("pid")
    if guard_pid not in ancestors or meta.get("phase") != "running":
        raise RuntimeError("The active NPU reservation does not own this process")
    fields = Path(f"/proc/{guard_pid}/stat").read_text().rsplit(")", 1)[1].split()
    if int(fields[19]) != meta.get("start_ticks"):
        raise RuntimeError("NPU reservation PID identity changed")
    return meta
