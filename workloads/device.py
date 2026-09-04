"""Process-wide device selection shared by workloads, tests, and benchmarks.

One name, read once at import: a workload that hard-codes a device string cannot be
run anywhere else, and a per-call ``device=`` argument would let a test and the
reference program it checks against disagree.

``npu`` is the default because that is the hardware these kernels are written for.
``TILEOPS_DEVICE`` names another one -- for a CPU-only reference run, say -- and is an
environment variable rather than an argument because it describes the machine, not
the call.
"""

from __future__ import annotations

import os

#: Where workload inputs are allocated. Read at import: every workload in a process
#: places its tensors on the same device.
DEVICE = os.environ.get("TILEOPS_DEVICE") or "npu"

__all__ = ["DEVICE"]
