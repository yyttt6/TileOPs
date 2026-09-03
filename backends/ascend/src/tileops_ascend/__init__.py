"""TileOPs backend for Huawei Ascend NPU. PM-owned: frozen after Wave 1.

Importing this module performs the whole registration. TileOPs enumerates the
``tileops.backends`` entry-point group while constructing its first op, imports this
module, and the calls at the bottom fill its registry.

**Device claiming is off by default, and that is deliberate.** TileOPs has a
no-fall-back rule: once a detector claims a device type, *every* op on those devices
must have a registered builder, and a missing one is an error rather than a fall back
to the kernels TileOPs ships. Registering a detector before all 178 ops are covered
would therefore make the whole backend unusable rather than partially usable.

So during development the caller names the target::

    GemmFwdOp(target="ascend")

which works without a detector: TileOPs' ``known_targets()`` is the union of the
detector table and the builder table, so registering one builder is enough to make
the name resolvable. Set ``TILEOPS_ASCEND_CLAIM_DEVICES=1`` to also register the
detector and have plain ``GemmFwdOp()`` pick this backend from the input device --
only once coverage is complete.

See docs/DECISIONS.md D001, D002, D007.
"""

from __future__ import annotations

import os

import torch

from tileops.backend import register_detector

from . import families  # noqa: F401  -- importing it performs every registration
from ._registry import REGISTERED, TARGET, register

__all__ = ["REGISTERED", "TARGET", "claims_devices", "register"]

#: Env var that turns device detection on. Not a config file and not an argument:
#: what decides which kernel runs should be visible in how the process was started.
CLAIM_ENV = "TILEOPS_ASCEND_CLAIM_DEVICES"


def detect(device: torch.device) -> bool:
    """Whether *device* is the kind of device these kernels are written for.

    Devices only -- not dtypes, not shapes, and never an exception. Whether a
    particular call is supported is ``build_kernel``'s answer, since only it sees the
    dtypes and the parameters.

    ``torch_npu`` exposes Ascend cards as device type ``"npu"`` (verified:
    ``torch.randn(4, 4).npu().device.type == "npu"``), so the string is not read from
    a vendor runtime.
    """
    return device.type == "npu"


def claims_devices() -> bool:
    """Whether this import registered a detector. See the module docstring."""
    return os.environ.get(CLAIM_ENV, "") == "1"


if claims_devices():
    register_detector(target=TARGET, detect=detect)
