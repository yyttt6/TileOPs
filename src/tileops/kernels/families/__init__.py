"""One module per op family; importing this package performs every registration.

PM-owned: frozen after Wave 1. A new family is added here by one line, and the
family module owns its own ops exclusively -- which is what lets several executors
work in parallel without touching a shared file (docs/DECISIONS.md D007).

Every family module is imported **explicitly**, and deliberately so: under the
no-fallback rule (D002) a family that fails to import would not fail loudly -- its
ops would simply be absent, and each one would surface much later as an
``OpNotAvailableError`` at launch. An ImportError here is far cheaper to diagnose.
A glob-based auto-import would trade that startup error for a runtime mystery.

Regression guard: ``tests/npu/test_registration_reachable.py`` asserts
that a plain ``import tileops.kernels`` registers every op that importing the family
modules directly would register. That test exists because on 2026-08-26 this file
imported only ``elementwise``: 24 ops were reachable through the ``tileops.backends``
entry point while 111 were reachable by importing family modules by hand -- the
harness did the latter, so every coverage JSON passed and the gap stayed invisible.
"""

from __future__ import annotations

from . import attention  # noqa: F401
from . import attention_bwd  # noqa: F401
from . import attention_decode  # noqa: F401
from . import convolution  # noqa: F401
from . import elementwise  # noqa: F401
from . import elementwise_activation  # noqa: F401
from . import elementwise_unary_math  # noqa: F401
from . import gemm  # noqa: F401
from . import indexed_reduce  # noqa: F401
from . import linear_attention  # noqa: F401
from . import moe  # noqa: F401
from . import normalization_spatial  # noqa: F401
from . import pool  # noqa: F401
from . import position_encoding  # noqa: F401
from . import reduction  # noqa: F401
from . import reduction_count_nonzero  # noqa: F401
from . import scan  # noqa: F401
from . import sequence_modeling  # noqa: F401
from . import two_pass  # noqa: F401

__all__ = [
    "attention",
    "attention_bwd",
    "attention_decode",
    "convolution",
    "elementwise",
    "elementwise_activation",
    "elementwise_unary_math",
    "gemm",
    "indexed_reduce",
    "linear_attention",
    "moe",
    "normalization_spatial",
    "pool",
    "position_encoding",
    "reduction",
    "reduction_count_nonzero",
    "scan",
    "sequence_modeling",
    "two_pass",
]
