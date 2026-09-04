"""Every op reachable by importing family modules by hand must also be reachable
through a plain ``import tileops.kernels``.

Why this test exists (2026-08-26): ``families/__init__.py`` imported only
``elementwise``. A plain ``import tileops.kernels`` -- which is what the
``tileops.backends`` entry point does -- registered **24** ops, while importing the
13 family modules directly registered **111**. The harness imported family modules
directly, so all 68 passing coverage JSONs were genuine yet described ops that the
real plugin path could not reach. Under the no-fallback rule (D002) each unreachable
op surfaces as an ``OpNotAvailableError`` at launch, never as an import failure, so
nothing pointed at the cause.

The asymmetry is the whole point: coverage measures "does the kernel compute the
right answer", this measures "can the op layer find the kernel at all". Neither
implies the other.
"""

import glob
import importlib
import os

import tileops.kernels  # noqa: F401  -- the plain import under test
from tileops.kernels._registry import REGISTERED

_FAMILIES_DIR = os.path.join(os.path.dirname(tileops.kernels.__file__), "families")

#: Family modules an executor has created but that PM has not yet wired into
#: ``families/__init__.py``. PM adds the import -- and deletes the entry here -- when
#: the round that created the module is adjudicated. Wiring a half-written module in
#: early is worse than leaving it out: an explicit import means one broken family
#: module breaks the whole package, so an in-flight file would take every other op
#: down with it.
#:
#: This list is what keeps the guard honest. Its purpose is to catch a family that was
#: *silently forgotten*; an in-flight one is not forgotten, it is pending. Adding an
#: entry is a deliberate act that shows up in a diff, so the failure mode this test
#: was written for -- 87 ops unreachable and nothing pointing at why -- cannot recur
#: by omission. Every entry needs a task number, and an entry that outlives its task
#: is itself the defect.
PENDING_PM_INTEGRATION = {"attention_indexing"}  # T129/R192, in flight
# NOTE: set(), not {} -- {} is an empty dict, and
# ``set_of_modules - {}`` raises TypeError rather than doing what it looks like.
#
# Add an in-flight family module here as a string with its task number, e.g.
#     PENDING_PM_INTEGRATION = {"attention_decode"}  # T073, in flight
# and remove it when that round is adjudicated and the import is added to
# families/__init__.py.


def _family_module_names():
    return sorted(
        os.path.basename(p)[:-3]
        for p in glob.glob(os.path.join(_FAMILIES_DIR, "*.py"))
        if not p.endswith("__init__.py")
    )


def test_plain_import_reaches_every_family_module():
    """A family module on disk but absent from ``families/__init__.py`` is a defect."""
    declared = set(tileops.kernels.families.__all__)
    on_disk = set(_family_module_names())
    missing = on_disk - declared - PENDING_PM_INTEGRATION
    assert not missing, (
        "family modules exist on disk but are not imported by families/__init__.py, "
        f"so their ops are unreachable through the tileops.backends entry point: "
        f"{sorted(missing)}"
    )


def test_plain_import_registers_every_op():
    """Importing family modules by hand must not reveal ops the plain import missed."""
    reachable = set(REGISTERED)
    for name in _family_module_names():
        importlib.import_module(f"tileops.kernels.families.{name}")
    after_manual_import = set(REGISTERED)
    pending_ops = set()
    for name in PENDING_PM_INTEGRATION:
        mod = importlib.import_module(f"tileops.kernels.families.{name}")
        pending_ops |= {
            op for op, builder in REGISTERED.items()
            if getattr(builder, "__module__", "") == mod.__name__
        }
    hidden = after_manual_import - reachable - pending_ops
    assert not hidden, (
        f"{len(hidden)} ops register only when a family module is imported directly, "
        f"and are therefore unreachable through a plain `import tileops.kernels`. "
        f"If a family module is in flight, add it to PENDING_PM_INTEGRATION with its "
        f"task number; otherwise wire it into families/__init__.py: "
        f"{sorted(hidden)}"
    )
