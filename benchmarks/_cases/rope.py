"""Case factories of the rope family."""

from benchmarks._cases import Entry, default_binder
from benchmarks.api import Implementation
from benchmarks.baselines import private_inputs
from workloads.rope import RoPECall, RoPECosSinCall

ENTRIES = {
    name: Entry(RoPECall)
    for name in (
        "RoPEFwdOp",
        "RoPELlama31FwdOp",
        "YaRNFwdOp",
        "LongRoPEFwdOp",
        "RoPENeoxPositionIdsFwdOp",
    )
}


def _rope_cos_sin(op, case) -> Implementation:
    """The op on a private ``x`` where it rotates ``x`` in place."""
    if not case.arguments["inplace"]:
        return default_binder(op, case)
    return private_inputs(op, case.inputs, 0)


ENTRIES.update(
    {
        "RoPECosSinFwdOp": Entry(RoPECosSinCall, binder=_rope_cos_sin),
    }
)
