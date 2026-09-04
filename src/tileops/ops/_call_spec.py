"""The base of every family's call record.

A *call record* is what an op states about one call, assembled in ``forward``
from op state plus what only the call knows: element type, inferred dims, and
the flags the caller passed. It is the argument the op layer's own validation
reads, and the value an error message names when a request contradicts itself.

Nothing here reads the device. A record describes the *request*; which set of
kernels serves it is `tileops.backend`'s answer, settled once per op instance
from the target and the input device. Keeping the two apart is what lets one
record be constructed on a host with no accelerator attached at all.
"""

import dataclasses

__all__ = ["CallSpec"]


@dataclasses.dataclass(frozen=True)
class CallSpec:
    """Base for a family's call record. A family subclasses it and adds fields."""

    def __str__(self) -> str:
        """The facts of the call, without the fields nobody set.

        An error names the call, and a record of mostly default fields buries
        the two that decided it.
        """
        default = type(self)()
        return ", ".join(
            f"{f.name}={getattr(self, f.name)!r}"
            for f in dataclasses.fields(self)
            if getattr(self, f.name) != getattr(default, f.name)
        )
