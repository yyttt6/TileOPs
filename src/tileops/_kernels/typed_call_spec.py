"""Device tuning facts used by post-v0.1 kernel interfaces."""

import dataclasses

import torch

from tileops._kernels.call_spec import CallSpec, _at_default, _DeviceFact
from tileops._utils.device import on_current_index

_DEVICE_FACTS = (
    "arch",
    "calibration",
    "sm_count",
    "smem_budget",
    "threads_per_sm",
    "regs_per_sm",
    "smem_per_sm",
    "l2_bytes",
)


@dataclasses.dataclass(frozen=True)
class TypedCallSpec(CallSpec):
    """A call with the CUDA-specific tuning facts published by the newer interfaces."""

    threads_per_sm: int = dataclasses.field(
        default=_DeviceFact("threads_per_sm"), compare=False, kw_only=True
    )

    regs_per_sm: int = dataclasses.field(
        default=_DeviceFact("regs_per_sm"), compare=False, kw_only=True
    )

    smem_per_sm: int = dataclasses.field(
        default=_DeviceFact("smem_per_sm"), compare=False, kw_only=True
    )

    l2_bytes: int = dataclasses.field(default=_DeviceFact("l2_bytes"), compare=False, kw_only=True)

    def __post_init__(self) -> None:
        stated = frozenset(f for f in _DEVICE_FACTS if f"_{f}" in vars(self))
        object.__setattr__(self, "stated_device_facts", stated)
        device = self.device
        if device is None:
            return
        if not isinstance(device, torch.device):
            device = torch.device(device)
        object.__setattr__(self, "device", on_current_index(device))

    @property
    def device_index(self) -> "int | None":
        """The index of ``device``, or ``None`` for a record that names no device."""
        return self.device.index if self.device is not None else None

    def _read_device_facts(self) -> None:
        """Fill the device facts the caller left unstated from ``device``.

        Read through ``tileops.backend._dispatch.device_facts_and_tuning``: TileOPs
        reads CUDA's, a backend registers another device type's, and a device of a type
        nobody registered has none. Only a CUDA device has tuning facts. A record that
        named no device reads the current one of the process's accelerator, and has none
        where there is no accelerator.
        """
        from tileops.backend._dispatch import device_facts_and_tuning

        read, tuning = device_facts_and_tuning(self.device)
        facts = self.__dict__
        facts.setdefault("_arch", read.arch)
        facts.setdefault("_sm_count", read.sm_count)
        facts.setdefault("_smem_budget", read.smem_budget)
        for name, value in tuning._asdict().items():
            facts.setdefault(f"_{name}", value)

    def _on_device(self, device: torch.device) -> "CallSpec":
        """The same call on *device*; the device facts it did not state come from *device*."""
        fields = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.init and f.name not in (*_DEVICE_FACTS, "device")
        }
        stated = {name: getattr(self, name) for name in self.stated_device_facts}
        return type(self)(**fields, **stated, device=device)

    def __str__(self) -> str:
        """The facts of the call, without the fields nobody set.

        A selection failure names the call, and a record of mostly default
        fields buries the device facts that decided it.
        """
        facts = {name: getattr(self, name) for name in _DEVICE_FACTS}
        stated = [
            f"{f.name}={getattr(self, f.name)!r}"
            for f in dataclasses.fields(self)
            if f.name not in (*_DEVICE_FACTS, "device") and not _at_default(self, f)
        ]
        return ", ".join([*(f"{name}={value}" for name, value in facts.items()), *stated])
