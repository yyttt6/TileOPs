"""What crosses the boundary between TileOPs and a backend. Types only."""

from __future__ import annotations

from typing import Callable, NamedTuple, Union

import torch


class TensorSpec(NamedTuple):
    """What one tensor is, without the tensor. Handed to ``build_kernel``."""

    device: torch.device
    dtype: torch.dtype
    shape: tuple[int, ...]

    @staticmethod
    def of(tensor: torch.Tensor) -> "TensorSpec":
        """Describe *tensor*."""
        return TensorSpec(tensor.device, tensor.dtype, tuple(tensor.shape))


#: One call's result. A purely mutating op returns ``None``: ``torch.library.custom_op``
#: cannot express a return value aliasing an input.
KernelResult = Union[torch.Tensor, tuple[torch.Tensor, ...], None]

#: What a ``build_kernel`` hands back: the thing the op calls with its tensors. Callable is
#: the whole contract -- no base class, no required methods -- so a backend is free to
#: return a bound method, a closure over a compiled artifact, or an instance of its own
#: kernel class. The op layer holds it, caches it per input signature, and calls it.
Kernel = Callable[..., KernelResult]

#: Called ``build_kernel(*inputs, **params)``: a `TensorSpec` per input in
#: ``signature.inputs`` order — ``None`` for an ``optional: true`` input the call did not
#: pass, so presence is read off the slot rather than off how many slots there are — then
#: ``signature.params`` by keyword. Both lists are per-op, which the type system cannot
#: express, hence ``...``.
BuildKernel = Callable[..., Callable[..., KernelResult]]

#: "Is this the kind of device my kernels are written for" — ``False``, not an exception,
#: for the rest. Per-call support is ``build_kernel``'s answer; it sees the dtypes too.
DetectFn = Callable[[torch.device], bool]


#: What ``target=`` and the process default accept: the name a backend gives its own set
#: of kernels, or ``None`` to decide from the input device. There is no third answer --
#: every kernel this project runs comes from a backend, so "run the in-tree one instead"
#: names nothing.
Target = Union[str, None]
