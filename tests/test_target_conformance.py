"""Every implemented op, served by a target for another kind of device.

What a backend is promised, checked on the whole manifest rather than on a sample: its
builder is described with the op's ``forward`` inputs, the kernel it returns is called
with exactly those tensors, and nothing on the way asks a CUDA device anything. A new op
joins by having a manifest entry; its call is the manifest call with the smallest inputs.
"""

import math

import pytest
import torch

from tileops.backend import TensorSpec
from tileops.backend import _registry as registry
from tileops.manifest import load_adts, load_manifest
from tileops.manifest._plan import entry_plan
from tileops.manifest._registry import op_class
from tileops.manifest._workload import instantiate
from tileops.manifest.typed._workload import Call as TypedCall
from tileops.manifest.typed._workload import Elements

pytestmark = pytest.mark.smoke


def _from_call(cls: type, name: str, entry: dict) -> tuple:
    """The op, its ``forward`` arguments and the outputs its call declares, from the manifest
    call with the smallest inputs."""
    plan = entry_plan(name, entry, load_adts(), resolve=False)
    calls = [
        instantiate(plan, row, case, metadata=False)
        for row in entry["workloads"]
        for case in row.get("dtype_cases") or [{}]
    ]

    def elements(tensor):
        pairs = tensor.items if isinstance(tensor, Elements) else (tensor,)
        return sum(math.prod(shape) for shape, _ in pairs)

    call = min(calls, key=lambda c: sum(elements(t) for t in c.tensors.values()))

    def empty(spec):
        return torch.empty(spec.shape, dtype=getattr(torch, spec.dtype))

    typed = isinstance(call, TypedCall)
    tensors = {
        n: (call.scalars.get(n) if typed else None)
        if spec is None
        else [empty(e) for e in spec]
        if isinstance(spec, tuple)
        else empty(spec)
        for n, spec in call.specs.items()
    }
    args = [tensors[n] for n in plan.sig.inputs]
    while args and args[-1] is None:
        if typed and list(plan.sig.inputs.values())[len(args) - 1].default is not None:
            break
        args.pop()
    written = (
        call.written if typed else {n for n, t in plan.sig.inputs.items() if t.mutated is True}
    )
    outputs = [
        tensors[t.alias] if t.alias in written else tensors[o] for o, t in plan.sig.outputs.items()
    ]
    return cls(**call.arguments(tensors)), tuple(args), outputs


def _implemented() -> list[str]:
    return sorted(n for n, e in load_manifest().items() if e.get("status") == "implemented")


@pytest.fixture(autouse=True)
def isolated_registry():
    state = registry.snapshot()
    registry.DETECTORS.clear()
    registry.BUILDERS.clear()
    registry.LOAD_FAILURES.clear()
    registry.default_target = None
    registry._loaded = True
    registry.register_detector("elsewhere", lambda device: device.type == "cpu")
    registry.default_target = "elsewhere"
    yield
    registry.restore(state)


@pytest.fixture(autouse=True)
def no_cuda():
    """A host with no CUDA: any device query fails the call that makes it."""
    from tileops._utils import forget_device_properties

    def refuse(*args, **kwargs):
        raise AssertionError("queried a CUDA device")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "is_available", lambda: False)
        patch.setattr(torch.accelerator, "current_accelerator", lambda **_: None)
        for name in (
            "current_device",
            "get_device_capability",
            "get_device_properties",
            "get_device_name",
        ):
            patch.setattr(torch.cuda, name, refuse)
        for name in ("current_device_index", "device_index"):
            patch.setattr(torch.accelerator, name, refuse)
        forget_device_properties()
        yield
    # After the patches are undone, so no answer read under them stays cached.
    forget_device_properties()


@pytest.mark.parametrize("name", _implemented())
def test_a_target_is_described_and_called_with_the_forward_inputs(name):
    entry = load_manifest()[name]
    cls = op_class(name, entry)
    op, args, declared_outputs = _from_call(cls, name, entry)
    declared = tuple(entry["signature"].get("inputs") or {})
    passed = args + (None,) * (len(declared) - len(args))
    described = tuple(
        None
        if t is None
        else tuple(map(TensorSpec.of, t))
        if isinstance(t, (tuple, list))
        else TensorSpec.of(t)
        if isinstance(t, torch.Tensor)
        else type(t)
        for t in passed
    )
    expected_params = op._manifest_params()
    seen, returned = [], []

    def build_kernel(*specs, **params):
        assert specs == described
        assert params == expected_params

        def kernel(*tensors, **writes):
            seen.append(tensors)
            result = declared_outputs
            returned.append(
                None if not result else result[0] if len(result) == 1 else tuple(result)
            )
            return returned[-1]

        return kernel

    registry.register_kernel_builder(name, "elsewhere", build_kernel)

    positional, keywords = cls._signature.sig.forward_arguments(args)
    result = op(*positional, **keywords)

    assert len(seen) == 1, "the kernel the target built runs the call"

    # A tensor sequence is the same tensors in order, held in whatever list the call built.
    def same(a, b):
        if isinstance(b, (list, tuple)):
            return type(a) is type(b) and len(a) == len(b) and all(map(same, a, b))
        return a is b

    assert all(same(a, b) for a, b in zip(seen[0], passed, strict=True)), (
        "called with what it was described"
    )
    outputs = tuple(entry["signature"]["outputs"])
    if all(o in declared for o in outputs):
        assert result is None, "an op that writes its caller's buffer returns nothing"
    elif isinstance(returned[0], tuple):
        assert all(a is b for a, b in zip(result, returned[0], strict=True))
    else:
        assert result is returned[0], "the op returns what the target's kernel returned"
    assert hasattr(cls, "_signature"), "a target is held to its signature"
    inputs = entry["signature"].get("inputs") or {}
    assert all((inputs[o] or {}).get("mutated") for o in outputs if o in inputs), (
        "an output passed in as an input is one the call writes"
    )
