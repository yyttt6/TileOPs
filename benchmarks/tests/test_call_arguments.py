"""Constructor metadata must not allocate unrelated workload inputs."""

from types import SimpleNamespace

from workloads.workload_base import CallWorkload


def test_metadata_only_arguments_do_not_materialize():
    def forbidden(device):
        raise AssertionError("metadata-only constructor generated tensors")

    call = SimpleNamespace(
        signature=SimpleNamespace(ctor_tensors=()),
        materialize=forbidden,
        arguments=lambda tensors: {"dim": (1, 2), "keepdim": False, "tensors": tensors},
    )
    assert CallWorkload(call, device="cpu").arguments() == {
        "dim": (1, 2),
        "keepdim": False,
        "tensors": {},
    }


def test_tensor_constructor_arguments_still_materialize():
    tensor = object()
    devices = []

    def materialize(device):
        devices.append(device)
        return {"weight": tensor}

    call = SimpleNamespace(
        signature=SimpleNamespace(ctor_tensors=("weight",)),
        materialize=materialize,
        arguments=lambda tensors: tensors,
    )
    assert CallWorkload(call, device="cpu").arguments() == {"weight": tensor}
    assert devices == ["cpu"]
