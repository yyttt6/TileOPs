from types import SimpleNamespace

import pytest

from benchmarks.ascend.support import unsupported_reason
from tileops.kernels.ascend.fft.c2c import AscendFFTC2CKernel


def case(name, dtype="complex128", n=4096):
    return SimpleNamespace(id=name, _call=SimpleNamespace(tensors={"input": ([64, n], dtype)}))


@pytest.mark.parametrize(
    "name,n", [("throughput-4k-complex128", 4096), ("scientific-1m-complex128", 1048576)]
)
def test_declared_fft_cases_require_real_kernel_refusal(name, n):
    reason = unsupported_reason(
        "FFTC2CFwdOp", case(name, n=n), {"FFTC2CFwdOp": {"ascend": AscendFFTC2CKernel}}
    )
    assert "FP64/complex128 is unsupported" in reason


def test_other_cases_are_not_silently_excluded():
    assert unsupported_reason("FFTC2CFwdOp", case("new-complex128"), {}) is None
    assert unsupported_reason("OtherOp", case("throughput-4k-complex128"), {}) is None
    assert unsupported_reason("FFTC2CFwdOp", case("latency-4k-complex64", "complex64"), {}) is None


def test_manifest_drift_is_an_error():
    with pytest.raises(RuntimeError, match="manifest"):
        unsupported_reason("FFTC2CFwdOp", case("throughput-4k-complex128", "complex64"), {})


def test_new_support_cannot_remain_hidden(monkeypatch):
    monkeypatch.setattr(AscendFFTC2CKernel, "refusal", lambda call: None)
    with pytest.raises(RuntimeError, match="support changed"):
        unsupported_reason(
            "FFTC2CFwdOp",
            case("throughput-4k-complex128"),
            {"FFTC2CFwdOp": {"ascend": AscendFFTC2CKernel}},
        )
