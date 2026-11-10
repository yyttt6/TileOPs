"""CPU controls for pinned reference startup and SDK restoration."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nightly_support"))
from reference_runtime import initialize_reference_runtime  # noqa: E402


def test_non_fla_op_does_not_require_reference_environment(monkeypatch):
    monkeypatch.delenv("ASCEND950_REFERENCE_ROOT", raising=False)
    initialize_reference_runtime("GEMMFwdOp")


def test_fla_requires_pin(monkeypatch):
    monkeypatch.delenv("ASCEND950_REFERENCE_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="missing"):
        initialize_reference_runtime("DeltaNetFwdOp")


@pytest.mark.parametrize("cached_root", ["/runtime", "/wrong"])
def test_runtime_selection_restores_tileops_sdk(monkeypatch, cached_root):
    monkeypatch.setenv("ASCEND950_REFERENCE_ROOT", "/reference")
    monkeypatch.setenv("ASCEND_RUNTIME_ROOT", "/runtime")
    monkeypatch.setenv("ASCEND_HOME_PATH", "/compiler-sdk")
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=f"/reference/{name}/__init__.py"),
    )
    monkeypatch.setitem(
        sys.modules,
        "triton.backends.ascend.utils",
        SimpleNamespace(_get_ascend_path=lambda: Path(cached_root)),
    )
    if cached_root == "/wrong":
        with pytest.raises(RuntimeError, match="different"):
            initialize_reference_runtime("GLAFwdOp")
    else:
        initialize_reference_runtime("GLAFwdOp")
    import os

    assert os.environ["ASCEND_HOME_PATH"] == "/compiler-sdk"


def test_wrong_import_origin_rejected(monkeypatch):
    monkeypatch.setenv("ASCEND950_REFERENCE_ROOT", "/reference")
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: SimpleNamespace(origin="/wrong/__init__.py")
    )
    with pytest.raises(RuntimeError, match="resolve"):
        initialize_reference_runtime("GLAFwdOp")
