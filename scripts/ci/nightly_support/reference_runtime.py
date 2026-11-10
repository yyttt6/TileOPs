"""Initialize the pinned FLA runtime metadata under the existing NPU guard."""

import importlib.util
import os
from pathlib import Path


def initialize_reference_runtime(op):
    if op not in {"DeltaNetFwdOp", "GLAFwdOp"}:
        return
    if not os.environ.get("ASCEND950_REFERENCE_ROOT"):
        raise RuntimeError("Pinned FLA reference environment is missing; source the CI profile")
    root = Path(os.environ["ASCEND950_REFERENCE_ROOT"]).resolve()
    for name in ("triton", "fla"):
        spec = importlib.util.find_spec(name)
        if (
            spec is None
            or spec.origin is None
            or not Path(spec.origin).resolve().is_relative_to(root)
        ):
            raise RuntimeError(f"{name} does not resolve from the pinned reference overlay")
    runtime = os.environ["ASCEND_RUNTIME_ROOT"]
    previous = os.environ.get("ASCEND_HOME_PATH")
    try:
        # The pinned Triton adapter caches its runtime root. CANN 9.2 is only a
        # compiler SDK here; complete installation metadata belongs to runtime 9.1.
        os.environ["ASCEND_HOME_PATH"] = runtime
        from triton.backends.ascend.utils import _get_ascend_path

        if Path(_get_ascend_path()).resolve() != Path(runtime).resolve():
            raise RuntimeError("Triton was initialized against a different CANN runtime")
    finally:
        if previous is None:
            os.environ.pop("ASCEND_HOME_PATH", None)
        else:
            os.environ["ASCEND_HOME_PATH"] = previous
