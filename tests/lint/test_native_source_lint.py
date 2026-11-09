"""Ascend prelude exceptions must not bypass the other native-source rules."""

from pathlib import Path

import native_source_lint as lint
import pytest


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("directory", "source", "rejected"),
    [
        ("ascend", "prelude=source", True),
        ("ascend", "prelude=source # lint: ascend-prelude allow: native SIMD", False),
        ("other", "prelude=source # lint: ascend-prelude allow: native SIMD", True),
        ("ascend", "prelude=source # lint: ascend-prelude allow: ", True),
        ("ascend", "import_source(code) # lint: ascend-prelude allow: native SIMD", True),
        ("ascend", "__device__ # lint: ascend-prelude allow: native SIMD", True),
        ("ascend", "asm(code) # lint: ascend-prelude allow: native SIMD", True),
        ("other", "flags=csrc_include('kernel.h')", False),
    ],
)
def test_native_source_boundary(tmp_path: Path, directory: str, source: str, rejected: bool):
    path = tmp_path / "src" / "tileops" / "_kernels" / directory / "probe.py"
    path.parent.mkdir(parents=True)
    path.write_text(source + "\n")
    assert bool(lint.findings(path)) is rejected
