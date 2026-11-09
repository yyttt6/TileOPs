#!/usr/bin/env python3
"""Keep native source in csrc, with explicit Ascend prelude compatibility exceptions.

Ascend's T.Kernel embeds its prelude into the compiled IR. Its native SIMD helpers
cannot use CUDA's pre-include flags. Existing measured helpers may retain their
prelude with a reason on each use; this never exempts asm or CUDA source strings.
"""

import re
import sys
from pathlib import Path

from common import files, report

_NATIVE = re.compile(
    r"\b(import_source\s*\(|(__)?asm(__)?(\s+(__)?volatile(__)?)?\s*\()|__(device|global)__"
)
_PRELUDE = re.compile(r"\bprelude\s*=")
_ALLOW = re.compile(r"# lint: ascend-prelude allow: \S")


def findings(path: Path) -> list[str]:
    """Return forbidden native-source uses, without exempting other code on a line."""
    # These fixtures intentionally contain rejected source fragments.
    if path.as_posix().endswith("tests/lint/test_native_source_lint.py"):
        return []
    ascend = "src/tileops/_kernels/ascend/" in path.as_posix()
    out = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if _NATIVE.search(line) or (_PRELUDE.search(line) and not (ascend and _ALLOW.search(line))):
            out.append(f"{path}:{number}: native source belongs in src/tileops/csrc")
    return out


if __name__ == "__main__":
    sys.exit(
        report(
            e
            for p in files(
                sys.argv[1:], ("src/tileops", "workloads", "benchmarks", "tests"), (".py",)
            )
            for e in findings(p)
        )
    )
