"""Generate T105 mixed-kernel artifacts for the static guard gates."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import torch

sys.modules.setdefault(
    "tileops_ascend.families.two_pass", types.ModuleType("tileops_ascend.families.two_pass")
)
from tileops_ascend.kernels import elementwise_mixed as mixed  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/reports/R105-data/artifacts"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind in ("and", "or", "xor"):
        for a_shape, b_shape, tag in (
            ((4097,), (4097,), "4097"),
            ((2, 3, 683), (1, 3, 1), "2x3x683-broadcast"),
        ):
            _, _, out_shape, a_strides, b_strides = mixed._shape_info(a_shape, b_shape)
            a_numel = int(torch.tensor(a_shape).prod())
            b_numel = int(torch.tensor(b_shape).prod())
            out_numel = int(torch.tensor(out_shape).prod())
            kernel = mixed._compile_bitwise(
                a_numel, b_numel, out_numel, out_shape,
                a_strides, b_strides, "int32", kind,
            )
            path = OUT / f"bitwise-{kind}-{tag}-int32.cu"
            path.write_text(kernel.get_kernel_source(), encoding="utf-8")
            paths.append(path)
    for shape, channels, inner, tag in (
        ((1, 4, 1025), 4, 1025, "1x4x1025"),
        ((2, 3, 700), 3, 700, "2x3x700"),
    ):
        n = int(torch.tensor(shape).prod())
        kernel = mixed._compile_prelu(
            n, channels, shape, "float16", "float16", "float16", channels, inner
        )
        path = OUT / f"prelu-{tag}-fp16.cu"
        path.write_text(kernel.get_kernel_source(), encoding="utf-8")
        paths.append(path)
    rows = [
        {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in paths
    ]
    (ROOT / "docs/reports/R105-data/artifact-sha256.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
