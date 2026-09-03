"""T105 contract probe: compare direct mixed kernels with external builder paths."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import torch

# The shared worktree currently has a duplicate registration between the concurrent
# two_pass and normalization_spatial tasks.  Keep this probe scoped to the families
# needed here without changing either PM-owned file.
sys.modules.setdefault(
    "tileops_ascend.families.two_pass", types.ModuleType("tileops_ascend.families.two_pass")
)

from tileops.backend import TensorSpec  # noqa: E402
from tileops.ops.elementwise import (  # noqa: E402
    BitwiseAndFwdOp,
    BitwiseNotFwdOp,
    BitwiseOrFwdOp,
    BitwiseXorFwdOp,
    PreluFwdOp,
)
from tileops_ascend.families.elementwise import (  # noqa: E402
    build_bitwise_kernel,
    build_prelu,
)
from tileops_ascend.kernels import elementwise_mixed as mixed  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


def _err(fn):
    try:
        value = fn()
        return {"ok": True, "type": type(value).__name__, "value": repr(value)}
    except Exception as exc:  # noqa: BLE001 - the exception is the evidence
        return {"ok": False, "type": type(exc).__name__, "error": str(exc)}


def main() -> None:
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES must be set")
    n = 4097
    a = torch.arange(n, device="npu", dtype=torch.int32)
    b = (a * 3 + 5).to(torch.int32)
    x = torch.linspace(-1, 1, 4100, device="npu", dtype=torch.float16).reshape(1, 4, 1025)
    w = torch.tensor([0.1, 0.2, 0.3, 0.4], device="npu", dtype=torch.float16)
    rows = []

    direct = mixed._compile_bitwise(n, n, n, (n,), (1,), (1,), "int32", "and")
    direct_out = direct(a, b)
    torch.npu.synchronize()
    direct_cpu = direct_out.cpu()
    rows.append({
        "path": "direct_kernel",
        "op": "BitwiseAndFwdOp",
        "result": {
            "changed_from_zero": bool((direct_cpu != 0).any()),
            "equal": bool(torch.equal(direct_cpu, torch.bitwise_and(a.cpu(), b.cpu()))),
        },
    })
    sentinel = torch.full((n,), -777777, device="npu", dtype=torch.int32)
    explicit = _err(lambda: direct(a, b, sentinel))
    torch.npu.synchronize()
    explicit["sentinel_changed"] = bool((sentinel.cpu() != -777777).any()) if explicit["ok"] else None
    rows.append({"path": "direct_kernel_explicit_C", "op": "BitwiseAndFwdOp", "result": explicit})

    for name, builder, specs in (
        ("PreluFwdOp", build_prelu, (TensorSpec.of(x), TensorSpec.of(w))),
        ("BitwiseAndFwdOp", lambda aa, bb: build_bitwise_kernel(aa.shape, bb.shape, aa.dtype, op_kind="and"),
         (TensorSpec.of(a), TensorSpec.of(b))),
    ):
        rows.append({"path": "family_builder", "op": name, "result": _err(lambda: builder(*specs))})

    for cls, inputs, name in (
        (PreluFwdOp, (x, w), "PreluFwdOp"),
        (BitwiseAndFwdOp, (a, b), "BitwiseAndFwdOp"),
        (BitwiseOrFwdOp, (a, b), "BitwiseOrFwdOp"),
        (BitwiseXorFwdOp, (a, b), "BitwiseXorFwdOp"),
        (BitwiseNotFwdOp, (a,), "BitwiseNotFwdOp"),
    ):
        op = cls(target="ascend")
        result = _err(lambda op=op, inputs=inputs: op(*inputs))
        if result["ok"]:
            torch.npu.synchronize()
        rows.append({"path": "tileops_op", "op": name, "result": result})

    out = ROOT / "docs/reports/R105-data/contract-probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
