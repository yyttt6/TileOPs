"""Public elementwise regressions; invoke the entire command under npu-guard."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'TileOPs/src'), str(ROOT / 'TileOPs'), str(ROOT)]
from benchmarks.ascend.guard import require_reservation
require_reservation()

import torch
import torch_npu
import pytest
from tileops.backend import registry

torch.npu.set_device(0)
p = torch.npu.get_device_properties(0)
assert torch.npu.get_device_name(0) == 'Ascend950DT_9582'
assert (p.cube_core_num, p.vector_core_num) == (32, 64)
torch.npu.set_device_limit(0, cube_num=32, vector_num=64)
registry.ensure_loaded()
assert not registry.LOAD_FAILURES, registry.LOAD_FAILURES
plan = json.loads((ROOT / 'ELEMENTWISE_PLAN.json').read_text())
assert all('ascend' in registry.REGISTERED_KERNEL_TYPES.get(p['op'], {}) for p in plan)
files = [
    'test_unary_math.py', 'test_activation.py', 'test_special_elementwise.py',
    'test_binary_arith.py', 'test_elementwise_binary_broadcast.py',
    'test_bitwise.py', 'test_logical.py', 'test_comparison.py',
    'test_comparison_boundaries.py', 'test_fused_gated.py',
    'test_where_clamp_masked_fill_conformance.py',
    'test_elementwise_unary_activation_conformance.py',
]
try:
    rc = pytest.main([*[str(ROOT / 'TileOPs/tests/ops' / f) for f in files],
                      '-q', '--tileops-device=npu', '--tileops-target=builtin',
                      *sys.argv[1:]])
finally:
    torch.npu.synchronize()
raise SystemExit(rc)
