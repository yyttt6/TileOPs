"""Catch stale manifest names before an otherwise green CPU preflight."""
import ast
from pathlib import Path

from tileops.manifest import load_manifest


def test_literal_ascend_registrations_name_current_manifest_entries():
    root = Path(__file__).resolve().parents[3]
    manifest = load_manifest()
    stale = []
    for path in (root / 'src/tileops/_kernels/ascend').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = getattr(node.func, 'id', getattr(node.func, 'attr', None))
            if name != 'register_kernel_type':
                continue
            op = node.args[0]
            if isinstance(op, ast.Constant) and isinstance(op.value, str) and op.value not in manifest:
                stale.append((str(path.relative_to(root)), op.value))
    assert not stale, f'Ascend registrations outside the current manifest: {stale}'
