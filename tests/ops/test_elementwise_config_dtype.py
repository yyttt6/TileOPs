"""Regression tests for the elementwise op layer's constructor surface."""

import inspect

import pytest

# Strategy lives in the kernel config dict, not in op/kernel ctor kwargs


@pytest.mark.smoke
def test_elementwise_ops_do_not_expose_strategy_kwarg():
    """No elementwise Op constructor exposes a ``strategy`` parameter.

    Strategy selection is a target's concern -- ``build_kernel`` sees the dtypes and
    the parameters and picks its own schedule -- so the op layer must not re-expose it
    as ctor plumbing.
    """
    import tileops.ops.elementwise as ew
    from tileops.ops.op_base import Op

    checked = 0
    for name in dir(ew):
        obj = getattr(ew, name)
        if not (isinstance(obj, type) and issubclass(obj, Op)):
            continue
        params = inspect.signature(obj.__init__).parameters
        assert "strategy" not in params, f"{name}.__init__ still exposes a strategy kwarg"
        checked += 1
    assert checked > 0, "Test bug: no Op classes discovered"
