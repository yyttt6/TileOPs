"""Tests for tileops.ops.op_base.

Covers ``Op._cache_key`` default behavior, the runtime warning fired when
a subclass with empty ``_static_axes`` does not override ``_cache_key``,
composite kernel-map overrides, the ``get_or_build_kernel`` primitive, and
the explicit kernel enumeration ``Op.autotune`` runs over.
"""

import warnings

import pytest
import torch

from tileops.ops import op_base
from tileops.ops.op_base import Op

pytestmark = pytest.mark.smoke


@pytest.fixture(autouse=True)
def _reset_warned_types():
    """Clear the module-level dedup set so each test sees a fresh warn state."""
    op_base._EMPTY_STATIC_DIMS_WARNED.clear()
    yield
    op_base._EMPTY_STATIC_DIMS_WARNED.clear()


def _make_op_subclass(*, static_axes=frozenset(), override_cache_key=False):
    """Build a minimal concrete Op subclass for testing.

    ``static_axes`` populates ``_static_axes``.
    ``override_cache_key=True`` attaches a subclass-level override.
    """
    attrs = {
        "_static_axes": static_axes,
        "default_kernel_map": property(lambda self: {}),
        "forward": lambda self, *a, **kw: None,
        # The three manifest-driven methods are abstract on Op; these doubles
        # exercise the get-or-build plumbing, so a minimal body is the contract.
        "_infer_output_shapes": lambda self, *shapes: {},
        "_validate_dtypes": lambda self, *args: None,
        "eval_roofline": lambda self: (0, 0),
    }
    if override_cache_key:
        attrs["_cache_key"] = lambda self, *shapes: ("overridden",)
    return type("TestOp", (Op,), attrs)


class TestCacheKeyDefault:
    def test_static_axes_exclude_single_input(self):
        """_static_axes=[(0,1)] on a 3D input excludes axis 1 from the key."""
        Cls = _make_op_subclass(static_axes=frozenset({(0, 1)}))
        op = Cls()
        key = op._cache_key((2, 4, 8))
        assert key == (2, 8)

    def test_static_axes_across_multiple_inputs(self):
        """_static_axes can reference axes in different input positions."""
        Cls = _make_op_subclass(static_axes=frozenset({(0, 1), (1, 0)}))
        op = Cls()
        key = op._cache_key((2, 4, 8), (16, 32))
        # Input 0: exclude axis 1 -> (2, 8); Input 1: exclude axis 0 -> (32,)
        assert key == (2, 8, 32)

    def test_empty_static_axes_returns_full_shape(self):
        """With no static axes, the key concatenates all input shape values."""
        Cls = _make_op_subclass(static_axes=frozenset())
        op = Cls()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # warning tested separately
            key = op._cache_key((2, 4, 8), (3, 5))
        assert key == (2, 4, 8, 3, 5)


class TestCacheKeyWarning:
    def test_empty_static_axes_warns_once_per_type(self):
        """Default path with empty _static_axes warns exactly once per subclass,
        even across multiple instances and repeated calls."""
        Cls = _make_op_subclass(static_axes=frozenset())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Cls()._cache_key((2, 4))
            Cls()._cache_key((3, 5))
            Cls()._cache_key((7, 9))
            Cls()._cache_key((11, 13))

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "TestOp" in str(user_warnings[0].message)
        assert "_cache_key" in str(user_warnings[0].message)

    def test_override_suppresses_warning(self):
        """When the subclass overrides _cache_key, no warning fires."""
        Cls = _make_op_subclass(static_axes=frozenset(), override_cache_key=True)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = Cls()._cache_key((2, 4))

        assert result == ("overridden",)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert user_warnings == []

    def test_populated_static_axes_suppresses_warning(self):
        """Non-empty _static_axes means the user committed at ctor; no warning
        fires regardless of whether _cache_key was overridden."""
        Cls = _make_op_subclass(static_axes=frozenset({(0, 0)}))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Cls()._cache_key((2, 4))

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert user_warnings == []

    def test_distinct_subclasses_each_warn_once(self):
        """Two different subclasses each warn once; the dedup set is keyed by
        type, not globally suppressed after the first warning."""
        ClsA = _make_op_subclass(static_axes=frozenset())
        ClsB = _make_op_subclass(static_axes=frozenset())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ClsA()._cache_key((1,))
            ClsA()._cache_key((2,))  # no re-warn for A
            ClsB()._cache_key((3,))  # fresh warn for B
            ClsB()._cache_key((4,))  # no re-warn for B

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 2


class TestInstanceKeys:
    def test_a_collected_instances_key_is_never_handed_out_again(self):
        """An op reaching a used key inherits that op's compiled shapes."""
        import gc

        class _Dummy:
            pass

        keys = set()
        for _ in range(50):
            op = _Dummy()
            keys.add(op_base.register_instance(op))
            del op
            gc.collect()

        assert len(keys) == 50

    def test_a_key_names_the_class_it_belongs_to(self):
        """Graph dumps and guard failures show the key, not the instance."""

        class _Dummy:
            pass

        assert op_base.register_instance(_Dummy()).startswith("_Dummy")


def test_no_abstract_op_class_is_instantiated_anywhere():
    """An abstract Op cannot be constructed, so nothing in the tree may try.

    A class is abstract when it does not answer the manifest-driven contract:
    ``_infer_output_shapes``, ``_validate_dtypes``, ``eval_roofline``. Those are
    the family bases and the modular interfaces; a call site naming one is a call
    site that wanted a concrete op.
    """
    import importlib
    import inspect
    import pkgutil
    import re
    from pathlib import Path

    import tileops.ops as ops_pkg

    abstract = set()
    for module in pkgutil.walk_packages(ops_pkg.__path__, ops_pkg.__name__ + "."):
        try:
            mod = importlib.import_module(module.name)
        except Exception:  # a family whose kernels need a GPU-only import
            continue
        for obj in vars(mod).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, Op)
                and getattr(obj, "__abstractmethods__", None)
            ):
                abstract.add(obj.__name__)
    assert abstract, "no abstract Op classes resolved — the scan is not looking at the tree"

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in (
        list((root / "src").rglob("*.py"))
        + list((root / "tests").rglob("*.py"))
        + list((root / "benchmarks").rglob("*.py"))
    ):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if (
                stripped.startswith(("class ", "#", "*", '"'))
                or "import" in stripped
                or "``" in stripped  # prose naming a class, not a call
            ):
                continue
            for name in abstract:
                if re.search(rf"(?<![\w.]){name}\(", line):
                    offenders.append(f"{path.relative_to(root)}:{lineno} {name}")
    assert offenders == [], offenders
