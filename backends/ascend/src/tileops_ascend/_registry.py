"""Registration helpers. PM-owned: frozen after Wave 1.

A family module never calls ``tileops.backend.register_kernel_builder`` directly; it
calls `register`. One indirection buys three things a direct call cannot:

* the target name lives in exactly one place, so a typo in a family module cannot
  create a second, silently-unreachable target;
* every builder is wrapped by `_guard`, which turns a builder that returns a
  non-callable into an error naming the op rather than a ``TypeError`` at launch;
* the registration table is enumerable for tests and for the coverage report.
"""

from __future__ import annotations

from typing import Callable

from tileops.backend import register_kernel_builder

#: The one place this backend's target is named.
TARGET = "ascend"

#: op name -> builder, in registration order. Read by tests and the coverage report.
REGISTERED: dict[str, Callable] = {}


def register(op: str) -> Callable[[Callable], Callable]:
    """Decorator: register the decorated function as *op*'s ``build_kernel``.

    The decorated function's signature must be the op's manifest signature: one
    positional ``TensorSpec | None`` per ``signature.inputs`` entry in declaration
    order, then ``signature.params`` by keyword.

    Raises:
        RuntimeError: *op* already has a builder in this process.
    """
    def decorate(build_kernel: Callable) -> Callable:
        if op in REGISTERED:
            raise RuntimeError(
                f"{op} already has a builder ({REGISTERED[op].__module__}."
                f"{REGISTERED[op].__qualname__}); {build_kernel.__module__}."
                f"{build_kernel.__qualname__} cannot replace it."
            )
        guarded = _guard(op, build_kernel)
        REGISTERED[op] = build_kernel
        register_kernel_builder(op=op, target=TARGET, build_kernel=guarded)
        return build_kernel

    return decorate


def _guard(op: str, build_kernel: Callable) -> Callable:
    """Wrap *build_kernel* so a non-callable return is reported here, not at launch.

    The op layer checks ``callable()`` too, but its message names the target rather
    than the builder; this one names the builder that produced the bad value.
    """
    def build(*inputs, **params):
        kernel = build_kernel(*inputs, **params)
        if not callable(kernel):
            raise TypeError(
                f"{build_kernel.__module__}.{build_kernel.__qualname__} (op {op!r}) "
                f"returned {kernel!r}, which is not callable. A build_kernel must "
                f"return something invocable as (*tensors)."
            )
        return kernel

    build.__name__ = getattr(build_kernel, "__name__", "build")
    build.__qualname__ = getattr(build_kernel, "__qualname__", "build")
    build.__module__ = getattr(build_kernel, "__module__", __name__)
    return build
