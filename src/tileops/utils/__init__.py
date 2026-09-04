"""Helpers with no home of their own.

Device probes used to live here -- architecture, multiprocessor count, device
name -- because in-tree kernels selected on them. Kernels now come from backend
distributions, and a backend reads its own hardware inside its ``build_kernel``,
where the shapes and dtypes it needs are also in scope. Nothing in this package
touches a device any more.
"""

from .utils import str2dtype

__all__ = ["str2dtype"]
