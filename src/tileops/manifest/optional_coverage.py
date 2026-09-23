"""Manifest coverage facts shared by validation and benchmark reporting.

An optional input without a workload supplying it is a valid operator contract,
but its supplied-input path has no declared test coverage. This includes entries
with zero workloads and spec-only entries. No workload or signature is mutated.
"""

from __future__ import annotations


def optional_inputs_untested(entry: dict) -> list[str]:
    """Return optional names without an explicit ``<name>_shape`` workload.

    This describes declared workload coverage, not incidental adapter inputs or
    a claim that a declared workload has executed successfully. Malformed schema
    remains the validator's responsibility.
    """
    signature = entry.get("signature")
    inputs = signature.get("inputs") if isinstance(signature, dict) else None
    if not isinstance(inputs, dict):
        return []
    workloads = entry.get("workloads")
    rows = [row for row in workloads if isinstance(row, dict)] if isinstance(workloads, list) else []
    return sorted(
        name for name, attrs in inputs.items()
        if isinstance(name, str) and isinstance(attrs, dict)
        and attrs.get("optional") is True
        and not any(f"{name}_shape" in row for row in rows)
    )
