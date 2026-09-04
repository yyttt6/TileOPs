"""Numeric constants the op layer itself needs.

Format facts, not hardware facts: a value here is a property of a dtype or of a
reference formula, so it reads the same on every target.
"""

__all__ = ["FP8_E4M3_MAX"]

#: Largest finite ``float8_e4m3fn`` value; quantizers clamp to +-FP8_E4M3_MAX.
FP8_E4M3_MAX: float = 448.0
