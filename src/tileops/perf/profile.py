"""Hardware profile loader.

Reads YAML profiles from src/tileops/perf/profiles/ and returns them as dicts.
This is the M6 -> M5 data contract interface (see docs/design/architecture.md).

YAML files store only ``theoretical`` and ``calibration`` values.
``effective = theoretical * calibration`` is computed at load time.

**Roof keys name Ascend compute units.** A 910B1 AI Core is a Cube unit for
matrix contractions and a Vector unit for everything else, so a roof key is
``"cube.<dtype>"`` or ``"vector.<dtype>"``. The two names are the hardware's
own; nothing here invents a third vocabulary for them.
"""

from pathlib import Path

import yaml

_PROFILES_DIR = Path(__file__).parent / "profiles"

# Keys whose values are numeric but arrive as strings from PyYAML
# (scientific notation like 4800e9 is not YAML-native float syntax).
_NUMERIC_KEYS = frozenset({"theoretical", "calibration", "calibration_burst", "effective"})


def get_profile_path(soc_name: str) -> Path:
    """Return the path to a hardware profile YAML.

    Args:
        soc_name: Profile name without extension (e.g. "ascend910b1").

    Returns:
        Path to the YAML file.

    Raises:
        FileNotFoundError: If no profile exists for the given name.
    """
    path = _PROFILES_DIR / f"{soc_name}.yaml"
    if not path.exists():
        available = [p.stem for p in _PROFILES_DIR.glob("*.yaml")]
        raise FileNotFoundError(f"No hardware profile '{soc_name}'. Available: {available}")
    return path


def _coerce_numeric_strings(obj, key=None):
    """Recursively convert known numeric string values to floats.

    Only converts values whose dict key is in ``_NUMERIC_KEYS``, avoiding
    unintended coercion of string fields like ``compute_capability``.
    """
    if isinstance(obj, dict):
        return {k: _coerce_numeric_strings(v, key=k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numeric_strings(v) for v in obj]
    if isinstance(obj, str) and key in _NUMERIC_KEYS:
        try:
            return float(obj)
        except ValueError:
            return obj
    return obj


def _inject_effective(profile):
    """Compute effective = theoretical * calibration for hbm and the compute sections.

    A section with only ``theoretical`` is left alone: profiles are created from
    datasheet numbers first and calibrated by benchmarks/hardware/ afterwards.
    """
    sections = [profile.get("hbm")]
    for group in ("cube", "vector"):
        sections.extend(profile.get(group, {}).values())
    for section in sections:
        if isinstance(section, dict) and "effective" not in section and "calibration" in section:
            section["effective"] = section["theoretical"] * section["calibration"]


# Cube dtype keys, by the dtype the contraction consumes. Encode side of the
# roof-key format; ``resolve_roof`` is the decode.
#
# 910B1 has no FP8 Cube path, so ``float8_*`` is deliberately absent: a soft-FP8
# contraction computes in fp16 and is priced there (see the handwritten-baseline
# fairness clause on soft-FP8). Asking for an fp8 roof raises rather than
# quoting a ceiling the hardware does not have.
_CUBE_DTYPE_KEYS = {
    "float16": "fp16",
    "bfloat16": "bf16",
    "float32": "fp32",
}


def cube_roof(dtype) -> str:
    """Cube roof key for a contraction computing at *dtype*.

    The Cube unit is the 910B1's matrix engine: it is what prices a matmul's
    FLOPs, the way tensor cores do on a GPU.

    Args:
        dtype: The dtype the matmul consumes — a ``torch.dtype`` or its
            string name. Ops pass ``self.dtype`` directly.

    Returns:
        A profile key such as ``"cube.bf16"``.

    Raises:
        ValueError: If *dtype* has no Cube section in the profile schema
            (including ``None`` — the op has not bound a dtype yet).
    """
    name = str(dtype).removeprefix("torch.") if dtype is not None else None
    key = _CUBE_DTYPE_KEYS.get(name) if name is not None else None
    if key is None:
        raise ValueError(
            f"no Cube roof for dtype {dtype!r}; known dtypes: {sorted(_CUBE_DTYPE_KEYS)}"
        )
    return f"cube.{key}"


def find_profile(device_name: str) -> dict | None:
    """Load the profile whose ``device`` field names *device_name*, or ``None``.

    Args:
        device_name: The device name the runtime reports
            (``torch.npu.get_device_name()``), e.g. ``"Ascend910B1"``.

    Returns:
        The loaded profile dict, or ``None`` when no profile claims the
        device — the caller leaves speed-of-light readings blank rather
        than guessing a ceiling.
    """
    for path in _PROFILES_DIR.glob("*.yaml"):
        profile = load_profile(path.stem)
        if profile.get("device") == device_name:
            return profile
    return None


def resolve_roof(profile: dict, key: str) -> dict | None:
    """Resolve a roof key like ``"cube.bf16"`` to its profile section.

    Args:
        profile: A dict from :func:`load_profile`.
        key: ``"<unit>.<dtype>"`` as declared by ``Op.compute_roof()``.

    Returns:
        The section dict (with ``theoretical`` / ``effective``), or ``None``
        when the profile has no calibrated entry for the key.
    """
    unit, _, dt = key.partition(".")
    section = (profile.get(unit) or {}).get(dt)
    if isinstance(section, dict) and "effective" in section and "theoretical" in section:
        return section
    return None


def load_profile(soc_name: str) -> dict:
    """Load a hardware profile as a dict.

    Args:
        soc_name: Profile name without extension (e.g. "ascend910b1").

    Returns:
        Dict with keys: device, soc, hbm, cube, vector. Each rate section
        includes a computed ``effective`` field.
    """
    path = get_profile_path(soc_name)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data = _coerce_numeric_strings(data)
    _inject_effective(data)
    return data
