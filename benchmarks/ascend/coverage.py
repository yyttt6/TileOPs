"""Discover Ascend benchmark coverage, including operators composed of supported delegates."""

import importlib


def include_composites(direct, delegates):
    """Include a composite only after all its declared delegates are supported."""
    supported = set(direct)
    while True:
        added = {
            name for name, children in delegates.items() if children and set(children) <= supported
        }
        if added <= supported:
            return supported
        supported.update(added)


def registered_coverage():
    """Read kernel and composite metadata without constructing ops or accessing a device."""
    from tileops.backend import registry
    from tileops.manifest import load_manifest

    manifest = load_manifest()
    direct = {
        op
        for op, kinds in registry.REGISTERED_KERNEL_TYPES.items()
        if any(name == "ascend" or name.startswith("ascend_") for name in kinds)
    }
    families = {manifest[name]["family"] for name in direct if name in manifest}
    modules = {family: importlib.import_module(f"tileops.{family}") for family in families}
    delegates = {}
    for name, entry in manifest.items():
        module = modules.get(entry["family"])
        cls = getattr(module, name, None) if module is not None else None
        children = getattr(cls, "delegate_types", {})
        if children and not getattr(cls, "kernel_types", {}):
            delegates[name] = sorted({child.__name__ for child in children.values()})
    supported = include_composites(direct, delegates)
    return {
        "operators": sorted(supported),
        "delegates": {name: children for name, children in delegates.items() if name in supported},
    }
