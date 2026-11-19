"""Conservative per-entry selection for the GEMM benchmark harness."""

import ast
import subprocess

REGISTRY = "benchmarks/_cases/gemm.py"
WORKLOADS = "workloads/gemm.py"


def _read(root, revision, path):
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"], cwd=root, capture_output=True, text=True
    )
    return (
        ast.parse(result.stdout, filename=path) if result.returncode == 0 else ast.Module(body=[])
    )


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _signature(nodes):
    return [ast.dump(node) for node in nodes]


def _simple_argument(node):
    return isinstance(node, (ast.Name, ast.Constant)) or (
        isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
    )


def _registry(module):
    entries = None
    other = []
    for node in module.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "ENTRIES"
        ):
            if entries is not None or not isinstance(node.value, ast.Dict):
                return None
            entries = {}
            for key, value in zip(node.value.keys, node.value.values, strict=True):
                if (
                    not isinstance(key, ast.Constant)
                    or not isinstance(key.value, str)
                    or not key.value.endswith("Op")
                    or key.value in entries
                    or not isinstance(value, ast.Call)
                    or not isinstance(value.func, ast.Name)
                    or value.func.id != "Entry"
                    or any(not _simple_argument(arg) for arg in value.args)
                    or any(
                        kw.arg is None or not _simple_argument(kw.value) for kw in value.keywords
                    )
                ):
                    return None
                entries[key.value] = value
        elif isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)) or (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        ):
            other.append(node)
        else:
            return None
    return (entries, other) if entries else None


def _definitions(module):
    definitions = {}
    other = []
    for node in module.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            if node.name in definitions:
                return None
            definitions[node.name] = node
        else:
            other.append(node)
    return definitions, other


def _dependencies(names, definitions):
    seen = set()
    pending = names & definitions.keys()
    while pending:
        name = pending.pop()
        seen.add(name)
        pending.update((_names(definitions[name]) & definitions.keys()) - seen)
    return seen


def select_gemm_harness(root, base, changed):
    """Return a proven subset, or all known GEMM entries when proof fails."""
    old_module, new_module = (_read(root, rev, REGISTRY) for rev in (base, "HEAD"))
    old_registry, new_registry = map(_registry, (old_module, new_module))
    family = {
        n.value
        for module in (old_module, new_module)
        for n in ast.walk(module)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and n.value.endswith("Op")
        and n.value.isidentifier()
    }
    if old_registry is None or new_registry is None:
        return family
    old_entries, old_other = old_registry
    new_entries, new_other = new_registry
    family.update(old_entries.keys() | new_entries.keys())
    if _signature(old_other) != _signature(new_other) or list(old_entries) != list(new_entries):
        return family
    if any(
        _names(module) & {"globals", "locals", "eval", "exec", "__import__"}
        for module in (old_module, new_module)
    ):
        return family

    affected = {op for op in new_entries if ast.dump(old_entries[op]) != ast.dump(new_entries[op])}
    if WORKLOADS not in changed:
        return affected

    # Attribute the changed workload definitions through each Entry's workload
    # import and its local dependency closure, in both revisions.
    workload_modules = [_read(root, rev, WORKLOADS) for rev in (base, "HEAD")]
    if any(
        _names(module) & {"globals", "locals", "eval", "exec", "__import__"}
        for module in workload_modules
    ):
        return family
    modules = list(map(_definitions, workload_modules))
    if None in modules:
        return family
    (old_defs, old_top), (new_defs, new_top) = modules
    if _signature(old_top) != _signature(new_top):
        return family
    common = old_defs.keys() & new_defs.keys()
    if [n for n in old_defs if n in common] != [n for n in new_defs if n in common]:
        return family
    touched = old_defs.keys() ^ new_defs.keys()
    touched.update(n for n in common if ast.dump(old_defs[n]) != ast.dump(new_defs[n]))
    imports = {
        alias.asname or alias.name: alias.name
        for node in new_other
        if isinstance(node, ast.ImportFrom) and node.module == "workloads.gemm"
        for alias in node.names
    }
    if not imports or "*" in imports:
        return family
    registry_helpers = {node.name: node for node in new_other if isinstance(node, ast.FunctionDef)}
    mapped = set()
    for entries, defs in ((old_entries, old_defs), (new_entries, new_defs)):
        for op, entry in entries.items():
            names = _names(entry)
            for helper in _dependencies(names, registry_helpers):
                names |= _names(registry_helpers[helper])
            roots = {imports[n] for n in names & imports.keys()}
            if not roots or not roots <= defs.keys():
                return family
            dependencies = _dependencies(roots, defs)
            # This concrete basis workload is exercised by the W4A16 public tests.
            if "GEMMW4A16Workload" in roots:
                dependencies.add("GEMMW4A16BasisWorkload")
            hit = touched & dependencies
            if hit:
                affected.add(op)
                mapped.update(hit)
    return affected if mapped == touched and len(affected) <= 1 else family
