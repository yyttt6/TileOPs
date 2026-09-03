#!/usr/bin/env python3
"""Flag coverage JSONs that are older than the kernel code they describe.

A canonical coverage JSON is a *measurement*. If the kernel changed after the
measurement was taken, the JSON describes code that no longer exists -- and because
the project counts coverage from these files, a stale JSON silently misreports
progress in whichever direction the change went.

Both directions have bitten this project (2026-08-26):

* ``AbsFwdOp.json`` said ``blocked`` from the F013 era. T058 then restructured
  ``elementwise_unary.py`` -- incidentally fixing F013 -- and R061 measured 48/48
  clean, but nothing regenerated the JSON. The op counted as failing for a day.
* ``MoePermuteNopadFwdOp.json`` holds ``cmd="adapter unavailable"`` and
  ``per_case=[]`` from before its harness adapter existed, while T043 had actually
  verified the op bit-exact at 4096x7168.

The rule is the mirror image of the dispatch rule already in the playbook -- a
deliverable's mtime must be newer than the code's. Here: a coverage JSON's mtime
must be newer than every kernel file its op's builder transitively reaches.

Resolution is **per-op**, not per-family. A first attempt treated every
``kernels/*.py`` that the family module imports as an input to every op in that
family; on a live tree that flagged 73 of 84 JSONs, because eight executors touch
kernel files continuously and each family module imports several. A gate that flags
87% of everything carries no information.

So the edge is built from the builder function itself. The names its body references
come from the live function object (``__code__.co_names``, plus any nested code
objects), and each name is resolved through the family module's import statements to
the ``kernels/*.py`` that defines it. ``AddFwdOp`` then depends on
``elementwise_binary.py`` and not on ``elementwise_predicate.py``, even though its
family module imports both.

Reading the live function rather than scanning the source for a literal
``@register("Op")`` decorator matters: most elementwise ops are registered in a loop
(``for name in NAMES: register(name)(fn)``), so there is no literal to find. An earlier
AST-only version silently left 40 of 84 JSONs unchecked -- among them ``AbsFwdOp``, the
op F013 was diagnosed on.

Results are reported in **two tiers**, because mtime cannot tell the two apart on its
own and conflating them destroys the signal:

* ``STALE`` -- a **kernel template** the builder uses is newer than the JSON. Templates
  are shared machinery; an edit there really does change every op built on it. Act on
  these.
* ``SIBLING`` -- only the **family module** is newer. That file is where builders are
  registered, so it changes every time an executor adds an unrelated op to the same
  family. Usually harmless, occasionally real (the builder's own body lives there too).
  Review, do not re-measure blindly.

Collapsing both into one list flagged 33 of 44 JSONs on a live tree -- almost entirely
sibling edits from eight concurrent executors.

Note what this gate cannot see: a change that leaves mtimes untouched, or a change in
``tilelang-ascend`` itself. The exact form of this check is artifact identity, not
mtime -- a JSON is trustworthy iff current codegen reproduces the artifact SHA256 the
JSON was measured against (see FAILED_ATTEMPTS F014, DECISIONS D017). mtime is the
approximation available until coverage JSONs carry that hash.

Usage:
    python audit_stale_coverage.py [--coverage-dir DIR]
Exit code 0 = no stale JSONs, 1 = at least one is stale.
"""
import argparse
import ast
import json
import os
import sys


def _import_origin(tree, kernels_dir):
    """imported name -> kernels/*.py that defines it, for one module's imports."""
    origin = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module.rsplit(".", 1)[-1]
            path = os.path.join(kernels_dir, f"{mod}.py")
            if os.path.exists(path):
                for alias in node.names:
                    origin[alias.asname or alias.name] = path
        elif isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.rsplit(".", 1)[-1]
                path = os.path.join(kernels_dir, f"{mod}.py")
                if os.path.exists(path):
                    origin[alias.asname or mod] = path
    return origin


def _referenced_names(func, _depth=0):
    """Global names the function body references, including nested code objects."""
    code = getattr(func, "__code__", None)
    if code is None or _depth > 4:
        return set()
    names = set(code.co_names)
    for const in code.co_consts:
        if hasattr(const, "co_names"):
            names |= set(const.co_names)
    closure = getattr(func, "__wrapped__", None)
    if closure is not None:
        names |= _referenced_names(closure, _depth + 1)
    for cell in (getattr(func, "__closure__", None) or ()):
        try:
            inner = cell.cell_contents
        except ValueError:
            continue
        if callable(inner) and hasattr(inner, "__code__"):
            names |= _referenced_names(inner, _depth + 1)
    return names


def builder_kernel_deps(builder, family_file, kernels_dir):
    """kernels/*.py files this specific builder reaches, via its referenced names."""
    try:
        tree = ast.parse(open(family_file).read())
    except (SyntaxError, OSError):
        return set()
    origin = _import_origin(tree, kernels_dir)
    used = _referenced_names(builder)
    return {origin[name] for name in used if name in origin}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage-dir", default="tileops-ascend-harness/coverage")
    args = ap.parse_args(argv)

    import tileops_ascend
    from tileops_ascend._registry import REGISTERED

    pkg_dir = os.path.dirname(tileops_ascend.__file__)
    kernels_dir = os.path.join(pkg_dir, "kernels")

    stale, sibling, unresolved, checked = [], [], [], 0
    for name in sorted(os.listdir(args.coverage_dir)):
        if not name.endswith(".json") or "_selftest" in name:
            continue
        path = os.path.join(args.coverage_dir, name)
        try:
            op = json.load(open(path)).get("op") or name[:-5]
        except (json.JSONDecodeError, OSError):
            stale.append((name, "unreadable JSON", None))
            continue
        builder = REGISTERED.get(op)
        if builder is None:
            continue          # not registered: T085's accounting problem, not staleness
        family_file = getattr(sys.modules.get(builder.__module__), "__file__", None)
        if not family_file:
            continue
        reached = builder_kernel_deps(builder, family_file, kernels_dir)
        inputs = reached | {family_file}
        if not reached:
            # No kernel module resolved from the builder's referenced names. The
            # family file is still compared below, but the template edge is missing,
            # so say so rather than imply a clean result. A gate that quietly drops
            # part of its input is the failure mode F015 was written about.
            unresolved.append(op)
        checked += 1
        json_mtime = os.path.getmtime(path)
        newer_kernels = [f for f in inputs
                         if f != family_file and os.path.getmtime(f) > json_mtime]
        if newer_kernels:
            newest = max(newer_kernels, key=os.path.getmtime)
            stale.append((op, os.path.relpath(newest)))
        elif os.path.getmtime(family_file) > json_mtime:
            sibling.append((op, os.path.relpath(family_file)))

    print(f"checked {checked} coverage JSONs against their per-op kernel inputs")
    if stale:
        print(f"\nSTALE  {len(stale)} JSON(s) measured before a kernel template they "
              f"use changed -- re-measure before counting these:")
        for op, cause in stale:
            print(f"        {op:46} newer template: {cause}")
    if sibling:
        print(f"\nSIBLING  {len(sibling)} JSON(s) older only than their family module "
              f"(usually an unrelated op was registered there) -- review, do not "
              f"re-measure blindly:")
        for op, cause in sibling:
            print(f"        {op:46} newer family:   {cause}")
    if unresolved:
        print(f"\nUNCHECKED  {len(unresolved)} JSON(s) whose op could not be tied to a "
              f"kernel template -- no imported kernel name found in the builder body. "
              f"Only their family module was compared, so a template edit would NOT "
              f"be caught for these:")
        for op in sorted(unresolved):
            print(f"        {op}")
    if not stale and not sibling and not unresolved:
        print("OK   no coverage JSON predates the code it measures")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
