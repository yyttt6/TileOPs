"""Pre-compile the programs a changed operator's cases need, before measurement starts.

Compilation is around 99% of a case's wall time on a runner that has not built the branch
before: one case measured 20 955 ms cold against 0.8 ms once built. That cost runs on the
critical path with the device held, one program at a time. This pass takes the programs the
measurement would build, hands them to TileLang's own parallel compiler, and lets the
measurement find them in the cache.

Nothing here instantiates a kernel, builds benchmark inputs or launches on the device: the
programs come from each kernel's ``compilations()``, a pure function of the call record, and
the call record comes from the manifest rather than from tensors. An operator whose kernel
does not implement ``compilations()`` is reported and skipped; its cases then compile lazily
exactly as before.

**The receipt reports compiler invocations, not cache directory contents.** Counting entries
in the cache tree says what is on disk; it cannot tell a build that found its program from
one that rebuilt over it. Every exit path therefore writes ``compilers``, the number of
device-compiler subprocesses the pass started, and that number -- compared with what the
measurement path spends afterwards -- is what separates a reuse from a rebuild.

TileLang caches in two layers and they answer differently: the kernel entry is keyed on the
program text and the pass options, while the device binary is keyed on the emitted code and
the device compile flags. A changed pass option that leaves the emitted code identical
therefore rebuilds the entry and reuses the binary, which costs no compiler launch -- so the
receipt reports the count rather than naming a verdict from it.
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path


def _select(registered, interface):
    """The one kernel class of *interface* this op registers.

    Selection happens on the class, not on an instance: the implementations registered for an
    op are keyed, and an op with exactly one implementation of the interface has nothing to
    choose between. Where several are registered the pass declines rather than guessing,
    because picking wrong pre-compiles programs the call will not use -- wasted, not wrong,
    but wasted every run.
    """
    keys = [key for key, cls in registered.items() if issubclass(cls, interface)]
    if len(keys) != 1:
        raise ValueError(
            f"{len(keys)} implementations of {interface.__name__} are registered "
            f"({sorted(keys)}); this pass only serves ops with exactly one"
        )
    return registered[keys[0]]


def plan_for(op_name, source):
    """(descriptions, skipped case ids, unsupported reason) for every case of *op_name*."""
    sys.path[:0] = [str(source / "src"), str(source)]
    from benchmarks.api import cases
    from tileops.backend import _registry
    from tileops.backend._registry import REGISTERED_KERNEL_TYPES
    from tileops.manifest import load_manifest

    manifest = load_manifest()
    _registry.ensure_loaded()
    if op_name not in REGISTERED_KERNEL_TYPES:
        # A source tree run from PYTHONPATH has no entry-point metadata, so the backend
        # modules are never discovered. Import the family's kernel package directly; on an
        # installed tree this is already done and the import returns from sys.modules.
        importlib.import_module(f"tileops._kernels.ascend.{manifest[op_name]['family']}")
    op_class = getattr(importlib.import_module(f"tileops.{manifest[op_name]['family']}"), op_name)
    interface = next(iter(op_class.interfaces.values()))

    registered = REGISTERED_KERNEL_TYPES.get(op_name, {})
    if not registered:
        return [], [], f"{op_name} has no registered kernel implementation"

    programs, skipped = {}, []
    for case in cases(op_name):
        tensors, params = case._call.tensors, case._call.params
        try:
            kernel_class = _select(registered, interface)
        except ValueError as exc:
            return [], [], f"{op_name}: {exc}"
        found = tuple(kernel_class.compilations(tensors, params))
        if not found:
            skipped.append(case.id)
            continue
        for item in found:
            programs.setdefault(item, case.id)
    return list(programs), skipped, None


def _write(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


class CompilerCounter:
    """Count the device-compiler subprocesses this process starts.

    A cache hit is proved by showing the compiler did *not* run, so the pass has to see
    every launch. TileLang reaches the compiler only through ``subprocess``, so the count is
    taken by wrapping ``Popen`` for the duration of the build: ``subprocess.run`` and
    ``check_output`` both look ``Popen`` up on the module, so one wrapper covers them.
    """

    count = 0
    _original = None

    @classmethod
    def install(cls):
        import subprocess

        if cls._original is not None:
            return
        cls.count = 0
        cls._original = subprocess.Popen
        original = cls._original

        def popen(args, *rest, **kwargs):
            if _is_device_compiler(args):
                cls.count += 1
            return original(args, *rest, **kwargs)

        subprocess.Popen = popen

    @classmethod
    def remove(cls):
        import subprocess

        if cls._original is None:
            return
        subprocess.Popen = cls._original
        cls._original = None


def _is_device_compiler(args):
    """Whether an argv starts the Ascend device compiler or its linker."""
    if isinstance(args, bytes):
        args = args.decode()
    if isinstance(args, str):
        return Path(args).name in ("bisheng", "ld.lld")
    try:
        first = args[0]
    except (TypeError, IndexError, KeyError):
        return False
    return Path(str(first)).name in ("bisheng", "ld.lld")


def _identity(source):
    """Which source tree and toolchain this receipt describes.

    A receipt without these cannot be tied to the thing it claims to be about: the same
    program set built from another tree or another compiler is not the same evidence.
    Absent git or an unreadable tree is recorded as ``None`` rather than failing the pass,
    which is an optimisation and must not fail the job over its own bookkeeping.
    """
    import hashlib
    import subprocess

    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=source, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    toolchain = {}
    for name in ("BISHENG_HOME", "ASCEND_HOME_PATH", "ASCEND_NPU_ARCH"):
        toolchain[name] = os.environ.get(name)
    try:
        import tilelang

        toolchain["tilelang"] = tilelang.__version__
    except ImportError:
        toolchain["tilelang"] = None
    bisheng = None
    for candidate in (
        os.path.join(os.environ.get("BISHENG_HOME", ""), "bin", "bisheng"),
        os.environ.get("ASCEND_HOME_PATH", "") and
        os.path.join(os.environ["ASCEND_HOME_PATH"], "bin", "bisheng"),
    ):
        if candidate and os.path.isfile(candidate):
            bisheng = candidate
            break
    if bisheng is None:
        import shutil

        bisheng = shutil.which("bisheng")
    if bisheng:
        try:
            with open(bisheng, "rb") as handle:
                toolchain["bisheng_sha256"] = hashlib.sha256(handle.read()).hexdigest()
            toolchain["bisheng_version"] = subprocess.check_output(
                [bisheng, "--version"], text=True, stderr=subprocess.STDOUT,
                timeout=60,
            ).splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError, IndexError):
            toolchain["bisheng_sha256"] = None
    return dict(commit=git("rev-parse", "HEAD"), tree=git("rev-parse", "HEAD^{tree}"),
                toolchain=toolchain)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--op", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("TILEOPS_PRECOMPILE_WORKERS", "16")),
        help="compile workers; capped by the program count and never left to the library "
        "default, so a large operator cannot saturate the runner. Each compile is a "
        "single-core bisheng subprocess, so the useful ceiling is the program count, not "
        "the core count",
    )
    args = parser.parse_args()

    receipt = dict(op=args.op, source=str(args.source.resolve()),
                   identity=_identity(args.source.resolve()))
    try:
        programs, skipped, unsupported = plan_for(args.op, args.source.resolve())
    except Exception as exc:
        # Enumeration is as much a stage of this pass as compiling is. A receipt that only
        # records compile failures would report "no programs" for an operator that could not
        # be enumerated at all, which reads like an operator that does not support this.
        _write(
            args.output,
            {**receipt, "status": "failed", "stage": "enumeration", "compilers": None,
             "error": f"{type(exc).__name__}: {exc}"},
        )
        return 1

    receipt.update(programs=len(programs), skipped_cases=skipped)
    if unsupported:
        _write(args.output, {**receipt, "status": "unsupported", "reason": unsupported,
                             "compilers": None})
        return 0
    if not programs:
        _write(
            args.output,
            {**receipt, "status": "skipped", "compilers": None,
             "reason": "no case's kernel implements compilations()"},
        )
        return 0

    import tilelang
    from tilelang.jit import par_compile

    try:
        # One group per distinct compile option set: a program built under different
        # options produces an object the later call will not find, so mixing them wastes
        # the build. The description's identity canonicalises dict/list options, which a
        # tuple of the raw fields would raise on.
        groups = {}
        for item in programs:
            groups.setdefault(item.identity()[1:], []).append(item)
    except Exception as exc:
        _write(
            args.output,
            {**receipt, "status": "failed", "stage": "grouping", "compilers": None,
             "error": f"{type(exc).__name__}: {exc}"},
        )
        return 1

    workers = max(1, min(args.workers, len(programs)))
    began, compiled, errors = time.monotonic(), 0, []
    CompilerCounter.install()
    try:
        for items in groups.values():
            try:
                built = par_compile(
                    [item.program for item in items],
                    out_idx=items[0].out_idx,
                    target=items[0].target,
                    target_host=items[0].target_host,
                    execution_backend=items[0].execution_backend,
                    pass_configs=items[0].pass_configs,
                    compile_flags=items[0].compile_flags,
                    num_workers=workers,
                )
                compiled += len(built)
            except Exception as exc:  # recorded; the measurement loop still reports per case
                errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        compilers = CompilerCounter.count
        CompilerCounter.remove()

    # What a cache hit looks like, which depends on how many layers answered. A program
    # TileLang has built before costs no device compile; one whose kernel entry it has but
    # whose device binary a changed compile-flag set invalidated costs a device compile
    # only. Neither is a hit unless the compiler stayed idle for it, so the receipt says
    # which happened rather than assuming.

    _write(
        args.output,
        {
            **receipt,
            "status": "failed" if errors else "ok",
            "stage": "compilation" if errors else "done",
            "seconds": round(time.monotonic() - began, 3),
            "tilelang": tilelang.__version__,
            "workers": workers,
            "compiled": compiled,
            "groups": len(groups),
            # Every device-compiler launch the pass caused. Zero means nothing was built;
            # the caller compares this with its own count to show the work is not repeated.
            "compilers": compilers,
            "errors": errors,
        },
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
