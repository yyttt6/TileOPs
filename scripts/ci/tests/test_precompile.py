"""The compile-only contract: what a pre-compilation pass may and may not assume."""

import dataclasses

import pytest

from tileops._kernels.kernel_base import Compilation, Kernel

pytestmark = pytest.mark.smoke


def test_the_default_reports_nothing():
    """A kernel that does not implement the contract opts out by returning nothing."""

    class Plain(Kernel):
        def forward(self, *args, **kwargs):  # pragma: no cover - never called
            raise NotImplementedError

    assert Plain.compilations({"input": ((4,), "float16")}, {}) == ()


def test_a_description_carries_every_compile_option():
    """What par_compile needs must travel with the program, or the pass cannot group it."""
    names = {f.name for f in dataclasses.fields(Compilation)}
    assert {
        "program",
        "out_idx",
        "target",
        "target_host",
        "execution_backend",
        "pass_configs",
        "compile_flags",
    } <= names


def test_descriptions_with_different_options_are_different():
    """Two builds under different options must not deduplicate into one."""
    a = Compilation(program="p", target="ascend", pass_configs={"x": 1})
    b = Compilation(program="p", target="ascend", pass_configs={"x": 2})
    assert a != b


def test_a_description_builds_with_its_own_options(monkeypatch):
    """build() must forward the description's options, not defaults."""
    seen = {}

    class FakeTileLang:
        @staticmethod
        def compile(program, **kwargs):
            seen.update(program=program, **kwargs)
            return "built"

    import sys

    monkeypatch.setitem(sys.modules, "tilelang", FakeTileLang)
    item = Compilation("program", out_idx=-1, target="ascend", compile_flags=["-O2"])
    assert item.build() == "built"
    assert seen["program"] == "program"
    assert seen["target"] == "ascend"
    assert seen["compile_flags"] == ["-O2"]
    assert seen["out_idx"] == -1


def test_mutable_compile_options_still_hash_and_deduplicate():
    """The configurations a real builder passes are dicts and lists, not tuples.

    A frozen dataclass's automatic hash walks the fields and raises on the first
    unhashable one, so the identity has to canonicalise them. This is the case the
    field-presence test above cannot see.
    """
    a = Compilation(
        program=object(),
        out_idx=[1, 2],
        pass_configs={"tl.enable_fast_math": False, "tl.disable_warp_specialized": True},
        compile_flags=["-O2", "-fPIC"],
    )
    b = Compilation(
        program=a.program,
        out_idx=[1, 2],
        # Same options, different insertion order: one key.
        pass_configs={"tl.disable_warp_specialized": True, "tl.enable_fast_math": False},
        compile_flags=["-O2", "-fPIC"],
    )
    assert hash(a) == hash(b)
    assert a == b
    assert len({a, b}) == 1


def test_a_changed_option_is_a_different_key():
    """Deduplication must not collapse two builds that differ in an option."""
    program = object()
    a = Compilation(program=program, pass_configs={"x": 1})
    b = Compilation(program=program, pass_configs={"x": 2})
    c = Compilation(program=program, pass_configs={"x": 1}, compile_flags=["-O3"])
    assert a != b
    assert a != c
    assert len({a, b, c}) == 3


def test_a_different_program_is_a_different_key():
    a = Compilation(program=object())
    b = Compilation(program=object())
    assert a != b
    assert len({a, b}) == 2


def test_grouping_preserves_options_and_caps_workers(monkeypatch):
    import sys
    from types import SimpleNamespace

    from scripts.ci.precompile import compile_programs

    calls = []

    def parallel(programs, **kwargs):
        calls.append((programs, kwargs))
        return [object() for _ in programs]

    monkeypatch.setitem(sys.modules, "tilelang.jit", SimpleNamespace(par_compile=parallel))
    a = Compilation(object(), out_idx=[1], pass_configs={"x": True}, compile_flags=["-O2"])
    b = Compilation(object(), out_idx=[1], pass_configs={"x": True}, compile_flags=["-O2"])
    c = Compilation(object(), out_idx=[1], pass_configs={"x": False}, compile_flags=["-O2"])
    count, workers = compile_programs([a, b, c], 16)
    assert count == 3 and workers == [2, 1]
    assert calls[0][1]["pass_configs"] == {"x": True}
    assert calls[1][1]["pass_configs"] == {"x": False}
    assert calls[0][1]["out_idx"] == [1]
    assert calls[0][1]["compile_flags"] == ["-O2"]
    assert calls[0][1]["ignore_error"] is False


def test_partial_compiler_results_are_not_success(monkeypatch):
    import sys
    from types import SimpleNamespace

    from scripts.ci.precompile import compile_programs

    monkeypatch.setitem(
        sys.modules, "tilelang.jit", SimpleNamespace(par_compile=lambda *a, **k: [None])
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        compile_programs([Compilation(object())], 1)


def test_unreserved_import_rejected_with_receipt(monkeypatch, tmp_path):
    import json
    import sys

    from benchmarks.ascend import guard
    from scripts.ci import precompile

    def reject():
        raise RuntimeError("reservation missing")

    monkeypatch.setattr(guard, "require_reservation", reject)
    monkeypatch.setattr(precompile, "plan_for", lambda *a: pytest.fail("enumerated before guard"))
    output = tmp_path / "receipt.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "precompile",
            "--source",
            str(tmp_path),
            "--operators",
            "ReLUFwdOp",
            "--output",
            str(output),
        ],
    )
    assert precompile.main() == 1
    receipt = json.loads(output.read_text())
    assert receipt["stage"] == "reservation"
    assert receipt["status"] == "failed"
