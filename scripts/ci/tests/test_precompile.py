"""The compile-only contract: what a pre-compilation pass may and may not assume."""

import dataclasses
import sys

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


def test_an_identical_program_deduplicates():
    """The same kernel body built twice is two objects and one compile.

    ``Compilation.identity`` hashes the program text TileLang keys its cache on, so a
    builder called once per case does not hand the pass the same work repeatedly.
    """

    class StandIn:
        def __init__(self, text):
            self.text = text

        def script(self, show_meta=False):
            return self.text

    a = Compilation(program=StandIn("main: 1"), pass_configs={"x": 1})
    b = Compilation(program=StandIn("main: 1"), pass_configs={"x": 1})
    c = Compilation(program=StandIn("main: 2"), pass_configs={"x": 1})
    assert a == b
    assert a != c
    assert len({a, b, c}) == 2


def test_the_counter_sees_every_launch_subprocess_makes(monkeypatch):
    """The compiler count must observe run(), check_output() and Popen() alike.

    A cache hit is proved by the *absence* of a compiler launch, so a wrapper that only
    saw one of the three ways ``subprocess`` starts a process would let a rebuild pass for
    a reuse. The launches below use a shell builtin, so no real binary is required.
    """
    import subprocess

    from scripts.ci.precompile import CompilerCounter

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    CompilerCounter.install()
    try:
        CompilerCounter.count = 0
        subprocess.run(["true"], check=True)
        subprocess.check_output(["true"])
        subprocess.Popen(["true"]).wait()
        assert CompilerCounter.count == 0, "a non-compiler launch was counted"

        # A compiler name is what is counted, whatever the arguments or the path prefix.
        for argv in (["/opt/ascend/bin/bisheng", "--version"], ["ld.lld"]):
            with pytest.raises(OSError):
                subprocess.run(argv, check=True)
        assert CompilerCounter.count == 2
    finally:
        CompilerCounter.remove()
