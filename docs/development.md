# Development

How to install, test, lint, and benchmark TileOPs from a source checkout.

Commands here are the ones CI runs. Where CI passes `-c constraints.txt`, use it locally too — that file pins the versions CI validates, and omitting it resolves a different dependency set than the one your PR is tested against.

## Install from source

An Ascend NPU is required to run the test suite. See [Prerequisites](../README.md#installation) for supported Python, PyTorch, CANN, and TileLang versions.

```bash
git clone https://github.com/yyttt6/TileOPs
cd TileOPs
pip install -e '.[dev]' -c constraints.txt
pre-commit install
```

If CANN and TileLang are already installed system-wide and the build fails while re-resolving them, skip build isolation:

```bash
PIP_NO_BUILD_ISOLATION=1 pip install -e '.[dev]' -c constraints.txt
```

`[dev]` adds ruff, codespell, pytest, pytest-xdist, and pyyaml. `[bench]` adds the baseline libraries the benchmarks compare against — see [Benchmarks](#benchmarks).

## Environment

There is no container: an Ascend NPU is a host device, and CANN is installed system-wide.
Work in a virtualenv (or the shared conda env the benchmark machine uses) against the
already-installed toolkit:

| | |
|---|---|
| Device | Ascend 910B1 (Atlas A2 training series) |
| Toolkit | CANN 8.5.0 (`$ASCEND_HOME_PATH`) |
| PyTorch | 2.7.1 with the matching `torch_npu` |
| Compiler | TileLang with the Ascend backend, `bishengir-compile` |

```bash
pip install -e . --no-deps          # the environment already carries the pinned stack
python -m pytest -q tests -m smoke
```

`--no-deps` is deliberate: letting pip resolve `torch` here would replace the build that
`torch_npu` was compiled against.

⚠️ **Never `pip install` into a shared environment while a benchmark is running** — a
replaced `torch` mid-run silently changes what the numbers mean. Build your own venv or use
`--target`.

⚠️ `ASCEND_RT_VISIBLE_DEVICES` **renumbers** the devices: after setting it, the visible card
is logical id `0`. Pin one card per concurrent run so two runs do not share a device.

## Tests

Tests are tiered by marker. Pick the tier by how much you need to cover, not by how long you can wait:

| Command                                                   | Covers                                  |
| --------------------------------------------------------- | --------------------------------------- |
| `python -m pytest -q tests -m smoke`                      | Fast critical path. What every PR runs. |
| `python -m pytest -q tests -m "smoke or full"`            | Standard correctness coverage.          |
| `python -m pytest -q tests -m "smoke or full or nightly"` | Exhaustive and long-running cases.      |
| `python -m pytest -q tests`                               | Everything, including unmarked tests.   |

Narrow to one file or case the usual way — `python -m pytest -q tests/ops/test_gemm.py -k tuned`.

Two suites do not need a NPU and are worth running before pushing:

```bash
python -m pytest -q tests/test_validate_manifest.py   # manifest spec validator
python -m pytest -q benchmarks/tests                  # benchmark harness contract
```

The `packaging` marker is separate from the tiers: it is a minimal wheel-install sanity check, one case per op family, run against an installed wheel rather than a source checkout.

## Lint

`pre-commit install` (above) runs the hooks on every commit. To check the whole tree as CI does:

```bash
pre-commit run --all-files
```

## Docstrings

Docstrings are the API reference on
[the docs site](https://yyttt6.github.io/TileOPs.github.io/), and mkdocstrings
renders them **as Markdown** — reStructuredText reaches the page as literal text.

Three docstrings per op, each answering one question:

| Docstring  | Answers             | Sections                                               |
| ---------- | ------------------- | ------------------------------------------------------ |
| the class  | what the op is      | prose: the formula, the shapes, when a kernel rebuilds |
| `__init__` | how to construct it | `Args:`                                                |
| `forward`  | how to call it      | `Args:`, `Returns:`, `Raises:`, `Example:` last        |

Both members need one: the page gives every member an entry, so a missing
docstring publishes a heading with nothing under it. Parameters go on `__init__`,
not in the class's `Args:`, and the example goes last in `forward` — a class
docstring renders above both signatures.

How to write each element:

| Element                | Write                                                              | Not                                                              |
| ---------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------- |
| Code example           | a fenced block inside `Example:`: ```` ```python linenums="1" ```` | `>>>` prompts — `>` is a Markdown blockquote                     |
| Tensor shape           | `$[B \\times M \\times K]$`                                        | `` `[B, M, K]` ``                                                |
| Formula                | `$d_i = a_i \\mathbin{@} b_i$`, or `$$…$$` on its own line         | `.. math::`                                                      |
| Four or more variants  | a table                                                            | an indented bullet list, which Markdown folds into one paragraph |
| Callout                | `!!! note "Title"`, body indented four spaces                      | `.. note::`                                                      |
| Cross-reference        | `` `torch.nn.functional.rms_norm` ``                               | `:func:` and the other roles                                     |
| Identifier, path, flag | inline code                                                        | math                                                             |

Two that bite:

- **Double every backslash.** A docstring is a regular string literal, so
  `\\times` is a tab followed by `imes`. Write `\\\\times`, or make the docstring raw.
- **A shape is a product of dimensions.** `` `(flops, bytes)` `` is a return pair
  and stays code.

[`bmm.py`](../src/tileops/ops/gemm/bmm.py) carries all of it and is the one to copy
from. `scripts/lint/op_docstrings_lint.py` fails a missing docstring,
reStructuredText, or an `Args:` left on a class; it runs as the
`op-docstrings-lint` pre-commit hook and in the `pre-commit` CI job.

## Benchmarks

Benchmarks compare against external baselines, which the `bench` extra pulls in:

```bash
PIP_NO_BUILD_ISOLATION=1 pip install -e '.[dev,bench]' -c constraints.txt
```

`PIP_NO_BUILD_ISOLATION=1` is required here: several baselines build against the installed PyTorch, and an isolated build environment would fetch a different one.

The Ascend baselines are separate checkouts, not pip packages: the handwritten AscendC libraries (`catlass`, `ops-nn`, `ops-math`, `ops-transformer`, `sgl-kernel-npu`) are built in place and reached through a C shim or `torch.ops.npu.*`. A baseline is only usable once its kernel binaries are actually compiled for `ascend910b` — an `aclnn` entry point whose kernel was not built **silently falls back to the CANN built-in**.

```bash
python -m pytest benchmarks/            # all benchmarks
python -m pytest benchmarks/ops/attention/bench_gqa.py -q
```

Benchmark reporting rules and tolerances are in [design/testing.md](design/testing.md).

## Packaging check

To reproduce the wheel checks CI runs before a release:

```bash
pip install build twine
python -m build
twine check dist/*
```

## Working on a change

Design docs and `src/tileops/manifest/` are the authoritative spec — code conforms to the spec, not the other way around. Start from [design/architecture.md](design/architecture.md) for the module map, and [design/ops-design.md](design/ops-design.md) to add an op.
