# Op Interface Design — Reference

The contracts an op is built against: base-class attributes, family protocol variables, naming, parameter design, calling conventions, and what CI enforces.

## Slot Rules

The per-slot codegen rules live with their consumer:
[`.claude/skills/scaffold-op/slot-rules.md`](../../.claude/skills/scaffold-op/slot-rules.md).
This document holds the contracts those rules emit against.

## Family-Base Protocol (Appendix) <a id="base-class-protocol"></a>

Per-family protocol variables, declared by L2 bases and overridden by L3 ops.

| Variable      | Family      | Purpose                                                                                                          |
| ------------- | ----------- | ---------------------------------------------------------------------------------------------------------------- |
| `_kernel_key` | reduction   | The slot name the op asks its target for                                                                         |
| `_op_kind`    | reduction   | Which reduction it is (`"sum"` / `"prod"` for `CumulativeOp`; `"sum"`, `"mean"`, … for `_ReduceOpBase`)          |
| `_op_name`    | elementwise | `torch.library.custom_op` registration key, and the slot name                                                    |

**The `scaffold-op` skill does NOT emit these variables** — a family base reads them, and adding one requires updating that base, every concrete op under it, and the manifest schema if applicable.

### `Op` base class attributes ([`src/tileops/ops/op_base.py`](../../src/tileops/ops/op_base.py))

| Attribute      | Type                                 | Purpose                                                                                      |
| -------------- | ------------------------------------ | -------------------------------------------------------------------------------------------- |
| `target`       | `Target`                             | Which set of kernels serves this instance; `None` decides from the input device              |
| `dtype`        | `Optional[torch.dtype]`              | Dtype of the most recent `forward()`; `None` before the first one                            |
| `device`       | `Optional[Union[torch.device, str]]` | Device (default `'npu'`)                                                                    |
| `input_shapes` | `Optional[list[tuple]]`              | Expected input tensor shapes (for introspection and non-runtime consumers)                   |
| `tune`         | `bool`                               | Whether kernels this op builds tune themselves; read by a factory when it runs               |
| `_static_axes` | `frozenset[tuple[int, int]]`         | Static axes as `(input_index, axis)` pairs (default `frozenset()`); consumed by `_cache_key` |

Abstract interface: `forward()`. Manifest-driven methods (codegen-emitted by concrete ops): `_infer_output_shapes`, `_validate_dtypes`, `eval_roofline`.

#### Kernel caching and enumeration methods

Rationale and the role / entry vocabulary: [ops-design.md § Kernel caching and enumeration](ops-design.md#kernel-caching-and-enumeration).

| Method                                             | Purpose                                                                                                                                                                                                                                                                    |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_or_build_kernel(name, inputs)`                | Return the kernel for this call, asking the target for it once on a miss. The only get-or-build in L1-L3. `inputs` is the tensors the kernel will be handed, one slot per `signature.inputs` entry; the built kernel is keyed on their device, dtype and shape |
| `built_kernels(name)`                              | Read-only view of a name's entries; empty before its first build. Introspection only, never dispatch                                                                                                                                                                       |
| `kernel_delegates()`                               | The ops whose kernels this op runs. Default `()`; a composite op overrides it                                                                                                                                                                                              |
| `iter_kernels()`                                   | Every kernel the op holds, deduplicated: entries and delegates                                                                                                                                                                                                             |
| `autotune()`                                       | Puts the op in tuned mode: tunes built kernels, and sets `tune` so later builds tune too                                                                                                                                                                                   |

### What a kernel is ([`src/tileops/backend/protocol.py`](../../src/tileops/backend/protocol.py))

`Kernel` is a type alias, not a base class: `Callable[..., KernelResult]` — whatever a target's `build_kernel` handed back, called with the tensors the op was handed. There is no required base, no required method, and no attribute this layer reads. A backend is free to return a closure over a compiled artifact, a bound method, or an instance of a class of its own.

Unlike `Op`, a kernel **is** built for one call signature: the op layer keys it on the device, dtype and shape of every input slot, so a second dtype or a second shape asks the target for a second kernel. Everything a kernel specializes on — tiles, pipelining, which schedule to use — is decided inside `build_kernel`, which is the only place that sees both the shapes and the hardware.

`autotune()` on an op sets `tune` and calls `autotune()` on every kernel that has one; a kernel that does not is left alone.

## Optional Hooks (Appendix)

Hooks family bases expose for op-specific semantics. The `scaffold-op` skill does NOT emit these.

| Hook              | Family    | Default                     | Override example                                      |
| ----------------- | --------- | --------------------------- | ----------------------------------------------------- |
| `_validate_dim()` | reduction | accept `int` or `list[int]` | `ArgmaxFwdOp._validate_dim` restricts to scalar `int` |

A hook that compensates for what a kernel cannot do belongs to that kernel, not here: the op hands over the tensor its manifest declares.

### `_cache_key` override (L1-level, not family-specific)

`Op._cache_key(self, *input_shapes) -> Hashable` defaults to projecting non-static axes via `self._static_axes`. Override when the kernel's math permits coarser keying — e.g., RMSNorm only depends on the non-static axis product `M`:

```python
class RMSNormFwdOp(Op):
    def _cache_key(self, x_shape):
        dim = self.dim % len(x_shape)
        return (math.prod(s for i, s in enumerate(x_shape) if i != dim),)
```

**When `_static_axes` is empty, override is mandatory** — the default keys by the full input shape (one kernel compile per distinct shape). The base emits a once-per-type `UserWarning` when invoked with empty `_static_axes` and no subclass override.

## Naming Conventions (Appendix) <a id="naming-conventions"></a>

- **Op class:** `{PascalCaseName}{Direction}Op`. `Direction` ∈ {`Fwd`, `Bwd`}, mandatory. Manifest key must equal `cls.__name__`. Abbreviation casing: `RMSNormFwdOp`, `SSDDecodeFwdOp` — fully uppercase per `.claude/rules/code-style.md`. Slot [S6](#slot-s6).
- **Slot names:** `snake_case`, one per computation the op asks a target for, never one per implementation. Passed to `get_or_build_kernel`; a backend registers against the manifest op name, so the slot name is the op's own vocabulary for its parts (`gqa_bwd_preprocess_kernel` and `gqa_bwd_kernel` are two slots of one op).
- **Builder functions:** `snake_case`, e.g. `def rms_norm_fwd(M, N, dtype, ...): ...`.
- **Filenames:** all-lowercase with underscores. Multi-word abbreviations stay fully lowercase (`rms_norm.py`, `ssd_decode.py`; never `RMSNorm.py` or `Ssd_decode.py`). Norm-related names never contract (`rms_norm`, not `rmsnorm`).

## Codegen Details (Appendix) <a id="codegen"></a>

The manifest ([`src/tileops/manifest/`](../../src/tileops/manifest/)) is the sole source of truth. Dtype validation and shape inference derive from manifest; roofline codegen is defined in [roofline.md](roofline.md).

### Parameter design <a id="parameter-design"></a>

Three time points: (1) manifest — constraint structure; (2) `__init__` — user commits `static_dims` values; (3) `forward` — shapes concrete, commitments validated, dtype read from the tensors. See [manifest.md § `static_dims`](manifest.md#static_dims).

**Dtype belongs to time point 3, never to 2.** The tensors carry it, so requiring the caller to restate it at construction only creates a second source that can disagree with the first. Constructing an op therefore commits to shape structure and nothing about element type.

|                          | Fixed-rank op           | Arbitrary-rank op                                            |
| ------------------------ | ----------------------- | ------------------------------------------------------------ |
| Manifest has `shape`     | yes                     | no                                                           |
| `__init__` shape source  | `shape` dimension names | `static_dims`                                                |
| Undeclared dimensions    | none                    | derived from tensor at forward time                          |
| Kernel construction time | forward (first call)    | forward (first encounter)                                    |
| Forward keying           | dtype                   | opaque to L1; carries every input that changes what is built |

### Calling conventions

- **Fully static op:** `_infer_output_shapes` called once in `__init__`, result stored as an instance attribute.
- **Op with dynamic dims:** `_infer_output_shapes` called once dynamic dims resolve, and by the fake while tracing.
- **Kernel construction:** in `_eager_forward`, through `get_or_build_kernel` — never in the traced `forward`, which is one call to the op's operator ([Compile Dispatch Boundary](ops-design.md#compile-dispatch-boundary)). See [Slot S16](#slot-s16).
- **`_validate_dtypes`:** runs on every call, and is the only place an op rejects a dtype.
- **Non-runtime consumers** (validator, graph compiler): call `_infer_output_shapes` with concrete shape tuples without constructing tensors. Roofline consumers use interfaces in [`roofline.md`](roofline.md).

### Inheritance in family-base hierarchies

| Scenario                                             | Codegen method defined at | Concrete op action    |
| ---------------------------------------------------- | ------------------------- | --------------------- |
| Family shares logic                                  | L2 family base            | Inherits, no override |
| Family member has variant logic (e.g., multi-output) | L3 concrete op            | Overrides             |
| Op inherits L1 directly (T2)                         | L3 concrete op            | Scaffold emits body   |

### Consistency enforcement

| Check                                                    | Mechanism                            |
| -------------------------------------------------------- | ------------------------------------ |
| Manifest schema and declared fields are well-formed      | Validator (CI), L0 checks            |
| `__init__` params match manifest `params`                | Validator signature check (L1)       |
| `static_dims` keys are `__init__` parameters             | Validator signature check (L1)       |
| `shape_rules` syntax is valid                            | Validator `shape_rules` parsing (L2) |
| `_infer_output_shapes` output satisfies `shape_rules`    | Validator infer-shape parity (L2)    |
| `dtype`/`dtype_combos` strings are valid                 | Validator dtype conformance (L3)     |
| `_validate_dtypes` matches `dtype_combos` / dtype unions | Validator dtype parity (L3)          |
| Empty `static_dims` without `_cache_key` override        | `Op` base class runtime warning      |

Checks beyond this table are tracked as separate issues, not as spec status.

**Parity check coverage.** The L2 / L3 parity checks compare the manifest spec against the concrete method the op class defines. When the class has not migrated to the codegen protocol, the validator emits a **warning** naming the missing method — the gap is surfaced, never silently passed. When the method exists, the parity check runs and any disagreement is a hard L2 / L3 error. Ops whose method genuinely cannot be invoked in a CPU-only validator context must declare `status: spec-only`; there is no parity opt-out, and demotion is only legitimate when the implementation truly does not conform.

## Development Path (Appendix) <a id="development-path"></a>

Pragmatic sequence:

1. **New op inherits L1 directly (T2).** When a family has 1-2 ops, the op owns its full `forward()`. Transitional state.
1. **Family accumulates ops.** When 2-3 ops share identical `forward()` flow, extract an L2 family base.
1. **L1-direct and L1→L2→L3 coexist.** L1-direct ops are candidates for future L2 extraction, not an alternative design.

Create an L2 family base when multiple ops share the same `forward()` control flow, the shared boilerplate is substantial, and per-op differences fit into class variables or hooks. Do NOT create one when only 1 op uses the pattern, ops share math but differ in flow, or a common base would need excessive `if/else`.

### Adding a new family base <a id="adding-a-new-family-base"></a>

1. Implement 2-3 concrete T2 ops to understand the pattern before abstracting.
1. Identify shared `forward()` steps.
1. Extract shared steps into the base; lift per-op differences into class variables or overridable hooks (see [Family-Base Protocol (Appendix)](#base-class-protocol) and [Optional Hooks (Appendix)](#optional-hooks-appendix)).
1. Migrate existing ops; verify tests pass unchanged.
1. Register any new protocol variables in the Family-Base Protocol table.
