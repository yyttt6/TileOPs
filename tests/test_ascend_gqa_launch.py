"""Ascend GQA launch planning and dynamic-shape reuse."""

import dataclasses
import math
from types import SimpleNamespace

import pytest
import torch

from tileops._kernels.ascend.attention import gqa_dense_prefill as dense

pytestmark = pytest.mark.smoke


@dataclasses.dataclass(frozen=True)
class LaunchCall:
    batch: int = 1
    heads: int = 32
    heads_kv: int = 4
    dim: int = 128
    max_seqlen_q: int = 1
    seqlen_kv: int = 8192
    dtype: torch.dtype = torch.float16
    is_causal: bool = True
    sm_scale: float | None = None
    softcap: float = 0.0
    window_size_left: int = -1
    window_size_right: int = -1
    has_sinks: bool = False
    is_fp8: bool = False
    fuse_rope: bool = False
    rotary_dim: int = 64
    rope_layout: str = "neox"
    device: object = None


@pytest.mark.parametrize("cores", [1, 2, 8, 23, 24, 32, 36, 48, 64])
@pytest.mark.parametrize(
    "call",
    [
        LaunchCall(),
        LaunchCall(seqlen_kv=4096),
        LaunchCall(seqlen_kv=262144),
        LaunchCall(batch=32, heads_kv=8, seqlen_kv=4096),
        LaunchCall(batch=16, heads=48, heads_kv=1),
        LaunchCall(heads=8, heads_kv=1, dim=256),
        LaunchCall(batch=16, heads=8, heads_kv=1, dim=256),
    ],
)
def test_decode_capacity_and_partition_coverage(cores, call):
    variant, plan = dense.select_kernel(call, cores)
    assert variant == "decode-multibuffer"
    grid, splits = plan["cores"], plan["splits"]
    assert 0 < grid <= cores
    tasks = (
        call.batch
        * call.heads_kv
        * math.ceil((call.heads // call.heads_kv) / plan["query_block"])
        * splits
    )
    assigned = [task for block in range(grid) for task in range(block, tasks, grid)]
    assert sorted(assigned) == list(range(tasks))
    tiles = call.seqlen_kv // plan["kv_block"]
    partitions = [range(s * tiles // splits, (s + 1) * tiles // splits) for s in range(splits)]
    assert all(partitions)
    assert [tile for part in partitions for tile in part] == list(range(tiles))
    assert max(map(len, partitions)) - min(map(len, partitions)) <= 1
    assert plan.get("grid_fuse", False) == (splits > 1)
    if splits > 1:
        assert grid == cores
    else:
        assert plan["fuse_output"]


def test_sink_decode_keeps_each_head_and_its_sink_in_one_tile():
    call = LaunchCall(batch=64, heads=64, heads_kv=8, dim=64, dtype=torch.bfloat16, has_sinks=True)
    variant, plan = dense.select_kernel(call, 32)
    assert variant == "decode-multibuffer"
    assert plan["fuse_output"] and plan["splits"] == 1 and plan["cores"] == 32
    for overrides in ({"window_size_left": 127}, {"softcap": 2.0}, {"batch": 1}):
        assert (
            dense.select_kernel(dataclasses.replace(call, **overrides), 32)[0] == "packed-attention"
        )


def test_scaled_fp8_sink_prefill_tails():
    call = LaunchCall(
        max_seqlen_q=255, seqlen_kv=1793, heads=8, heads_kv=2, is_fp8=True, has_sinks=True
    )
    assert dense.select_kernel(call, 32) == (
        "prefill-preprocess",
        dict(num_cores=32, query_block=64, l0_k=128),
    )
    for overrides in ({"fuse_rope": True}, {"window_size_left": 127}, {"has_sinks": False}):
        assert (
            dense.select_kernel(dataclasses.replace(call, **overrides), 32)[0] == "packed-attention"
        )


def test_packed_entry_reuses_dynamic_shapes(monkeypatch):
    from tileops._kernels.attention.call_spec import AttentionCall

    monkeypatch.setattr(dense, "device_cube_cores", lambda _: 32)

    call = AttentionCall(
        batch=1,
        heads=8,
        heads_kv=2,
        dim=128,
        dtype=torch.float16,
        is_causal=True,
        max_seqlen_q=7,
        seqlen_kv=131,
        has_sinks=True,
    )
    identity = dense.AscendGQADenseKernel.entry_for(call)[0]
    for batch, sq, sk in ((2, 15, 255), (3, 17, 257)):
        resized = dataclasses.replace(
            call, batch=batch, max_seqlen_q=sq, seqlen_kv=sk, max_position=sk + 1
        )
        assert dense.AscendGQADenseKernel.entry_for(resized)[0] == identity
    for overrides in ({"window_size_left": 15}, {"dim": 64}, {"has_sinks": False}):
        changed = dataclasses.replace(call, **overrides)
        assert dense.AscendGQADenseKernel.entry_for(changed)[0] != identity


@pytest.mark.parametrize("axis", ["batch", "max_seqlen_q", "heads"])
def test_empty_dense_output_requires_no_compilation(monkeypatch, axis):

    call = dataclasses.replace(LaunchCall(is_fp8=True), **{axis: 0})
    monkeypatch.setattr(dense, "device_cube_cores", lambda _: 32)
    monkeypatch.setattr(dense.AscendGQADenseKernel, "_require_devices", lambda *a, **kw: None)
    monkeypatch.setattr(dense.tilelang, "compile", lambda *a, **kw: pytest.fail("empty launch"))
    kernel = dense.AscendGQADenseKernel(call)
    q = torch.empty(call.batch, call.max_seqlen_q, call.heads, call.dim, dtype=torch.float8_e4m3fn)
    k = torch.empty(call.batch, call.seqlen_kv, call.heads_kv, call.dim, dtype=q.dtype)
    o = kernel.forward(q, k, k)
    assert o.shape == q.shape and o.dtype == call.dtype


def test_tuned_rope_reuses_kernel_with_larger_tables(monkeypatch):
    from tileops._kernels.attention.call_spec import AttentionCall

    call = AttentionCall(
        batch=1,
        heads=8,
        heads_kv=2,
        dim=128,
        dtype=torch.float16,
        max_seqlen_q=256,
        seqlen_kv=1792,
        is_causal=True,
        fuse_rope=True,
        rotary_dim=64,
        max_position=1792,
    )
    monkeypatch.setattr(dense, "device_cube_cores", lambda _: 32)
    monkeypatch.setattr(dense.AscendGQADenseKernel, "_require_devices", lambda *a, **kw: None)
    observed = []

    def record(*args):
        observed.append(args[-2:])

    monkeypatch.setattr(dense.tilelang, "compile", lambda *a, **kw: record)
    key, build = dense.AscendGQADenseKernel.entry_for(call)
    kernel = build()
    assert kernel.config["variant"] == "prefill-preprocess"
    q = torch.empty(1, 256, 8, 128, dtype=call.dtype)
    k = torch.empty(1, 1792, 2, 128, dtype=call.dtype)
    for capacity in (1792, 4096):
        assert (
            dense.AscendGQADenseKernel.entry_for(dataclasses.replace(call, max_position=capacity))[
                0
            ]
            == key
        )
        cos, sin = (torch.empty(capacity, 32, dtype=call.dtype) for _ in range(2))
        kernel.forward(q, k, k, rope_cos=cos, rope_sin=sin)
        for view, table in zip(observed[-1], (cos, sin), strict=True):
            assert view.shape == (1792, 32) and view.data_ptr() == table.data_ptr()


def test_packed_forward_reuses_kernel_across_requests():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")

    from tileops.attention import GQADenseFwdOp
    from workloads.attention.gqa.dense import dense_gqa_ref

    op = GQADenseFwdOp(is_causal=True)
    generator = torch.Generator().manual_seed(1027)
    sinks = torch.linspace(-2, 4, 4)
    for batch, sq, sk in ((1, 3, 7), (2, 17, 131)):
        q, k, v = (
            torch.randn(batch, length, heads, 64, dtype=torch.float16, generator=generator)
            for length, heads in ((sq, 4), (sk, 2), (sk, 2))
        )
        expected = dense_gqa_ref(q, k, v, heads=4, heads_kv=2, is_causal=True, sinks=sinks)
        actual = op(q.npu(), k.npu(), v.npu(), sinks=sinks.npu())
        torch.testing.assert_close(actual.cpu(), expected, atol=0.002, rtol=0.002)
    (kernel,) = op.iter_kernels()
    assert kernel.config["variant"] == "packed-attention"


def test_split_selection_tracks_capacity_and_cache_key():
    call = LaunchCall()
    assert [dense.select_kernel(call, c)[1]["splits"] for c in (24, 32, 36, 48)] == [
        6,
        8,
        9,
        12,
    ]
    assert dense.configuration(call, 8)[2] == 4
    assert dense.configuration(call, 48)[2] == 24
    assert dense.configuration(call, 8)[2] == 4


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, "prefill-query-major"),
        ({"window_size_left": 127, "dtype": torch.bfloat16}, "prefill-query-major"),
        ({"window_size_left": 128}, "packed-attention"),
        ({"max_seqlen_q": 1}, "packed-attention"),
        ({"seqlen_kv": 255}, "packed-attention"),
        ({"is_fp8": True}, "packed-attention"),
    ],
)
def test_sink_prefill_dispatch(overrides, expected):
    call = dataclasses.replace(
        LaunchCall(max_seqlen_q=128, seqlen_kv=256, dim=64, has_sinks=True), **overrides
    )
    variant, plan = dense.select_kernel(call, 32)
    assert variant == expected
    if variant == "prefill-query-major":
        assert plan["num_cores"] == 32


@pytest.mark.parametrize("cores", [8, 23, 36, 48])
@pytest.mark.parametrize(
    "call",
    [
        LaunchCall(heads=8, heads_kv=2, max_seqlen_q=256, seqlen_kv=1792, is_fp8=True),
        LaunchCall(heads=8, heads_kv=2, max_seqlen_q=256, seqlen_kv=1792, fuse_rope=True),
        LaunchCall(heads_kv=8, max_seqlen_q=128, seqlen_kv=2048, dtype=torch.bfloat16),
    ],
)
def test_prefill_receives_device_capacity(cores, call):
    _, plan = dense.select_kernel(call, cores)
    assert plan["num_cores"] == cores


def test_device_query_uses_operand_device(monkeypatch):
    seen = []

    def properties(device):
        seen.append(device)
        assert type(device) is int
        return SimpleNamespace(cube_core_num=24 if device == 1 else 48)

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(current_device=lambda: 0, get_device_properties=properties),
        raising=False,
    )
    query = dense.device_cube_cores
    assert query(LaunchCall(device=torch.device("privateuseone:1"))) == 24
    assert query(LaunchCall()) == 48
    assert seen == [1, 0]


@pytest.mark.parametrize("cores", [0, -1])
def test_invalid_capacity_fails_before_launch(cores, monkeypatch):
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_device=lambda: 0,
            get_device_properties=lambda _: SimpleNamespace(cube_core_num=cores),
        ),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="positive"):
        dense.device_cube_cores(LaunchCall())
    with pytest.raises(ValueError, match="positive"):
        dense.select_kernel(LaunchCall(), cores)


@pytest.mark.parametrize("limit,expected", [(32, 32), (48, 36)])
@pytest.mark.parametrize(
    "device,index",
    [
        (None, 0),
        (0, 0),
        (1, 1),
        (torch.device("privateuseone"), 0),
        (torch.device("privateuseone:1"), 1),
    ],
)
def test_cube_capacity_respects_runtime_limit(monkeypatch, limit, expected, device, index):
    queries = []

    def get_limit(device_index):
        # torch_npu.get_device_limit rejects torch.device, unlike properties.
        assert type(device_index) is int
        queries.append(device_index)
        return {"cube_core_num": limit}

    npu = SimpleNamespace(
        current_device=lambda: 0,
        get_device_properties=lambda device: SimpleNamespace(cube_core_num=36),
        get_device_limit=get_limit,
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    assert dense.device_cube_cores(LaunchCall(device=device)) == expected
    assert queries == [index]


@pytest.mark.parametrize("limit", [0, -1, None, True, "32"])
def test_invalid_cube_limit_is_not_ignored(monkeypatch, limit):
    npu = SimpleNamespace(
        current_device=lambda: 0,
        get_device_properties=lambda device: SimpleNamespace(cube_core_num=36),
        get_device_limit=lambda device: {"cube_core_num": limit},
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    with pytest.raises(RuntimeError, match="invalid cube_core_num"):
        dense.device_cube_cores(LaunchCall())


@pytest.mark.parametrize("seq_len", [4096, 8192, 32768])
def test_decode_constructor_publishes_final_config(monkeypatch, seq_len):

    def get_limit(device_index):
        assert type(device_index) is int
        return {"cube_core_num": 32}

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_device=lambda: 0,
            get_device_properties=lambda _: SimpleNamespace(cube_core_num=36),
            get_device_limit=get_limit,
        ),
        raising=False,
    )
    compiled = []

    def compile_program(program, **kwargs):
        compiled.append((program, kwargs))
        return object()

    monkeypatch.setattr(dense.tilelang, "compile", compile_program)
    kernel = dense.AscendGQADenseKernel(LaunchCall(device=torch.device("cpu:0"), seqlen_kv=seq_len))
    assert len(compiled) == 1
    assert kernel.config["cube_cores"] == 32
    assert kernel.config["kv_splits"] == kernel.config["splits"]
    assert kernel.config["multibuffer"] and kernel.config["balanced_vectors"]
    with pytest.raises(TypeError, match="published kernel config"):
        kernel.config["kv_splits"] = 1
