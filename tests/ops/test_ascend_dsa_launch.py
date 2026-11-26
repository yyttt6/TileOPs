"""Host-side Ascend DSA launch planning and device capacity tests."""

import pytest
import torch


@pytest.mark.smoke
def test_ascend_dsa_dense_schedule_accepts_variable_shapes():
    from types import SimpleNamespace

    pytest.importorskip("tilelang.ascend.language")
    from tileops._kernels.ascend.attention.dsa_dense import uses_d512_schedule

    call = SimpleNamespace(
        kv_group=1, kv_stride=1, is_causal=True, dim=512, tail_dim=0, q_start_index_s=3000
    )
    q = torch.empty((2, 17, 128, 512), device="meta", dtype=torch.bfloat16)
    kv = torch.empty((2, 3072, 1, 512), device="meta", dtype=torch.bfloat16)
    idx = torch.empty((2, 17, 1, 384), device="meta", dtype=torch.int32)
    assert uses_d512_schedule(call, q, kv, idx)
    assert not uses_d512_schedule(call, q, kv, idx[..., :383])
    call.kv_stride = 2
    assert not uses_d512_schedule(call, q, kv, idx)


@pytest.mark.smoke
def test_ascend_dsa_schedules_respect_device_core_limits(monkeypatch):
    from types import SimpleNamespace

    pytest.importorskip("tilelang.ascend.language")
    from tileops._kernels.ascend.attention.dsa_dense import device_core_counts

    npu = SimpleNamespace(
        get_device_properties=lambda device: SimpleNamespace(cube_core_num=36, vector_core_num=72),
        get_device_limit=lambda device: {"cube_core_num": 32, "vector_core_num": 64},
    )
    # Restore the runtime before the suite's teardown hook accesses its allocator.
    with monkeypatch.context() as runtime_patch:
        runtime_patch.setattr(torch, "npu", npu, raising=False)
        assert device_core_counts(SimpleNamespace(index=1)) == (32, 64)
        del npu.get_device_limit
        assert device_core_counts(SimpleNamespace(index=1)) == (36, 72)


@pytest.mark.smoke
@pytest.mark.parametrize(("queries", "expected_28", "expected_32"), [(64, 2, 2), (32, 4, 2)])
def test_ascend_dsa_paged_split_schedule_uses_cube_count(queries, expected_28, expected_32):
    pytest.importorskip("tilelang.ascend.language")
    from tileops._kernels.ascend.attention.dsa_paged import choose_split_count

    assert choose_split_count(queries, 2048, 28, 512) == expected_28
    assert choose_split_count(queries, 2048, 32, 512) == expected_32
    assert choose_split_count(queries, 384, 32, 512) == 0


@pytest.mark.smoke
@pytest.mark.parametrize(("queries", "aivs"), [(48, 56), (48, 64), (47, 64), (65, 56)])
def test_ascend_dsa_paged_gather_schedule_covers_live_tiles(queries, aivs):
    pytest.importorskip("tilelang.ascend.language")
    from tileops._kernels.ascend.attention.dsa_paged_h64 import gather_schedule

    tiles = 64
    blocks, base, shared = gather_schedule(queries, tiles, aivs)
    assert 0 < blocks <= aivs
    if not shared:
        assert sorted(
            row for core in range(blocks) for row in range(core, queries, blocks)
        ) == list(range(queries))
        return

    helpers = blocks - queries
    rows = [
        row
        for core in range(helpers)
        for row in range(core * queries // helpers, (core + 1) * queries // helpers)
    ]
    assert rows == list(range(queries))
    for live in range(tiles + 1):
        primary = list(range(min(live, base))) + list(range(base + shared, live))
        helper = list(range(base, min(live, base + shared)))
        assert sorted(primary + helper) == list(range(live))


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("getter", "message"),
    [
        (0, "not callable"),
        (lambda device: [], "invalid Ascend device limits"),
        (lambda device: {"cube_core_num": 0}, "invalid Ascend cube_core_num"),
    ],
    ids=["non-callable", "non-dict", "zero-limit"],
)
def test_ascend_dsa_rejects_invalid_reported_limits(monkeypatch, getter, message):
    from types import SimpleNamespace

    pytest.importorskip("tilelang.ascend.language")
    from tileops._kernels.ascend.attention.dsa_dense import device_core_counts

    npu = SimpleNamespace(
        get_device_properties=lambda device: SimpleNamespace(cube_core_num=36, vector_core_num=72),
        get_device_limit=getter,
    )
    with monkeypatch.context() as runtime_patch:
        runtime_patch.setattr(torch, "npu", npu, raising=False)
        with pytest.raises(RuntimeError, match=message):
            device_core_counts(SimpleNamespace(index=1))
