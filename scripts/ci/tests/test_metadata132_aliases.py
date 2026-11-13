"""Reviewed metadata joins retain every tensor role, scalar and case boundary."""

import copy
import json
from pathlib import Path

import pytest

from scripts.ci.baseline_contracts import comparison_identity as identity

pytestmark = pytest.mark.smoke

FIXTURES = json.loads((Path(__file__).parent / "fixtures/metadata132_cases.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("sample", FIXTURES, ids=lambda s: s["op"] + "/" + s["case"])
def test_all_132_frozen_metadata_pairs(sample):
    old, new = copy.deepcopy(sample["old_params"]), copy.deepcopy(sample["new_params"])
    expected = identity(sample["op"], sample["case"], new)
    assert identity(sample["old_op"], sample["case"], old) == expected
    assert identity(sample["op"], sample["case"] + "-other", new) != expected
    assert old == sample["old_params"] and new == sample["new_params"]


def changed_leaves(value):
    if isinstance(value, dict):
        for key, item in value.items():
            for changed in changed_leaves(item):
                yield {**value, key: changed}
    elif isinstance(value, list):
        for index, item in enumerate(value):
            for changed in changed_leaves(item):
                yield [*value[:index], changed, *value[index + 1 :]]
    elif isinstance(value, bool):
        yield not value
    elif isinstance(value, (int, float)):
        yield value + 1
    elif value is None:
        yield 0.25
    elif isinstance(value, str):
        yield value + "-different"


@pytest.mark.parametrize("sample", FIXTURES, ids=lambda s: s["op"] + "/" + s["case"])
def test_changed_shapes_dtypes_layouts_scales_and_flags_do_not_join(sample):
    expected = identity(sample["old_op"], sample["case"], sample["old_params"])
    for changed in changed_leaves(sample["new_params"]):
        assert identity(sample["op"], sample["case"], changed) != expected, changed


@pytest.mark.parametrize(
    "op,pairs",
    [
        (
            "GQAPagedFwdOp",
            [("k_pages", "k_cache"), ("v_pages", "v_cache"), ("page_table", "block_table")],
        ),
        (
            "MHADecodePagedWithKVCacheFwdOp",
            [("k", "k_cache"), ("v", "v_cache"), ("real_seqlen_kv", "cache_seqlens")],
        ),
        ("FP8LightningIndexerFwdOp", [("cu_seqlen_ks", "starts"), ("cu_seqlen_ke", "ends")]),
        ("BmmFP8FwdOp", [("scale_a", "a_scale"), ("scale_b", "b_scale")]),
    ],
)
def test_duplicate_spellings_are_rejected_even_when_values_agree(op, pairs):
    for old, new in pairs:
        for other in [1, 2]:
            with pytest.raises(ValueError, match="Ambiguous"):
                identity(op, "case", {old: 1, new: other})


def test_gqa_omitted_validation_only_means_false():
    old = {"k_pages": [[8, 16, 4, 64], "float16"], "page_table": [[2, 4], "int32"]}
    current = {
        "k_cache": old["k_pages"],
        "block_table": old["page_table"],
        "validate_inputs": False,
    }
    assert identity("GQAPagedFwdOp", "case", old) == identity("GQAPagedFwdOp", "case", current)
    for value in [True, None, 0]:
        assert identity("GQAPagedFwdOp", "case", old) != identity(
            "GQAPagedFwdOp", "case", {**current, "validate_inputs": value}
        )


def test_no_fuzzy_names_or_cross_operator_tensor_aliases():
    p = {"k": [[16, 4, 64], "float16"]}
    assert identity("UnreviewedOp", "case", p) != identity(
        "UnreviewedOp", "case", {"k_cache": p["k"]}
    )
    assert identity("BMMFP8FWDOP", "case", p) != identity("BMMFP8FwdOp", "case", p)
    assert identity("DSAPagedFwdOp", "case", p) != identity("DSADenseFwdOp", "case", p)
    assert identity("GQAPagedFwdOp", "case", p) != identity("MHAPagedFwdOp", "case", p)
