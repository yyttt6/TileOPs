import copy

import pytest

from scripts.ci.baseline_contracts import comparison_identity as identity


def test_gelu_merge_keeps_modes_distinct_and_preserves_inputs():
    old = {'x': [[2048, 28672], 'float16'], 'output': [[2048, 14336], 'float16']}
    before = copy.deepcopy(old)
    exact = {'input': old['x'], 'output': old['output'], 'approximate': 'none'}
    tanh = {**exact, 'approximate': 'tanh'}
    assert identity('GeluAndMulFwdOp', 'llama-8b-ffn-prefill-float16', old) == identity(
        'GELUAndMulFwdOp', 'llama-8b-ffn-prefill-float16', exact)
    assert identity('GeluTanhAndMulFwdOp', 'llama-8b-ffn-prefill-float16', old) == identity(
        'GELUAndMulFwdOp', 'llama-8b-prefill-tanh-float16', tanh)
    assert identity('GeluAndMulFwdOp', 'case', old) != identity('GELUAndMulFwdOp', 'case', tanh)
    assert old == before
    with pytest.raises(ValueError, match='contradicts'):
        identity('GeluAndMulFwdOp', 'case', {**old, 'approximate': 'tanh'})


def test_grouped_transpose_and_shapes_cannot_be_erased():
    old = {'output': [[4, 8], 'float16'], 'transpose_a': False, 'transpose_b': True}
    new = {'d': [[4, 8], 'float16'], 'trans_a': False, 'trans_b': True}
    key = identity('GroupedGemmFwdOp', 'case', old)
    assert key == identity('GroupedGEMMFwdOp', 'case', new)
    assert key != identity('GroupedGEMMFwdOp', 'case', {**new, 'trans_b': False})
    assert key != identity('GroupedGEMMFwdOp', 'case', {**new, 'd': [[4, 9], 'float16']})


def test_fp8_scale_and_codes_keep_their_roles():
    old = {'input_tensor': [[1, 8, 1, 64], 'float16'],
           'output_tensor': [[1, 8, 1, 64], 'float8_e4m3fn'],
           'scale_tensor': [[1, 8, 1], 'float32']}
    new = {'x': old['input_tensor'], 'q': old['output_tensor'], 'scale': old['scale_tensor']}
    key = identity('FP8QuantFwdOp', 'case', old)
    assert key == identity('FP8QuantPerTokenGroupFwdOp', 'case', new)
    assert key != identity('FP8QuantPerTokenGroupFwdOp', 'case',
                           {**new, 'q': new['scale'], 'scale': new['q']})


def test_no_fuzzy_alias_or_conflicting_parameters():
    assert identity('GemmFwdOp', 'case', {'n': 4}) == identity('GEMMFwdOp', 'case', {'n': 4})
    assert identity('UnreviewedOp', 'case', {'n': 4}) != identity('UNREVIEWEDOp', 'case', {'n': 4})
    with pytest.raises(ValueError, match='Ambiguous'):
        identity('GroupedGemmFwdOp', 'case', {'output': 1, 'd': 1})


def test_current_alias_precedes_historical_exact_provider():
    from scripts.ci.pr_performance import matching_baselines

    def row(op, measurement):
        return dict(op=op, case='same', library='torch-eager',
                    workload={'params': {'n': 64}}, measurement_id=measurement)

    current = row('GemmFwdOp', 'current')
    history = row('GEMMFwdOp', 'historical')
    assert matching_baselines({'rows': [current], 'historical_rows': [history]},
                              'GEMMFwdOp', 'same', {'n': 64}) == [current]
