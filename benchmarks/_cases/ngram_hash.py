"""Benchmark case factories for ngram_hash."""

from benchmarks._cases import Entry
from benchmarks.baselines import private_inputs
from workloads.sequence_modeling.ngram_hash import NGramHashWorkloadCall, ngram_hash_result

ENTRIES = {}


def _bind(op, case):
    return private_inputs(lambda *args: ngram_hash_result(op, *args), case.inputs, 1)


ENTRIES.update(
    {
        "NGramHashFwdOp": Entry(NGramHashWorkloadCall, binder=_bind),
    }
)
