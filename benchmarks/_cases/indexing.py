"""Benchmark case factories for indexing."""

from benchmarks._cases import Entry
from workloads.indexing import ConcatWorkload, EmbeddingWorkload, MaskedScatterWorkload

ENTRIES = {}

ENTRIES.update(
    {
        "ConcatFwdOp": Entry(ConcatWorkload),
        "EmbeddingFwdOp": Entry(EmbeddingWorkload),
        "MaskedScatterFwdOp": Entry(MaskedScatterWorkload),
    }
)
