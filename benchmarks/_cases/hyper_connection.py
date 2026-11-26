"""Benchmark case factories for hyper_connection."""

from benchmarks._cases import Entry
from workloads.sequence_modeling.hyper_connection import (
    HyperConnectionHeadWorkload,
    HyperConnectionPostWorkload,
    HyperConnectionPreCall,
)

ENTRIES = {}

ENTRIES.update(
    {
        "HyperConnectionHeadFwdOp": Entry(HyperConnectionHeadWorkload.from_call),
        "HyperConnectionPostFwdOp": Entry(HyperConnectionPostWorkload.from_call),
        "HyperConnectionPreFwdOp": Entry(HyperConnectionPreCall),
    }
)
