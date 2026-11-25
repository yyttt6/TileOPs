"""Contracts imported from the typed upstream manifest revision."""

TYPED_OPS = frozenset(
    {
        "ConcatFwdOp",
        "EmbeddingFwdOp",
        "FP4QuantPerGroupFwdOp",
        "GatedPoolCompressFwdOp",
        "HyperConnectionHeadFwdOp",
        "HyperConnectionPostFwdOp",
        "HyperConnectionPreFwdOp",
        "MaskedScatterFwdOp",
        "MoEGroupedGEMMMXFP4FwdOp",
        "NGramHashFwdOp",
        "RMSNormGatedFwdOp",
        "RoPECosSinFwdOp",
    }
)
