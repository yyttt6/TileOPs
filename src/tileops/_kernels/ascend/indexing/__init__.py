"""Ascend implementations for indexing operators."""

from tileops._kernels.ascend.indexing.concat import AscendConcatKernel
from tileops._kernels.ascend.indexing.embedding import AscendEmbeddingKernel
from tileops._kernels.ascend.indexing.masked_scatter import AscendMaskedScatterKernel

__all__ = ["AscendConcatKernel", "AscendEmbeddingKernel", "AscendMaskedScatterKernel"]
