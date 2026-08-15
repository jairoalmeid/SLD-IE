"""
Módulo de Índice de Recuperação do Corpus Refinado (FAISS + Parquet).
"""

from src.sld.rag_index.models import (
    RAGIndexConfig,
    RAGIndexManifest,
    IndexStats,
    CorpusDistributionStats,
    CoverageStats,
    RAGQueryResult,
)
from src.sld.rag_index.builder import (
    RAGIndexBuilder,
    compute_corpus_distribution_stats,
    compute_coverage_stats,
)
from src.sld.rag_index.retriever import RAGIndexRetriever

__all__ = [
    "RAGIndexConfig",
    "RAGIndexManifest",
    "IndexStats",
    "CorpusDistributionStats",
    "CoverageStats",
    "RAGQueryResult",
    "RAGIndexBuilder",
    "RAGIndexRetriever",
    "compute_corpus_distribution_stats",
    "compute_coverage_stats",
]
