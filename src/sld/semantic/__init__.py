"""
Módulo Semântico (Segmentação, Embeddings, Índice Vetorial, Referências Multidimensionais e Busca).
"""

from .segmenter import segment_markdown, parse_markdown_with_yaml
from .embedding_service import EmbeddingService
from .vector_index import VectorIndex
from .semantic_reference import SemanticReferenceSet, SemanticAnchor
from .semantic_search import perform_multi_anchor_search, perform_semantic_search
from .evaluation_service import calibrate_threshold_by_minimum_recall, compute_ir_ranking_metrics, load_gold_standard, save_gold_standard
from .sampling_service import random_sampling, stratified_by_similarity_sampling, boundary_sampling
from .benchmark_service import run_model_benchmark

__all__ = [
    "segment_markdown",
    "parse_markdown_with_yaml",
    "EmbeddingService",
    "VectorIndex",
    "SemanticReferenceSet",
    "SemanticAnchor",
    "perform_multi_anchor_search",
    "perform_semantic_search",
    "calibrate_threshold_by_minimum_recall",
    "compute_ir_ranking_metrics",
    "load_gold_standard",
    "save_gold_standard",
    "random_sampling",
    "stratified_by_similarity_sampling",
    "boundary_sampling",
    "run_model_benchmark",
]
