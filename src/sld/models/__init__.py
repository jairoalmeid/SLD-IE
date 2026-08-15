"""
Modelos de dados do SLD.
"""
from .article import ArticleMetadata, ProcessedArticle
from .search_result import Segment, SearchResult
from .evaluation import GoldStandardAnnotation, ThresholdMetrics, EvaluationSummary, IRRankingMetrics
from .experiment import EnvironmentMetadata, ExperimentConfig, ExperimentManifest
from .concept_label import (
    ConceptLabel,
    CONCEPT_LABEL_NAMES,
    CONCEPT_LABEL_SHORT_NAMES,
    MULTILABEL_CLASSES,
    MULTILABEL_CLASS_NAMES,
    validate_and_sanitize_labels,
    labels_to_binary_vector,
    binary_vector_to_labels,
)
from .classification import (
    MultilabelAnnotation,
    ParagraphRecord,
    ModelVersionMetadata,
    PerClassMetrics,
    EvaluationReport,
)

__all__ = [
    "ArticleMetadata",
    "ProcessedArticle",
    "Segment",
    "SearchResult",
    "GoldStandardAnnotation",
    "ThresholdMetrics",
    "EvaluationSummary",
    "IRRankingMetrics",
    "EnvironmentMetadata",
    "ExperimentConfig",
    "ExperimentManifest",
    "ConceptLabel",
    "CONCEPT_LABEL_NAMES",
    "CONCEPT_LABEL_SHORT_NAMES",
    "MULTILABEL_CLASSES",
    "MULTILABEL_CLASS_NAMES",
    "validate_and_sanitize_labels",
    "labels_to_binary_vector",
    "binary_vector_to_labels",
    "MultilabelAnnotation",
    "ParagraphRecord",
    "ModelVersionMetadata",
    "PerClassMetrics",
    "EvaluationReport",
]
