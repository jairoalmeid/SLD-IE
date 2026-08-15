"""
Modelos de dados para experimentos de recuperação semântica, reprodutibilidade e manifestos.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class EnvironmentMetadata:
    """Metadados do ambiente de execução e versões de bibliotecas para auditabilidade."""
    python_version: str
    torch_version: str
    transformers_version: str
    sentence_transformers_version: str
    device: str
    cpu_info: str = ""
    cuda_device_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
            "sentence_transformers_version": self.sentence_transformers_version,
            "device": self.device,
            "cpu_info": self.cpu_info,
            "cuda_device_name": self.cuda_device_name,
        }


@dataclass
class ExperimentConfig:
    """Configuração reproduzível de um experimento de recuperação semântica."""
    run_id: str
    embedding_model: str
    model_revision: str = "main"
    similarity_metric: str = "cosine"
    normalize_embeddings: bool = True
    aggregation_strategy: str = "maximum"  # "maximum", "mean", "weighted_mean", "centroid"
    threshold: float = 0.50
    threshold_type: str = "exploratory"  # "exploratory" ou "calibrated"
    minimum_recall_target: float = 0.90
    batch_size: int = 32
    min_words: int = 10
    min_characters: int = 50
    max_characters: int = 1500
    long_text_strategy: str = "chunk"  # "chunk", "truncate", "skip"
    chunk_overlap: int = 100
    chunk_aggregation: str = "maximum"  # "maximum" ou "mean"
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "embedding_model": self.embedding_model,
            "model_revision": self.model_revision,
            "similarity_metric": self.similarity_metric,
            "normalize_embeddings": self.normalize_embeddings,
            "aggregation_strategy": self.aggregation_strategy,
            "threshold": round(self.threshold, 4),
            "threshold_type": self.threshold_type,
            "minimum_recall_target": round(self.minimum_recall_target, 4),
            "batch_size": self.batch_size,
            "min_words": self.min_words,
            "min_characters": self.min_characters,
            "max_characters": self.max_characters,
            "long_text_strategy": self.long_text_strategy,
            "chunk_overlap": self.chunk_overlap,
            "chunk_aggregation": self.chunk_aggregation,
            "random_seed": self.random_seed,
        }


@dataclass
class ExperimentManifest:
    """Manifesto completo e auditável de uma execução experimental."""
    run_id: str
    created_at: str
    config: ExperimentConfig
    environment: EnvironmentMetadata
    semantic_references: List[Dict[str, Any]]
    total_articles: int
    total_paragraphs: int
    valid_paragraphs: int
    excluded_paragraphs: int
    selected_paragraphs: int
    exclusion_reasons: Dict[str, int] = field(default_factory=dict)
    processing_time_seconds: float = 0.0
    embedding_time_seconds: float = 0.0
    similarity_time_seconds: float = 0.0
    paragraphs_per_second: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "config": self.config.to_dict(),
            "environment": self.environment.to_dict(),
            "semantic_references": self.semantic_references,
            "total_articles": self.total_articles,
            "total_paragraphs": self.total_paragraphs,
            "valid_paragraphs": self.valid_paragraphs,
            "excluded_paragraphs": self.excluded_paragraphs,
            "selected_paragraphs": self.selected_paragraphs,
            "exclusion_reasons": self.exclusion_reasons,
            "performance": {
                "processing_time_seconds": round(self.processing_time_seconds, 3),
                "embedding_time_seconds": round(self.embedding_time_seconds, 3),
                "similarity_time_seconds": round(self.similarity_time_seconds, 3),
                "paragraphs_per_second": round(self.paragraphs_per_second, 2),
            }
        }
