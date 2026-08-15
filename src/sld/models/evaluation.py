"""
Modelos de dados para avaliação quantitativa, calibração e métricas de recuperação semântica.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class GoldStandardAnnotation:
    """Anotação humana de relevância para um parágrafo."""
    paragraph_id: str
    human_label: int  # 1 = Relevante, 0 = Não Relevante
    annotator: str = "pesquisador"
    annotation_date: str = ""
    notes: Optional[str] = None
    article_id: str = ""
    text_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paragraph_id": self.paragraph_id,
            "human_label": self.human_label,
            "annotator": self.annotator,
            "annotation_date": self.annotation_date,
            "notes": self.notes or "",
            "article_id": self.article_id,
            "text_sha256": self.text_sha256,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoldStandardAnnotation":
        return cls(
            paragraph_id=data["paragraph_id"],
            human_label=int(data["human_label"]),
            annotator=data.get("annotator", "pesquisador"),
            annotation_date=data.get("annotation_date", ""),
            notes=data.get("notes"),
            article_id=data.get("article_id", ""),
            text_sha256=data.get("text_sha256", ""),
        )


@dataclass
class ThresholdMetrics:
    """Métricas de classificação calculadas para um determinado threshold semântico."""
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1_score: float
    specificity: float
    accuracy: float
    total_retrieved: int
    pct_corpus: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": round(self.threshold, 4),
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "specificity": round(self.specificity, 4),
            "accuracy": round(self.accuracy, 4),
            "total_retrieved": self.total_retrieved,
            "pct_corpus": round(self.pct_corpus, 4),
        }


@dataclass
class IRRankingMetrics:
    """Métricas de avaliação de ranking como sistema de Information Retrieval (IR)."""
    k: int
    precision_at_k: float
    recall_at_k: float
    mrr: float  # Mean Reciprocal Rank
    map_score: float  # Mean Average Precision
    ndcg_at_k: float  # Normalized Discounted Cumulative Gain

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "precision_at_k": round(self.precision_at_k, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "map_score": round(self.map_score, 4),
            "ndcg_at_k": round(self.ndcg_at_k, 4),
        }


@dataclass
class EvaluationSummary:
    """Resumo consolidado da calibração e validação metodológica."""
    calibrated_threshold: float
    calibration_criterion: str
    minimum_recall_target: float
    achieved_recall: float
    achieved_precision: float
    achieved_f1: float
    total_gold_standard_samples: int
    total_relevant_samples: int
    metrics_per_threshold: List[ThresholdMetrics] = field(default_factory=list)
    ranking_metrics: Optional[IRRankingMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibrated_threshold": round(self.calibrated_threshold, 4),
            "calibration_criterion": self.calibration_criterion,
            "minimum_recall_target": round(self.minimum_recall_target, 4),
            "achieved_recall": round(self.achieved_recall, 4),
            "achieved_precision": round(self.achieved_precision, 4),
            "achieved_f1": round(self.achieved_f1, 4),
            "total_gold_standard_samples": self.total_gold_standard_samples,
            "total_relevant_samples": self.total_relevant_samples,
            "metrics_per_threshold": [m.to_dict() for m in self.metrics_per_threshold],
            "ranking_metrics": self.ranking_metrics.to_dict() if self.ranking_metrics else None,
        }
