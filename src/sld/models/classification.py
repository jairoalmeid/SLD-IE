"""
Modelos de dados para registros de parágrafos, anotações multilabel, metadados de modelos e relatórios de avaliação.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict
from src.sld.models.concept_label import validate_and_sanitize_labels, labels_to_binary_vector, CONCEPT_LABEL_SHORT_NAMES


class AnnotationRecord(BaseModel):
    """Representa o registro estruturado de anotação de um parágrafo por um anotador."""
    annotation_id: str
    dataset_id: str = "ANNOTATION_SET_001"
    run_id: str = ""
    document_id: str = ""
    paragraph_id: str
    annotator_id: str = "ANN_001"
    annotator_name: Optional[str] = None
    annotation_source: str = "internal"  # internal, external_markdown, external_csv, external_jsonl

    label_0: bool = False
    label_1: bool = False
    label_2: bool = False
    label_3: bool = False
    label_4: bool = False
    label_5: bool = False

    annotation_status: str = "unannotated"  # unannotated, valid, invalid, conflict
    annotation_note: Optional[str] = ""
    text_hash: Optional[str] = ""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    imported_at: Optional[str] = None
    included_in_gold_standard: bool = True
    adjudicated: bool = False
    adjudicated_by: Optional[str] = None
    adjudicated_at: Optional[str] = None

    @property
    def labels_list(self) -> List[int]:
        """Retorna a lista de números das classes ativas (0 a 5)."""
        if self.label_0:
            return [0]
        active = []
        if self.label_1: active.append(1)
        if self.label_2: active.append(2)
        if self.label_3: active.append(3)
        if self.label_4: active.append(4)
        if self.label_5: active.append(5)
        return active

    @property
    def labels_binary(self) -> List[int]:
        """Retorna o vetor binário de 5 posições (para classes 1 a 5)."""
        return labels_to_binary_vector(self.labels_list)


class MultilabelAnnotation(BaseModel):
    """Representa uma decisão manual de rotulação (compatibilidade retroativa)."""
    paragraph_id: str
    article_id: str
    text: str
    semantic_score: Optional[float] = None
    labels: List[int] = Field(default_factory=lambda: [0])
    annotation_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    annotator: str = ""
    annotator_id: str = "ANN_001"
    annotation_notes: str = ""
    annotation_version: str = "v1.0"
    text_sha256: str = ""

    def sanitize_labels(self):
        """Aplica a Regra da Classe 0 no próprio objeto."""
        self.labels = validate_and_sanitize_labels(self.labels)

    @property
    def labels_binary(self) -> List[int]:
        """Retorna o vetor binário de 5 posições (para classes 1 a 5)."""
        return labels_to_binary_vector(self.labels)


class ParagraphRecord(BaseModel):
    """Representa a unidade fundamental de análise com proveniência e estado."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    paragraph_id: str
    article_id: str
    section: Optional[str] = ""
    paragraph_index: int = 0
    text: str
    text_sha256: str = ""
    semantic_score: Optional[float] = None
    best_anchor_id: Optional[str] = None
    anchor_scores: Dict[str, float] = Field(default_factory=dict)
    status: str = "RAW"  # RAW, SEMANTIC_CANDIDATE, SEMANTIC_REJECTED, MANUALLY_ANNOTATED, MODEL_RELEVANT, MODEL_NOT_RELEVANT, FINAL_CORPUS
    predicted_probabilities: Dict[str, float] = Field(default_factory=dict)
    predicted_labels: List[str] = Field(default_factory=list)
    embedding: Optional[Any] = Field(default=None, exclude=True)


class ModelVersionMetadata(BaseModel):
    """Snapshot completo de reprodutibilidade para modelos treinados."""
    model_id: str
    classifier_type: str  # "logistic_regression"
    embedding_model: str
    training_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    training_samples: int
    class_distribution: Dict[str, int]
    random_seed: int = 42
    train_articles: List[str]
    validation_articles: List[str]
    test_articles: List[str]
    thresholds: Dict[str, float]
    metrics: Dict[str, Any]
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    library_versions: Dict[str, str] = Field(default_factory=dict)


class PerClassMetrics(BaseModel):
    """Métricas quantitativas e diagnósticas para uma classe individual."""
    class_name: str
    threshold: float
    support_positive: int = 0
    support_negative: int = 0
    support: int = 0  # alias para support_positive
    prevalence: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    precision: Optional[float] = None
    recall: Optional[float] = None
    specificity: Optional[float] = None
    f1: Optional[float] = None
    binary_accuracy: float = 0.0
    balanced_accuracy: Optional[float] = None
    fpr: Optional[float] = None
    fnr: Optional[float] = None
    average_precision: Optional[float] = None
    roc_auc: Optional[float] = None
    f1_ci95: Optional[Tuple[float, float]] = None
    precision_ci95: Optional[Tuple[float, float]] = None
    recall_ci95: Optional[Tuple[float, float]] = None
    is_valid: bool = True
    note: Optional[str] = ""


class EvaluationReport(BaseModel):
    """Relatório comparativo de avaliação de desempenho dos modelos supervisionados."""
    model_id: str
    classifier_type: str
    total_articles: int
    total_paragraphs: int
    active_classes_count: int = 5
    label_cardinality: float = 0.0
    label_density: float = 0.0
    macro_f1: Optional[float] = None
    macro_f1_ci95: Optional[Tuple[float, float]] = None
    micro_f1: Optional[float] = None
    micro_f1_ci95: Optional[Tuple[float, float]] = None
    weighted_f1: Optional[float] = None
    macro_precision: Optional[float] = None
    micro_precision: Optional[float] = None
    macro_recall: Optional[float] = None
    micro_recall: Optional[float] = None
    hamming_loss: float = 0.0
    hamming_loss_ci95: Optional[Tuple[float, float]] = None
    subset_accuracy: float = 0.0
    exact_match_ci95: Optional[Tuple[float, float]] = None
    per_class_metrics: Dict[str, PerClassMetrics]  # Classes ativas 1 a 5
    class_0_metrics: Optional[PerClassMetrics] = None  # Classe 0 isolada
    confusion_matrices: Dict[str, List[List[int]]] = Field(default_factory=dict)
    cv_metrics_table: Optional[List[Dict[str, Any]]] = None
    consistency_checks: Dict[str, Any] = Field(default_factory=dict)
    leakage_warnings: List[str] = Field(default_factory=list)
    methodological_alerts: List[str] = Field(default_factory=list)
    evaluation_date: str = Field(default_factory=lambda: datetime.now().isoformat())
