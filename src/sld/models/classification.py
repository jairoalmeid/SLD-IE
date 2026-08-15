"""
Modelos de dados para registros de parágrafos, anotações multilabel, metadados de modelos e relatórios de avaliação.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
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
    """Métricas quantitativas para uma classe individual."""
    class_name: str
    threshold: float
    precision: float
    recall: float
    f1: float
    average_precision: float
    support: int


class EvaluationReport(BaseModel):
    """Relatório comparativo de avaliação de desempenho dos modelos no test set."""
    model_id: str
    classifier_type: str
    total_articles: int
    total_paragraphs: int
    macro_f1: float
    micro_f1: float
    weighted_f1: float
    macro_precision: float = 0.0
    micro_precision: float = 0.0
    macro_recall: float = 0.0
    micro_recall: float = 0.0
    hamming_loss: float = 0.0
    subset_accuracy: float = 0.0
    per_class_metrics: Dict[str, PerClassMetrics]
    confusion_matrices: Dict[str, List[List[int]]] = Field(default_factory=dict)
    evaluation_date: str = Field(default_factory=lambda: datetime.now().isoformat())
