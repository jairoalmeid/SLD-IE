"""
Cálculo formal de métricas multilabel (Precision, Recall, F1 macro/micro/weighted, Average Precision e PR curves).
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    hamming_loss,
    accuracy_score,
)
from src.sld.models.concept_label import MULTILABEL_CLASSES, CONCEPT_LABEL_SHORT_NAMES
from src.sld.models.classification import EvaluationReport, PerClassMetrics


def compute_multilabel_evaluation(
    model_id: str,
    classifier_type: str,
    y_true_binary: np.ndarray,
    y_probs: np.ndarray,
    y_pred_binary: np.ndarray,
    thresholds: Dict[str, float],
    total_articles: int = 0
) -> EvaluationReport:
    """
    Calcula relatório quantitativo completo de avaliação multilabel.
    - y_true_binary: (N, 5) matriz binária de rótulos reais
    - y_probs: (N, 5) matriz de probabilidades previstas pelo modelo
    - y_pred_binary: (N, 5) matriz de previsões binárias finais (pós-threshold)
    """
    total_paragraphs = y_true_binary.shape[0]

    # Métricas Globais Agregadas
    macro_f1 = float(f1_score(y_true_binary, y_pred_binary, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true_binary, y_pred_binary, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_binary, y_pred_binary, average="weighted", zero_division=0))

    macro_p = float(precision_score(y_true_binary, y_pred_binary, average="macro", zero_division=0))
    micro_p = float(precision_score(y_true_binary, y_pred_binary, average="micro", zero_division=0))

    macro_r = float(recall_score(y_true_binary, y_pred_binary, average="macro", zero_division=0))
    micro_r = float(recall_score(y_true_binary, y_pred_binary, average="micro", zero_division=0))

    hl = float(hamming_loss(y_true_binary, y_pred_binary)) if total_paragraphs > 0 else 0.0
    subset_acc = float(accuracy_score(y_true_binary, y_pred_binary)) if total_paragraphs > 0 else 0.0

    per_class_metrics: Dict[str, PerClassMetrics] = {}
    confusion_matrices: Dict[str, List[List[int]]] = {}

    for idx, class_id in enumerate(MULTILABEL_CLASSES):
        class_name = CONCEPT_LABEL_SHORT_NAMES[class_id]
        th = thresholds.get(class_name, 0.50)

        yt = y_true_binary[:, idx]
        yp_bin = y_pred_binary[:, idx]
        yp_score = y_probs[:, idx]

        p = float(precision_score(yt, yp_bin, zero_division=0))
        r = float(recall_score(yt, yp_bin, zero_division=0))
        f1 = float(f1_score(yt, yp_bin, zero_division=0))

        if np.sum(yt) > 0:
            ap = float(average_precision_score(yt, yp_score))
        else:
            ap = 0.0

        sup = int(np.sum(yt))

        per_class_metrics[class_name] = PerClassMetrics(
            class_name=class_name,
            threshold=th,
            precision=round(p, 4),
            recall=round(r, 4),
            f1=round(f1, 4),
            average_precision=round(ap, 4),
            support=sup,
        )

        cm = confusion_matrix(yt, yp_bin, labels=[0, 1])
        confusion_matrices[class_name] = cm.tolist()

    return EvaluationReport(
        model_id=model_id,
        classifier_type=classifier_type,
        total_articles=total_articles,
        total_paragraphs=total_paragraphs,
        macro_f1=round(macro_f1, 4),
        micro_f1=round(micro_f1, 4),
        weighted_f1=round(weighted_f1, 4),
        macro_precision=round(macro_p, 4),
        micro_precision=round(micro_p, 4),
        macro_recall=round(macro_r, 4),
        micro_recall=round(micro_r, 4),
        hamming_loss=round(hl, 4),
        subset_accuracy=round(subset_acc, 4),
        per_class_metrics=per_class_metrics,
        confusion_matrices=confusion_matrices,
    )


def compute_pr_curves_data(
    y_true_binary: np.ndarray,
    y_probs: np.ndarray
) -> Dict[str, Dict[str, List[float]]]:
    """Gera pontos (precision, recall) para plotagem de curvas Precision-Recall com Plotly."""
    pr_data = {}
    for idx, class_id in enumerate(MULTILABEL_CLASSES):
        class_name = CONCEPT_LABEL_SHORT_NAMES[class_id]
        yt = y_true_binary[:, idx]
        yp = y_probs[:, idx]

        if np.sum(yt) > 0:
            precision_pts, recall_pts, _ = precision_recall_curve(yt, yp)
            pr_data[class_name] = {
                "precision": precision_pts.tolist(),
                "recall": recall_pts.tolist(),
            }
        else:
            pr_data[class_name] = {"precision": [0.0], "recall": [0.0]}

    return pr_data
