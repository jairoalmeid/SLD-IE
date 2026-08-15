"""
Otimização empírica de thresholds individuais por classe no conjunto de validação.
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from src.sld.models.concept_label import MULTILABEL_CLASSES, CONCEPT_LABEL_SHORT_NAMES


def optimize_thresholds_per_class(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    step: float = 0.02,
    metric: str = "f1"
) -> Dict[str, float]:
    """
    Varre thresholds de 0.10 a 0.90 para cada uma das 5 classes no conjunto de validação
    e seleciona o threshold que maximiza o F1-score por classe.

    - val_probs: matriz NumPy de formato (N, 5) com probabilidades P(class_i | x).
    - val_targets: matriz binária NumPy de formato (N, 5) com rótulos reais.
    """
    thresholds: Dict[str, float] = {}
    candidate_thresholds = np.arange(0.10, 0.92, step)

    for idx, class_id in enumerate(MULTILABEL_CLASSES):
        class_name = CONCEPT_LABEL_SHORT_NAMES[class_id]
        y_true = val_targets[:, idx]
        y_prob = val_probs[:, idx]

        # Se a classe não tiver instâncias positivas no conjunto de validação, usa 0.50 por padrão
        if np.sum(y_true) == 0:
            thresholds[class_name] = 0.50
            continue

        best_th = 0.50
        best_score = -1.0

        for th in candidate_thresholds:
            y_pred = (y_prob >= th).astype(int)
            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "precision_recall_balance":
                p = precision_score(y_true, y_pred, zero_division=0)
                r = recall_score(y_true, y_pred, zero_division=0)
                score = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            else:
                score = f1_score(y_true, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_th = float(th)

        thresholds[class_name] = round(best_th, 4)

    return thresholds


def apply_class_thresholds(
    probs: np.ndarray,
    thresholds: Dict[str, float]
) -> np.ndarray:
    """
    Aplica thresholds específicos por classe sobre a matriz de probabilidades.
    Retorna matriz binária de formato (N, 5).
    """
    binary_preds = np.zeros_like(probs, dtype=int)
    for idx, class_id in enumerate(MULTILABEL_CLASSES):
        class_name = CONCEPT_LABEL_SHORT_NAMES[class_id]
        th = thresholds.get(class_name, 0.50)
        binary_preds[:, idx] = (probs[:, idx] >= th).astype(int)

    return binary_preds
