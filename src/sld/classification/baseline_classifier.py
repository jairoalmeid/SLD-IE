"""
Classificador Supervisionado Multilabel (Sentence Embeddings + OneVsRest Logistic Regression).
"""

import os
import json
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from src.sld.models.concept_label import MULTILABEL_CLASSES, CONCEPT_LABEL_SHORT_NAMES, validate_and_sanitize_labels
from src.sld.models.classification import ModelVersionMetadata
from src.sld.classification.threshold_optimizer import optimize_thresholds_per_class, apply_class_thresholds
from src.sld.utils.files import ensure_directory


class MultilabelLogisticClassifier:
    """Classificador Multilabel treinado diretamente sobre os Sentence Embeddings."""

    def __init__(
        self,
        random_state: int = 42,
        C: float = 1.0,
        class_weight: Optional[str] = "balanced",
        max_iter: int = 2000
    ):
        self.random_state = random_state
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.model = OneVsRestClassifier(
            LogisticRegression(
                solver="liblinear",
                C=self.C,
                class_weight=self.class_weight,
                random_state=self.random_state,
                max_iter=self.max_iter
            )
        )
        self.thresholds: Dict[str, float] = {
            CONCEPT_LABEL_SHORT_NAMES[c]: 0.50 for c in MULTILABEL_CLASSES
        }
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Treina o classificador One-vs-Rest e otimiza thresholds por classe se X_val for fornecido.
        - X_train: (N, D) matriz de embeddings de treino
        - y_train: (N, 5) matriz binária de rótulos de treino
        """
        if X_train.shape[0] == 0:
            raise ValueError("X_train está vazio. Forneça exemplos rotulados no Gold Standard.")

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        if X_val is not None and y_val is not None and X_val.shape[0] > 0:
            val_probs = self.predict_proba(X_val)
            self.thresholds = optimize_thresholds_per_class(val_probs, y_val)

        return self.thresholds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna a matriz de probabilidades (N, 5) para as 5 classes conceituais."""
        if not self.is_fitted:
            raise RuntimeError("O modelo ainda não foi treinado.")

        if X is None or X.size == 0:
            return np.empty((0, 5), dtype=np.float32)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        try:
            probs = self.model.predict_proba(X)
            if probs.shape[1] < 5:
                full_probs = np.zeros((X.shape[0], 5), dtype=np.float32)
                full_probs[:, :probs.shape[1]] = probs
                return full_probs
            return probs.astype(np.float32)
        except Exception:
            decisions = self.model.decision_function(X)
            sigmoids = 1.0 / (1.0 + np.exp(-decisions))
            return sigmoids.astype(np.float32)

    def predict(
        self,
        X: np.ndarray,
        thresholds: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Aplica os thresholds calibrados sobre as probabilidades e retorna matriz binária (N, 5)."""
        if X is None or X.size == 0:
            return np.empty((0, 5), dtype=int)

        probs = self.predict_proba(X)
        effective_ths = thresholds if thresholds is not None else self.thresholds
        return apply_class_thresholds(probs, effective_ths)

    def predict_with_thresholds(
        self,
        X: np.ndarray,
        thresholds: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Aplica thresholds customizados ou ótimos sobre as probabilidades e retorna matriz binária (N, 5)."""
        return self.predict(X, thresholds=thresholds)

    def save(self, output_dir: Path, metadata: Optional[ModelVersionMetadata] = None) -> Path:
        """Salva o modelo, thresholds e metadados de versão em disco."""
        output_path = Path(output_dir).expanduser().resolve()
        ensure_directory(output_path)

        joblib.dump(self.model, output_path / "logistic_classifier.joblib")
        with open(output_path / "thresholds.json", "w", encoding="utf-8") as f:
            json.dump(self.thresholds, f, indent=2)

        if metadata:
            with open(output_path / "model_metadata.json", "w", encoding="utf-8") as f:
                f.write(metadata.model_dump_json(indent=2))

        return output_path

    @classmethod
    def load(cls, input_dir: Path) -> "MultilabelLogisticClassifier":
        """Carrega modelo de regressão logística e thresholds salvos em disco."""
        input_path = Path(input_dir).expanduser().resolve()
        clf = cls()
        clf.model = joblib.load(input_path / "logistic_classifier.joblib")
        clf.is_fitted = True

        if (input_path / "thresholds.json").exists():
            with open(input_path / "thresholds.json", "r", encoding="utf-8") as f:
                clf.thresholds = json.load(f)

        return clf
