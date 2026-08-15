"""
Testes unitários para o classificador Supervisionado Multilabel (Regressão Logística) e otimização de thresholds.
"""

import pytest
import numpy as np
from src.sld.classification.baseline_classifier import MultilabelLogisticClassifier
from src.sld.classification.threshold_optimizer import optimize_thresholds_per_class, apply_class_thresholds


def test_baseline_logistic_regression_fit_predict():
    """Testa treino, predição de probabilidades e thresholds do classificador Multilabel LR."""
    rng = np.random.default_rng(42)

    # 50 exemplos sintéticos de treino (dimensão 10)
    X_train = rng.normal(size=(50, 10)).astype(np.float32)
    y_train = rng.integers(0, 2, size=(50, 5)).astype(int)

    # 20 exemplos de validação
    X_val = rng.normal(size=(20, 10)).astype(np.float32)
    y_val = rng.integers(0, 2, size=(20, 5)).astype(int)

    clf = MultilabelLogisticClassifier(random_state=42)
    thresholds = clf.fit(X_train, y_train, X_val, y_val)

    assert clf.is_fitted is True
    assert len(thresholds) == 5
    assert "definition" in thresholds

    # Teste de probabilidade (N, 5)
    probs = clf.predict_proba(X_val)
    assert probs.shape == (20, 5)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # Teste de predição binária (N, 5)
    preds = clf.predict(X_val)
    assert preds.shape == (20, 5)
    assert set(np.unique(preds)).issubset({0, 1})


def test_threshold_optimizer():
    """Testa busca empírica de thresholds por classe."""
    probs = np.array([
        [0.1, 0.9, 0.4, 0.8, 0.2],
        [0.7, 0.3, 0.8, 0.2, 0.9],
        [0.6, 0.8, 0.2, 0.7, 0.1],
    ], dtype=np.float32)

    targets = np.array([
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [1, 1, 0, 1, 0],
    ], dtype=int)

    thresholds = optimize_thresholds_per_class(probs, targets, step=0.1)
    assert len(thresholds) == 5

    binary_preds = apply_class_thresholds(probs, thresholds)
    assert binary_preds.shape == (3, 5)
