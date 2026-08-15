"""
Testes unitários para o avaliador multilabel e provedores de LLM.
"""

import pytest
import numpy as np
from src.sld.evaluation.multilabel_evaluator import compute_multilabel_evaluation, compute_pr_curves_data
from src.sld.llm.llm_provider import MockLLMProvider, OllamaProvider
from src.sld.models.classification import ParagraphRecord


def test_compute_multilabel_evaluation():
    """Testa cálculo de F1 macro, micro, per-class P/R/F1 e matrizes de confusão."""
    y_true = np.array([
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 1],
        [1, 1, 0, 0, 0],
    ], dtype=int)

    y_probs = np.array([
        [0.8, 0.2, 0.9, 0.1, 0.3],
        [0.1, 0.85, 0.2, 0.7, 0.9],
        [0.75, 0.9, 0.3, 0.2, 0.1],
    ], dtype=np.float32)

    thresholds = {
        "definition": 0.5,
        "determinant": 0.5,
        "type_dimension": 0.5,
        "causal_relation": 0.5,
        "property": 0.5,
    }

    y_pred = (y_probs >= 0.5).astype(int)

    report = compute_multilabel_evaluation(
        model_id="test_model_v1",
        classifier_type="logistic_regression",
        y_true_binary=y_true,
        y_probs=y_probs,
        y_pred_binary=y_pred,
        thresholds=thresholds,
        total_articles=2
    )

    assert report.model_id == "test_model_v1"
    assert report.total_paragraphs == 3
    assert report.macro_f1 >= 0.0
    assert "definition" in report.per_class_metrics
    assert "definition" in report.confusion_matrices

    pr_curves = compute_pr_curves_data(y_true, y_probs)
    assert "definition" in pr_curves


def test_mock_llm_provider():
    """Testa provedor LLM abstrato para o corpus final."""
    provider = MockLLMProvider()
    paragraphs = [
        ParagraphRecord(
            paragraph_id="P0001",
            article_id="ART_01",
            text="Trecho de vulnerabilidade física a inundações.",
            predicted_labels=["determinant", "type_dimension"],
            status="FINAL_CORPUS"
        )
    ]

    res = provider.analyze(paragraphs, "Extrair fatores de vulnerabilidade")
    assert "[Mock LLM Analysis]" in res
    assert "Analisados 1 parágrafos" in res
