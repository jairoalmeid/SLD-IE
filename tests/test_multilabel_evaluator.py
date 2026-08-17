"""
Testes unitários rigorosos para o avaliador supervisionado multilabel e provedores de LLM.
Contempla os 10 cenários obrigatórios de validação estatística e metodológica.
"""

import pytest
import numpy as np
from src.sld.evaluation.multilabel_evaluator import (
    compute_multilabel_evaluation,
    compute_bootstrap_confidence_intervals,
    check_dataset_leakage,
    generate_evaluation_tables,
    generate_evaluation_markdown,
    compute_pr_curves_data
)
from src.sld.llm.llm_provider import MockLLMProvider
from src.sld.models.classification import ParagraphRecord

THRESHOLDS_DEFAULT = {
    "definition": 0.5,
    "determinant": 0.5,
    "type_dimension": 0.5,
    "causal_relation": 0.5,
    "property": 0.5,
}


def test_1_perfect_prediction():
    """1. Teste de Predição Perfeita: Todas as métricas devem ser 1.0 (ou N/A onde aplicável)."""
    y_true = np.array([
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],
    ], dtype=int)
    y_probs = y_true.astype(float)
    y_pred = y_true.copy()

    report = compute_multilabel_evaluation(
        model_id="perfect_model",
        classifier_type="OneVsRest Logistic",
        y_true_binary=y_true,
        y_probs=y_probs,
        y_pred_binary=y_pred,
        thresholds=THRESHOLDS_DEFAULT,
        total_articles=2,
        n_bootstraps=100
    )

    assert report.macro_f1 == 1.0
    assert report.micro_f1 == 1.0
    assert report.subset_accuracy == 1.0
    assert report.hamming_loss == 0.0
    for m in report.per_class_metrics.values():
        if m.support_positive > 0:
            assert m.precision == 1.0
            assert m.recall == 1.0
            assert m.f1 == 1.0


def test_2_false_positives_and_false_negatives():
    """2. Teste com Falsos Positivos e Falsos Negativos."""
    y_true = np.array([
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
    ], dtype=int)
    y_pred = np.array([
        [1, 1, 0, 0, 0],  # FP para classe 1 (determinant)
        [0, 1, 0, 0, 0],  # FN para classe 0 (definition)
        [0, 1, 1, 0, 0],  # Acerto
        [0, 0, 0, 1, 1],  # FN para classe 2, FP para classe 4
    ], dtype=int)
    y_probs = y_pred.astype(float) * 0.9

    report = compute_multilabel_evaluation(
        model_id="imperfect_model",
        classifier_type="OneVsRest Logistic",
        y_true_binary=y_true,
        y_probs=y_probs,
        y_pred_binary=y_pred,
        thresholds=THRESHOLDS_DEFAULT,
        total_articles=1,
        n_bootstraps=50
    )

    # Verifica presença de FP e FN nas matrizes
    def_m = report.per_class_metrics["definition"]
    assert def_m.true_positives == 1
    assert def_m.false_negatives == 1
    assert def_m.precision == 1.0
    assert def_m.recall == 0.5
    assert def_m.f1 == pytest.approx(0.6667, abs=1e-3)

    det_m = report.per_class_metrics["determinant"]
    assert det_m.true_positives == 2
    assert det_m.false_positives == 1
    assert det_m.precision == pytest.approx(0.6667, abs=1e-3)
    assert det_m.recall == 1.0


def test_3_class_without_positive_examples():
    """3. Teste de Classe sem exemplos positivos: Deve apresentar N/A e justificativa sem erro."""
    y_true = np.array([
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],  # Classe 5 (property) tem suporte 0
        [1, 1, 0, 0, 0],
    ], dtype=int)
    y_pred = np.array([
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 1, 0, 0, 0],
    ], dtype=int)

    report = compute_multilabel_evaluation(
        model_id="zero_support_model",
        classifier_type="OneVsRest Logistic",
        y_true_binary=y_true,
        y_probs=y_pred.astype(float),
        y_pred_binary=y_pred,
        thresholds=THRESHOLDS_DEFAULT,
        n_bootstraps=50
    )

    prop_m = report.per_class_metrics["property"]
    assert prop_m.support_positive == 0
    assert prop_m.precision is None
    assert prop_m.recall is None
    assert prop_m.f1 is None
    assert prop_m.specificity == 1.0
    assert "Sem exemplos positivos" in prop_m.note


def test_4_class_without_negative_examples():
    """4. Teste de Classe sem exemplos negativos (todos positivos): Specificity e ROC-AUC = N/A."""
    y_true = np.array([
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
    ], dtype=int)  # definition é positivo em todas as instâncias (0 negativos)
    y_pred = y_true.copy()

    report = compute_multilabel_evaluation(
        model_id="all_pos_model",
        classifier_type="OneVsRest Logistic",
        y_true_binary=y_true,
        y_probs=y_pred.astype(float),
        y_pred_binary=y_pred,
        thresholds=THRESHOLDS_DEFAULT,
        n_bootstraps=50
    )

    def_m = report.per_class_metrics["definition"]
    assert def_m.support_negative == 0
    assert def_m.specificity is None
    assert def_m.roc_auc is None
    assert "Sem exemplos negativos" in def_m.note


def test_5_dataset_with_class_0_examples():
    """5. Teste de conjunto contendo exemplos da Classe 0 (Não Relevante)."""
    y_true = np.array([
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],  # Instância da Classe 0
        [0, 1, 0, 0, 0],
    ], dtype=int)
    y_pred = np.array([
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],  # Previsto corretamente como Classe 0
        [0, 1, 0, 0, 0],
    ], dtype=int)

    report = compute_multilabel_evaluation(
        model_id="c0_present_model",
        classifier_type="OneVsRest Logistic",
        y_true_binary=y_true,
        y_probs=y_pred.astype(float),
        y_pred_binary=y_pred,
        thresholds=THRESHOLDS_DEFAULT,
        n_bootstraps=50
    )

    c0 = report.class_0_metrics
    assert c0 is not None
    assert c0.support_positive == 1
    assert c0.true_positives == 1
    assert c0.precision == 1.0
    assert c0.recall == 1.0
    assert c0.f1 == 1.0


def test_6_dataset_without_class_0_examples():
    """6. Teste de conjunto sem exemplos da Classe 0: Exibe N/A com aviso metodológico."""
    y_true = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=int)  # Todos os parágrafos possuem ao menos 1 classe ativa
    y_pred = y_true.copy()

    report = compute_multilabel_evaluation(
        model_id="c0_absent_model",
        classifier_type="OneVsRest Logistic",
        y_true_binary=y_true,
        y_probs=y_pred.astype(float),
        y_pred_binary=y_pred,
        thresholds=THRESHOLDS_DEFAULT,
        n_bootstraps=50
    )

    c0 = report.class_0_metrics
    assert c0 is not None
    assert c0.support_positive == 0
    assert c0.precision is None
    assert c0.recall is None
    assert c0.f1 is None
    assert "A amostra de avaliação não contém parágrafos não relevantes" in c0.note


def test_7_multilabel_cooccurrence():
    """7. Teste de classificação multirrótulo com coocorrência (mais de uma classe por parágrafo)."""
    y_true = np.array([
        [1, 1, 1, 0, 0],  # 3 rótulos ativos
        [0, 1, 0, 1, 1],  # 3 rótulos ativos
        [1, 0, 0, 0, 1],  # 2 rótulos ativos
    ], dtype=int)
    y_pred = y_true.copy()

    report = compute_multilabel_evaluation(
        model_id="multilabel_cooc",
        classifier_type="OneVsRest Logistic",
        y_true_binary=y_true,
        y_probs=y_pred.astype(float),
        y_pred_binary=y_pred,
        thresholds=THRESHOLDS_DEFAULT,
        n_bootstraps=50
    )

    # 8 rótulos positivos em 3 parágrafos -> Cardinalidade = 8/3 = 2.6667
    assert report.label_cardinality == pytest.approx(8 / 3, abs=1e-3)
    # Densidade = 8 / (3 * 5) = 8 / 15 = 0.5333
    assert report.label_density == pytest.approx(8 / 15, abs=1e-3)


def test_8_macro_f1_active_classes_only():
    """8. Teste de cálculo de Macro-F1 estritamente sobre as classes 1 a 5 (sem Classe 0)."""
    y_true = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ], dtype=int)
    # Predição com acerto em classes 1 a 4 e erro na classe 5
    y_pred = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],  # Classe 5: FN=1, TP=0 -> F1 = 0
    ], dtype=int)

    report = compute_multilabel_evaluation(
        model_id="macro_test",
        classifier_type="OneVsRest Logistic",
        y_true_binary=y_true,
        y_probs=y_pred.astype(float),
        y_pred_binary=y_pred,
        thresholds=THRESHOLDS_DEFAULT,
        n_bootstraps=50
    )

    # Classes 1..4 têm F1=1.0, Classe 5 tem F1=0.0 -> Média das 5 classes = 4.0 / 5 = 0.8000
    assert report.macro_f1 == pytest.approx(0.8000, abs=1e-3)


def test_9_confusion_matrices_consistency():
    """9. Teste de Consistência das matrizes de confusão (TP + FP + FN + TN = N)."""
    rng = np.random.RandomState(123)
    N = 40
    y_true = rng.randint(0, 2, size=(N, 5))
    y_probs = rng.uniform(0, 1, size=(N, 5))
    y_pred = (y_probs >= 0.5).astype(int)

    report = compute_multilabel_evaluation(
        model_id="consistency_model",
        classifier_type="OneVsRest Logistic",
        y_true_binary=y_true,
        y_probs=y_probs,
        y_pred_binary=y_pred,
        thresholds=THRESHOLDS_DEFAULT,
        total_articles=5,
        n_bootstraps=100
    )

    assert report.consistency_checks["confusion_matrices_valid"] is True
    for m in report.per_class_metrics.values():
        assert m.true_positives + m.false_positives + m.false_negatives + m.true_negatives == N
        assert m.true_positives + m.false_negatives == m.support_positive
        assert m.true_negatives + m.false_positives == m.support_negative


def test_10_bootstrap_reproducibility():
    """10. Teste de Reprodutibilidade do Bootstrap com random_state fixo."""
    y_true = np.array([
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [1, 0, 0, 1, 0],
        [0, 1, 1, 0, 1],
    ], dtype=int)
    y_pred = y_true.copy()

    ci_1 = compute_bootstrap_confidence_intervals(y_true, y_true.astype(float), y_pred, THRESHOLDS_DEFAULT, n_bootstraps=200, random_state=42)
    ci_2 = compute_bootstrap_confidence_intervals(y_true, y_true.astype(float), y_pred, THRESHOLDS_DEFAULT, n_bootstraps=200, random_state=42)

    assert ci_1["macro_f1_ci95"] == ci_2["macro_f1_ci95"]
    assert ci_1["micro_f1_ci95"] == ci_2["micro_f1_ci95"]
    assert ci_1["exact_match_ci95"] == ci_2["exact_match_ci95"]


def test_data_leakage_detection():
    """Teste da detecção de sobreposição e vazamento de dados."""
    has_leak, alerts = check_dataset_leakage(
        train_paragraph_ids=["P01", "P02", "P03"],
        eval_paragraph_ids=["P03", "P04"],
        train_texts=["texto A", "texto B"],
        eval_texts=["texto B", "texto C"],
        train_articles=["ART_1", "ART_2"],
        eval_articles=["ART_2", "ART_3"]
    )
    assert has_leak is True
    assert len(alerts) == 3


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

