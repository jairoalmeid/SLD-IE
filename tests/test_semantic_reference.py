"""
Testes unitários para o gerenciador de referências semânticas multidimensionais e estratégias de agregação.
"""

import pytest
from src.sld.semantic.semantic_reference import SemanticReferenceSet, SemanticAnchor


def test_semantic_reference_set_defaults():
    """Testa inicialização do conjunto de âncoras (deve iniciar vazio por padrão)."""
    ref_set = SemanticReferenceSet()
    assert len(ref_set.anchors) == 0


def test_add_and_remove_anchor():
    """Testa adição e remoção de frases-âncora editáveis."""
    ref_set = SemanticReferenceSet()
    assert len(ref_set.anchors) == 0

    new_anchor = ref_set.add_anchor("Capacidade institucional de resposta a emergências.", "Vulnerabilidade institucional")
    assert len(ref_set.anchors) == 1
    assert new_anchor.id == "Q1"

    removed = ref_set.remove_anchor("Q1")
    assert removed is True
    assert len(ref_set.anchors) == 0


def test_aggregation_strategies():
    """Testa estratégias de agregação multi-âncora (maximum, mean, weighted_mean, centroid)."""
    scores = {"Q1": 0.80, "Q2": 0.40, "Q3": 0.60}
    weights = {"Q1": 1.0, "Q2": 2.0, "Q3": 1.0}

    # Maximum: max(0.80, 0.40, 0.60) = 0.80
    agg_max = SemanticReferenceSet.aggregate(scores, strategy="maximum")
    assert pytest.approx(agg_max, 0.001) == 0.80

    # Mean: (0.80 + 0.40 + 0.60) / 3 = 0.60
    agg_mean = SemanticReferenceSet.aggregate(scores, strategy="mean")
    assert pytest.approx(agg_mean, 0.001) == 0.60

    # Weighted Mean: (0.80*1 + 0.40*2 + 0.60*1) / (1 + 2 + 1) = 2.2 / 4 = 0.55
    agg_wm = SemanticReferenceSet.aggregate(scores, strategy="weighted_mean", weights=weights)
    assert pytest.approx(agg_wm, 0.001) == 0.55

    # Centroid
    agg_cent = SemanticReferenceSet.aggregate(scores, strategy="centroid", centroid_sim=0.75)
    assert pytest.approx(agg_cent, 0.001) == 0.75
