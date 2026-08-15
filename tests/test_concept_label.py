"""
Testes unitários para a taxonomia de rótulos conceituais e Regra da Classe 0.
"""

import pytest
from src.sld.models.concept_label import (
    ConceptLabel,
    validate_and_sanitize_labels,
    labels_to_binary_vector,
    binary_vector_to_labels,
)


def test_concept_label_enum_values():
    """Testa valores numéricos da enumeração de rótulos conceituais."""
    assert ConceptLabel.NOT_RELEVANT == 0
    assert ConceptLabel.DEFINITION == 1
    assert ConceptLabel.DETERMINANT == 2
    assert ConceptLabel.TYPE_DIMENSION == 3
    assert ConceptLabel.CAUSAL_RELATION == 4
    assert ConceptLabel.PROPERTY == 5


def test_class_0_mutual_exclusion_rule():
    """
    Testa rigorosamente a REGRA DA CLASSE 0:
    - [0] é válido
    - [0, 2] é INVÁLIDO e deve ser sanitizado para [2]
    - [1, 2, 4] é mantido como [1, 2, 4]
    - [] (vazio) retorna [0] por padrão
    """
    assert validate_and_sanitize_labels([0]) == [0]
    assert validate_and_sanitize_labels([0, 2]) == [2]
    assert validate_and_sanitize_labels([0, 1, 3, 5]) == [1, 3, 5]
    assert validate_and_sanitize_labels([2, 4]) == [2, 4]
    assert validate_and_sanitize_labels([]) == []


def test_labels_to_binary_vector():
    """Testa conversão de rótulos para vetor binário de 5 posições (classes 1 a 5)."""
    # Classe 0 (Não Relevante) -> todos zeros [0, 0, 0, 0, 0]
    assert labels_to_binary_vector([0]) == [0, 0, 0, 0, 0]
    assert labels_to_binary_vector([0, 2]) == [0, 1, 0, 0, 0]  # Regra da classe 0 remove 0

    # Classes [1, 3] -> [1, 0, 1, 0, 0]
    assert labels_to_binary_vector([1, 3]) == [1, 0, 1, 0, 0]

    # Classes [2, 4, 5] -> [0, 1, 0, 1, 1]
    assert labels_to_binary_vector([2, 4, 5]) == [0, 1, 0, 1, 1]


def test_binary_vector_to_labels():
    """Testa conversão inversa de vetor binário de 5 posições para lista de rótulos."""
    assert binary_vector_to_labels([0, 0, 0, 0, 0]) == [0]
    assert binary_vector_to_labels([1, 0, 1, 0, 0]) == [1, 3]
    assert binary_vector_to_labels([0, 1, 0, 1, 1]) == [2, 4, 5]
