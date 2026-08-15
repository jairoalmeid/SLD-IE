"""
Testes unitários para o módulo de remoção conservadora de referências bibliográficas.
"""

import pytest
from src.sld.ingestion.reference_remover import remove_references, ReferenceRemovalDecision


def test_reference_removal_high_confidence():
    """Testa remoção de referências quando há forte evidência (título, anos, posições finais)."""
    pages_data = [
        {"page": 1, "text": "Este é o corpo principal do artigo científico sobre biologia computacional."},
        {"page": 2, "text": "Continuação da discussão das metodologias e resultados obtidos."},
        {"page": 3, "text": "Considerações finais sobre o método apresentado.\n\nReferences\n[1] Smith, A. (2020). Journal of Science, 10(2), 123-130. https://doi.org/10.1234/5678\n[2] Silva, B., & Santos, C. (2021). Revista de Genômica, 45, 89-99.\n[3] Johnson, D. et al. (2019). Nature Genetics, 15, 300-312."}
    ]

    filtered_pages, decision = remove_references(pages_data, confidence_threshold=0.60)

    assert decision.references_removed is True
    assert decision.detected_title == "References"
    assert decision.start_page == 3
    assert decision.confidence >= 0.60
    assert len(filtered_pages) == 3
    assert "Considerações finais" in filtered_pages[2]["text"]
    assert "Smith, A. (2020)" not in filtered_pages[2]["text"]


def test_reference_preservation_low_confidence():
    """Testa preservação do conteúdo quando a palavra 'referências' é mencionada de forma ambígua no corpo."""
    pages_data = [
        {"page": 1, "text": "Neste trabalho fazemos referência às metodologias anteriores desenvolvidas em 2020."},
        {"page": 2, "text": "Concluímos que a nossa abordagem traz inovações importantes."}
    ]

    filtered_pages, decision = remove_references(pages_data, confidence_threshold=0.60)

    assert decision.references_removed is False
    assert len(filtered_pages) == 2
    assert "referência" in filtered_pages[0]["text"]


def test_empty_pages_data():
    """Testa comportamento seguro com lista vazia de páginas."""
    filtered_pages, decision = remove_references([], confidence_threshold=0.60)

    assert decision.references_removed is False
    assert decision.confidence == 0.0
    assert len(filtered_pages) == 0
