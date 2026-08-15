"""
Testes unitários para divisão de dados por article_id e prevenção de Data Leakage.
"""

import pytest
from src.sld.models.classification import MultilabelAnnotation
from src.sld.evaluation.group_splitter import split_annotations_by_article, verify_no_data_leakage


def test_split_annotations_by_article_no_data_leakage():
    """Testa divisão de anotações garantindo 0% data leakage entre os conjuntos."""
    annotations = []
    # 10 artigos diferentes, cada um com 5 parágrafos
    for art_idx in range(1, 11):
        art_id = f"SLD_ART_{art_idx:03d}"
        for p_idx in range(1, 6):
            p_id = f"P{p_idx:04d}"
            annotations.append(
                MultilabelAnnotation(
                    paragraph_id=p_id,
                    article_id=art_id,
                    text=f"Texto do artigo {art_id} parágrafo {p_id}",
                    labels=[1, 2] if p_idx % 2 == 0 else [0]
                )
            )

    train_annos, val_annos, test_annos = split_annotations_by_article(
        annotations, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, random_seed=42
    )

    assert len(train_annos) > 0
    assert len(val_annos) > 0
    assert len(test_annos) > 0

    # Verifica formalmente ausência de Data Leakage
    assert verify_no_data_leakage(train_annos, val_annos, test_annos) is True

    # Verifica que nenhum article_id de treino aparece no teste ou validação
    train_arts = set(a.article_id for a in train_annos)
    val_arts = set(a.article_id for a in val_annos)
    test_arts = set(a.article_id for a in test_annos)

    assert len(train_arts.intersection(val_arts)) == 0
    assert len(train_arts.intersection(test_arts)) == 0
    assert len(val_arts.intersection(test_arts)) == 0
