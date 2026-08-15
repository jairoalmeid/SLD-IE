"""
Divisão de dados sem Data Leakage (Group Split por article_id).
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from src.sld.models.classification import MultilabelAnnotation, ParagraphRecord


def split_annotations_by_article(
    annotations: List[MultilabelAnnotation],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[MultilabelAnnotation], List[MultilabelAnnotation], List[MultilabelAnnotation]]:
    """
    Divide anotações em Treino, Validação e Teste agrupando obrigatoriamente por `article_id`.
    Garante que 0% dos parágrafos de um mesmo artigo vazem entre os conjuntos.
    """
    if not annotations:
        return [], [], []

    # Se houver apenas 1 ou 2 artigos no total, atribui ao treino para evitar erros
    unique_articles = sorted(list(set(a.article_id for a in annotations)))
    if len(unique_articles) < 3:
        return annotations, [], []

    groups = np.array([a.article_id for a in annotations])

    # 1. Primeiro split: Separa Treino de (Validação + Teste)
    val_test_ratio = val_ratio + test_ratio
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_seed)
    train_idx, temp_idx = next(gss1.split(annotations, groups=groups))

    train_annos = [annotations[i] for i in train_idx]
    temp_annos = [annotations[i] for i in temp_idx]
    temp_groups = np.array([a.article_id for a in temp_annos])

    # Se val_ratio == 0, tudo o que sobrou vai para teste
    if val_ratio <= 0 or not temp_annos or len(set(temp_groups)) < 2:
        return train_annos, [], temp_annos

    # 2. Segundo split: Separa Validação de Teste proporcionalmente
    relative_val_ratio = val_ratio / val_test_ratio
    gss2 = GroupShuffleSplit(n_splits=1, train_size=relative_val_ratio, random_state=random_seed)
    val_sub_idx, test_sub_idx = next(gss2.split(temp_annos, groups=temp_groups))

    val_annos = [temp_annos[i] for i in val_sub_idx]
    test_annos = [temp_annos[i] for i in test_sub_idx]

    return train_annos, val_annos, test_annos


def verify_no_data_leakage(
    train_items: List[Any],
    val_items: List[Any],
    test_items: List[Any]
) -> bool:
    """Verifica formalmente que nenhum article_id é compartilhado entre train, val e test."""
    train_arts = set(getattr(x, "article_id", x) for x in train_items)
    val_arts = set(getattr(x, "article_id", x) for x in val_items)
    test_arts = set(getattr(x, "article_id", x) for x in test_items)

    leak_train_val = train_arts.intersection(val_arts)
    leak_train_test = train_arts.intersection(test_arts)
    leak_val_test = val_arts.intersection(test_arts)

    assert len(leak_train_val) == 0, f"Data Leakage entre Treino e Validação: {leak_train_val}"
    assert len(leak_train_test) == 0, f"Data Leakage entre Treino e Teste: {leak_train_test}"
    assert len(leak_val_test) == 0, f"Data Leakage entre Validação e Teste: {leak_val_test}"

    return True
