"""
Taxonomia de rótulos conceituais genéricos e regra de exclusividade mútua da Classe 0.
Compatível com qualquer conceito de pesquisa acadêmica (resiliência, risco, governança, etc.).
"""

from enum import IntEnum
from typing import List, Dict, Union, Set


class ConceptLabel(IntEnum):
    """Categorias formais da taxonomia conceitual genérica."""
    NOT_RELEVANT = 0
    DEFINITION = 1
    DETERMINANT = 2
    TYPE_DIMENSION = 3
    CAUSAL_RELATION = 4
    PROPERTY = 5


CONCEPT_LABEL_NAMES: Dict[int, str] = {
    ConceptLabel.NOT_RELEVANT: "0 — Não relevante",
    ConceptLabel.DEFINITION: "1 — Definição ou conceituação",
    ConceptLabel.DETERMINANT: "2 — Fator determinante",
    ConceptLabel.TYPE_DIMENSION: "3 — Tipo ou dimensão",
    ConceptLabel.CAUSAL_RELATION: "4 — Relação causal",
    ConceptLabel.PROPERTY: "5 — Característica ou propriedade",
}

CONCEPT_LABEL_SHORT_NAMES: Dict[int, str] = {
    ConceptLabel.NOT_RELEVANT: "not_relevant",
    ConceptLabel.DEFINITION: "definition",
    ConceptLabel.DETERMINANT: "determinant",
    ConceptLabel.TYPE_DIMENSION: "type_dimension",
    ConceptLabel.CAUSAL_RELATION: "causal_relation",
    ConceptLabel.PROPERTY: "property",
}

CONCEPT_LABEL_DESCRIPTIONS: Dict[int, str] = {
    ConceptLabel.NOT_RELEVANT: "O trecho não apresenta conteúdo conceitualmente útil para os objetivos definidos na pesquisa.",
    ConceptLabel.DEFINITION: "O trecho apresenta uma definição, delimitação, explicação conceitual ou formulação explícita sobre o conceito investigado.",
    ConceptLabel.DETERMINANT: "O trecho identifica uma condição, elemento, processo ou fator que contribui para produzir, aumentar, reduzir ou modificar o conceito investigado.",
    ConceptLabel.TYPE_DIMENSION: "O trecho apresenta categorias, tipos, dimensões, componentes ou subdivisões do conceito investigado.",
    ConceptLabel.CAUSAL_RELATION: "O trecho descreve uma relação de causa, efeito, mecanismo, mediação ou influência envolvendo o conceito investigado.",
    ConceptLabel.PROPERTY: "O trecho apresenta atributos, propriedades, comportamentos, condições ou características associadas ao conceito investigado.",
}

# As 5 classes conceituais ativas para classificação multilabel (1 a 5)
MULTILABEL_CLASSES: List[int] = [
    ConceptLabel.DEFINITION,
    ConceptLabel.DETERMINANT,
    ConceptLabel.TYPE_DIMENSION,
    ConceptLabel.CAUSAL_RELATION,
    ConceptLabel.PROPERTY,
]

MULTILABEL_CLASS_NAMES: List[str] = [
    CONCEPT_LABEL_SHORT_NAMES[c] for c in MULTILABEL_CLASSES
]


def validate_and_sanitize_labels(labels: Union[List[int], Set[int]]) -> List[int]:
    """
    Enforça a REGRA DA CLASSE 0:
    - As classes 1-5 são multilabel.
    - Se qualquer classe 1-5 estiver presente, a classe 0 é desativada.
    - Se apenas a classe 0 for selecionada, retorna [0].
    - Se nenhuma classe for selecionada, retorna [] para indicar 'unannotated'.
    """
    if labels is None:
        return []

    label_set = set(labels)
    active_concept_classes = label_set.intersection({1, 2, 3, 4, 5})

    if active_concept_classes:
        return sorted(list(active_concept_classes))
    elif 0 in label_set:
        return [0]
    
    return []


def labels_to_binary_vector(labels: List[int]) -> List[int]:
    """Converte lista de rótulos ativos em vetor binário de 5 posições (para classes 1 a 5)."""
    sanitized = validate_and_sanitize_labels(labels)
    if 0 in sanitized or not sanitized:
        return [0, 0, 0, 0, 0]
    return [1 if c in sanitized else 0 for c in MULTILABEL_CLASSES]


def binary_vector_to_labels(vec: List[int]) -> List[int]:
    """Converte vetor binário de 5 posições de volta para lista de rótulos (ou [0] se zeros)."""
    active = [c for c, val in zip(MULTILABEL_CLASSES, vec) if val == 1]
    return active if active else [0]
