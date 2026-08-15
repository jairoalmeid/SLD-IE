"""
Gerenciador do Conjunto de Referências Semânticas Multidimensionais (Semantic Reference Set).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class SemanticAnchor:
    """Representa uma sentença-âncora conceitual individual."""
    id: str  # e.g., Q1, Q2
    text: str
    description: str = ""
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "description": self.description,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticAnchor":
        return cls(
            id=data["id"],
            text=data["text"],
            description=data.get("description", ""),
            weight=float(data.get("weight", 1.0)),
        )


# Conjunto padrão de âncoras para pesquisa de vulnerabilidade a desastres
DEFAULT_DISASTER_VULNERABILITY_ANCHORS: List[SemanticAnchor] = [
    SemanticAnchor(
        id="Q1",
        text="Definição geral de vulnerabilidade no contexto de riscos e desastres ambientais e tecnológicos.",
        description="Conceito geral e caracterização conceitual de vulnerabilidade.",
        weight=1.0,
    ),
    SemanticAnchor(
        id="Q2",
        text="Condições físicas, ambientais e de infraestrutura vulnerável que aumentam a suscetibilidade a danos.",
        description="Vulnerabilidade física e infraestrutural.",
        weight=1.0,
    ),
    SemanticAnchor(
        id="Q3",
        text="Fatores sociais, demográficos, desigualdade e marginalização populacional produtores de vulnerabilidade.",
        description="Vulnerabilidade social e fatores demográficos.",
        weight=1.0,
    ),
    SemanticAnchor(
        id="Q4",
        text="Fatores econômicos, pobreza, escassez de recursos e perda de subsistência em desastres.",
        description="Vulnerabilidade econômica e perda de ativos.",
        weight=1.0,
    ),
    SemanticAnchor(
        id="Q5",
        text="Fatores institucionais, governança, capacidade frágil de resposta e ausência de políticas públicas.",
        description="Vulnerabilidade institucional e de governança.",
        weight=1.0,
    ),
    SemanticAnchor(
        id="Q6",
        text="Capacidade desigual de antecipação, absorção, adaptação e recuperação pós-desastre.",
        description="Capacidade de enfrentamento e resiliência diferencial.",
        weight=1.0,
    ),
    SemanticAnchor(
        id="Q7",
        text="Exposição contínua a perigos naturais e suscetibilidade a eventos climáticos extremos.",
        description="Exposição física e suscetibilidade ambiental.",
        weight=1.0,
    ),
]


class SemanticReferenceSet:
    """Gerencia o conjunto de sentenças-âncora e estratégias de agregação semântica."""

    AGGREGATION_STRATEGIES = ["maximum", "mean", "weighted_mean", "centroid"]

    def __init__(self, anchors: Optional[List[SemanticAnchor]] = None):
        self.anchors = anchors if anchors is not None else []

    def add_anchor(self, text: str, description: str = "", weight: float = 1.0) -> SemanticAnchor:
        anchor_id = f"Q{len(self.anchors) + 1}"
        anchor = SemanticAnchor(id=anchor_id, text=text, description=description, weight=weight)
        self.anchors.append(anchor)
        return anchor

    def remove_anchor(self, anchor_id: str) -> bool:
        initial_len = len(self.anchors)
        self.anchors = [a for a in self.anchors if a.id != anchor_id]
        # Re-indexa IDs Q1..Qn
        for idx, a in enumerate(self.anchors, start=1):
            a.id = f"Q{idx}"
        return len(self.anchors) < initial_len

    def get_anchor_texts(self) -> List[str]:
        return [a.text for a in self.anchors]

    def to_list_dicts(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self.anchors]

    @classmethod
    def from_list_dicts(cls, data_list: List[Dict[str, Any]]) -> "SemanticReferenceSet":
        anchors = [SemanticAnchor.from_dict(d) for d in data_list]
        return cls(anchors=anchors)

    @staticmethod
    def aggregate(
        anchor_sims: Dict[str, float],
        strategy: str = "maximum",
        weights: Optional[Dict[str, float]] = None,
        centroid_sim: Optional[float] = None
    ) -> float:
        """
        Agrega os scores de similaridade individuais por âncora segundo a estratégia metodológica escolhida.
        """
        if not anchor_sims:
            return 0.0

        scores = list(anchor_sims.values())

        if strategy == "maximum":
            return float(max(scores))
        elif strategy == "mean":
            return float(np.mean(scores))
        elif strategy == "weighted_mean":
            if not weights:
                return float(np.mean(scores))
            total_weight = sum(weights.get(k, 1.0) for k in anchor_sims.keys())
            if total_weight == 0:
                return float(np.mean(scores))
            weighted_sum = sum(anchor_sims[k] * weights.get(k, 1.0) for k in anchor_sims.keys())
            return float(weighted_sum / total_weight)
        elif strategy == "centroid":
            return float(centroid_sim if centroid_sim is not None else max(scores))

        return float(max(scores))
