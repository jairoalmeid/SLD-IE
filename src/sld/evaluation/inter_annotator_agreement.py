"""
Cálculo de concordância entre múltiplos anotadores (Cohen's Kappa e Percent Agreement).
Provê avaliação quantitativa da qualidade e reprodutibilidade do Gold Standard.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from src.sld.models.classification import AnnotationRecord
from src.sld.models.concept_label import MULTILABEL_CLASSES, CONCEPT_LABEL_NAMES


def compute_cohen_kappa(y1: Any, y2: Any) -> Dict[str, float]:
    """
    Calcula a concordância observada (p_o), esperada ao acaso (p_e) e o Cohen's Kappa (kappa)
    para dois vetores binários (0 ou 1) de anotações sobre as mesmas instâncias.

    Fórmula:
        kappa = (p_o - p_e) / (1 - p_e)
    """
    arr1 = np.asarray(y1, dtype=int)
    arr2 = np.asarray(y2, dtype=int)
    n = len(arr1)
    if n == 0 or len(arr2) != n:
        return {"p_o": 0.0, "p_e": 0.0, "kappa": 0.0, "n_samples": 0}

    # Concordância observada (p_o)
    p_o = float(np.mean(arr1 == arr2))

    # Proporções marginais de cada anotador para a classe 1
    p1_pos = float(np.mean(arr1 == 1))
    p2_pos = float(np.mean(arr2 == 1))

    p1_neg = 1.0 - p1_pos
    p2_neg = 1.0 - p2_pos

    # Concordância esperada ao acaso (p_e)
    p_e = (p1_pos * p2_pos) + (p1_neg * p2_neg)

    if abs(1.0 - p_e) < 1e-9:
        kappa = 1.0 if p_o == 1.0 else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return {
        "p_o": round(p_o, 4),
        "p_e": round(p_e, 4),
        "kappa": round(float(kappa), 4),
        "n_samples": n
    }


def compute_inter_annotator_agreement(
    annotations: List[AnnotationRecord]
) -> Dict[str, Any]:
    """
    Identifica parágrafos anotados por 2 anotadores distintos e calcula o Cohen's Kappa por classe.

    Retorna um relatório com a concordância por classe e o Macro Kappa consolidado.
    """
    # Agrupa por paragraph_id
    by_paragraph: Dict[str, List[AnnotationRecord]] = {}
    for a in annotations:
        if a.annotation_status == "valid":
            by_paragraph.setdefault(a.paragraph_id, []).append(a)

    # Filtra apenas parágrafos anotados por exatamente 2 anotadores distintos
    paired_items: List[Tuple[AnnotationRecord, AnnotationRecord]] = []
    for p_id, recs in by_paragraph.items():
        unique_annotators = {r.annotator_id for r in recs}
        if len(unique_annotators) >= 2:
            # Pega o primeiro registro de cada um dos dois primeiros anotadores
            seen_ids = set()
            pair = []
            for r in recs:
                if r.annotator_id not in seen_ids:
                    seen_ids.add(r.annotator_id)
                    pair.append(r)
                if len(pair) == 2:
                    break
            if len(pair) == 2:
                paired_items.append((pair[0], pair[1]))

    if not paired_items:
        return {
            "has_paired_annotations": False,
            "n_paired_paragraphs": 0,
            "message": "Nenhum parágrafo foi anotado por dois anotadores distintos para cálculo de concordância."
        }

    n_samples = len(paired_items)
    results_per_class: Dict[str, Dict[str, float]] = {}
    kappas: List[float] = []

    # Avalia para cada classe de 0 a 5
    for class_num in range(6):
        c_label = f"label_{class_num}"
        y1 = np.array([getattr(item[0], c_label) for item in paired_items], dtype=int)
        y2 = np.array([getattr(item[1], c_label) for item in paired_items], dtype=int)

        kappa_stats = compute_cohen_kappa(y1, y2)
        class_name = CONCEPT_LABEL_NAMES.get(class_num, f"Classe {class_num}")
        results_per_class[class_name] = kappa_stats
        kappas.append(kappa_stats["kappa"])

    macro_kappa = float(np.mean(kappas))

    return {
        "has_paired_annotations": True,
        "n_paired_paragraphs": n_samples,
        "annotator_pairs": list({(item[0].annotator_id, item[1].annotator_id) for item in paired_items}),
        "macro_kappa": round(macro_kappa, 4),
        "per_class_agreement": results_per_class
    }
