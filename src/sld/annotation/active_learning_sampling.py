"""
Estratégias de amostragem auditáveis para validação manual de parágrafos e Active Learning.
"""

from typing import List, Optional, Any
import numpy as np
from src.sld.models.classification import ParagraphRecord


def sample_paragraphs(
    records: List[ParagraphRecord],
    n_samples: int = 5,
    strategy: str = "random",
    random_seed: int = 42
) -> List[ParagraphRecord]:
    """
    Amostra `n_samples` parágrafos do corpus segundo o critério especificado:
    - 'random': Amostragem aleatória simples.
    - 'stratified': Amostragem estratificada por faixas/quantis de similaridade do cosseno.
    - 'uncertainty': Active Learning (maior incerteza, probabilidades próximas de 0.50).
    - 'top_k': Maiores scores de similaridade semântica por cosseno.
    """
    if not records:
        return []

    if len(records) <= n_samples:
        return list(records)

    rng = np.random.default_rng(random_seed)

    if strategy == "top_k":
        sorted_recs = sorted(records, key=lambda r: r.semantic_score or 0.0, reverse=True)
        return sorted_recs[:n_samples]

    elif strategy == "stratified":
        bins = [(0.0, 0.40), (0.40, 0.60), (0.60, 0.75), (0.75, 1.00)]
        grouped = {b: [] for b in bins}
        for r in records:
            s = r.semantic_score or 0.0
            for b in bins:
                if b[0] <= s < b[1] or (b[1] == 1.00 and s >= 1.00):
                    grouped[b].append(r)
                    break

        samples_per_bin = max(1, n_samples // len(bins))
        selected: List[ParagraphRecord] = []

        for b, bin_recs in grouped.items():
            if not bin_recs:
                continue
            k = min(len(bin_recs), samples_per_bin)
            chosen_idx = rng.choice(len(bin_recs), size=k, replace=False)
            selected.extend([bin_recs[i] for i in chosen_idx])

        if len(selected) < n_samples:
            remaining = [r for r in records if r not in selected]
            needed = min(n_samples - len(selected), len(remaining))
            if needed > 0:
                extra_idx = rng.choice(len(remaining), size=needed, replace=False)
                selected.extend([remaining[i] for i in extra_idx])

        return selected[:n_samples]

    elif strategy == "uncertainty":
        def calculate_uncertainty(r: ParagraphRecord) -> float:
            if not r.predicted_probabilities:
                return 0.0
            dists = [abs(prob - 0.50) for prob in r.predicted_probabilities.values()]
            return float(np.mean(dists)) if dists else 1.0

        sorted_recs = sorted(records, key=calculate_uncertainty)
        return sorted_recs[:n_samples]

    else:  # "random" por padrão
        chosen_idx = rng.choice(len(records), size=n_samples, replace=False)
        return [records[i] for i in chosen_idx]
