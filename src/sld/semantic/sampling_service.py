"""
Serviço de amostragem científica para anotação humana (Random, Stratified, Boundary Sampling).
"""

import random
from typing import List, Dict, Any
import numpy as np
from src.sld.models.search_result import SearchResult, Segment


def random_sampling(
    segments: List[Segment],
    sample_size: int = 30,
    seed: int = 42
) -> List[Segment]:
    """Amostragem aleatória simples sobre os parágrafos válidos."""
    valid_segs = [s for s in segments if s.status == "valid_paragraph"]
    if not valid_segs:
        return []

    rng = random.Random(seed)
    size = min(sample_size, len(valid_segs))
    return rng.sample(valid_segs, size)


def stratified_by_similarity_sampling(
    results: List[SearchResult],
    sample_size: int = 30,
    seed: int = 42
) -> List[SearchResult]:
    """
    Amostragem estratificada por faixas de similaridade
    (Muito Baixa, Baixa, Intermediária, Alta, Muito Alta).
    """
    if not results:
        return []

    scores = [r.aggregate_score for r in results]
    min_s, max_s = min(scores), max(scores)

    if min_s == max_s:
        return random_sampling_results(results, sample_size, seed)

    # 5 estratos de similaridade
    strata: Dict[str, List[SearchResult]] = {
        "muito_baixa": [],
        "baixa": [],
        "intermediaria": [],
        "alta": [],
        "muito_alta": []
    }

    step = (max_s - min_s) / 5.0
    for r in results:
        s = r.aggregate_score
        if s <= min_s + step:
            strata["muito_baixa"].append(r)
        elif s <= min_s + 2 * step:
            strata["baixa"].append(r)
        elif s <= min_s + 3 * step:
            strata["intermediaria"].append(r)
        elif s <= min_s + 4 * step:
            strata["alta"].append(r)
        else:
            strata["muito_alta"].append(r)

    per_stratum = max(1, sample_size // 5)
    rng = random.Random(seed)
    sampled: List[SearchResult] = []

    for name, items in strata.items():
        if items:
            k = min(per_stratum, len(items))
            sampled.extend(rng.sample(items, k))

    # Preenche se faltar para atingir sample_size
    if len(sampled) < sample_size and len(results) > len(sampled):
        remaining = [r for r in results if r not in sampled]
        k_extra = min(sample_size - len(sampled), len(remaining))
        sampled.extend(rng.sample(remaining, k_extra))

    return sampled


def boundary_sampling(
    results: List[SearchResult],
    threshold: float = 0.50,
    margin: float = 0.05,
    sample_size: int = 30,
    seed: int = 42
) -> List[SearchResult]:
    """
    Amostragem de fronteira selecionando parágrafos cujos scores
    estão próximos ao threshold semântico (theta +- margin).
    """
    boundary_items = [
        r for r in results if abs(r.aggregate_score - threshold) <= margin
    ]

    rng = random.Random(seed)
    if boundary_items:
        k = min(sample_size, len(boundary_items))
        return rng.sample(boundary_items, k)
    else:
        # Fallback para os mais próximos do threshold
        sorted_by_diff = sorted(results, key=lambda r: abs(r.aggregate_score - threshold))
        return sorted_by_diff[:min(sample_size, len(sorted_by_diff))]


def random_sampling_results(
    results: List[SearchResult],
    sample_size: int = 30,
    seed: int = 42
) -> List[SearchResult]:
    """Amostragem aleatória sobre a lista de SearchResult."""
    rng = random.Random(seed)
    size = min(sample_size, len(results))
    return rng.sample(results, size)
