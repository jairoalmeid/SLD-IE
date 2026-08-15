"""
Serviço de calibração de limiar, avaliação quantitativa por Mínimo Recall, Gold Standard e métricas de IR.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from src.sld.models.evaluation import (
    GoldStandardAnnotation,
    ThresholdMetrics,
    EvaluationSummary,
    IRRankingMetrics,
)
from src.sld.models.search_result import SearchResult, Segment


def load_gold_standard(filepath: Path) -> List[GoldStandardAnnotation]:
    """Carrega anotações humanas do arquivo JSON ou JSONL."""
    if not filepath.exists():
        return []
    annotations = []
    with open(filepath, "r", encoding="utf-8") as f:
        if filepath.suffix == ".jsonl":
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    annotations.append(GoldStandardAnnotation.from_dict(data))
        else:
            data_list = json.load(f)
            for d in data_list:
                annotations.append(GoldStandardAnnotation.from_dict(d))
    return annotations


def save_gold_standard(annotations: List[GoldStandardAnnotation], filepath: Path):
    """Salva anotações humanas do Gold Standard em formato JSONL."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for ann in annotations:
            f.write(json.dumps(ann.to_dict(), ensure_ascii=False) + "\n")


def calibrate_threshold_by_minimum_recall(
    results: List[SearchResult],
    gold_standard: List[GoldStandardAnnotation],
    minimum_recall: float = 0.90,
    threshold_step: float = 0.02
) -> Tuple[EvaluationSummary, Dict[str, SearchResult]]:
    """
    Realiza a calibração empírica do threshold semântico selecionando o maior threshold que satisfaz
    a restrição metodológica de Mínimo Recall (ex: Recall >= 0.90).
    """
    # Mapeia anotações humanas por paragraph_id
    gold_map = {ann.paragraph_id: ann.human_label for ann in gold_standard}
    if not gold_map:
        # Se não houver anotações humanas, gera sumário default sem calibração
        return _generate_uncalibrated_summary(results, minimum_recall), {}

    # Filtra apenas os resultados que possuem anotação no Gold Standard
    eval_pairs: List[Tuple[float, int, SearchResult]] = []
    for r in results:
        if r.paragraph_id in gold_map:
            label = gold_map[r.paragraph_id]
            eval_pairs.append((r.aggregate_score, label, r))

    if not eval_pairs:
        return _generate_uncalibrated_summary(results, minimum_recall), {}

    total_relevant = sum(1 for _, label, _ in eval_pairs if label == 1)
    total_samples = len(eval_pairs)

    if total_relevant == 0:
        return _generate_uncalibrated_summary(results, minimum_recall), {}

    thresholds = np.arange(0.20, 0.96, threshold_step)
    metrics_list: List[ThresholdMetrics] = []

    best_calibrated_threshold = 0.50
    highest_f1_for_recall = -1.0
    best_achieved_recall = 0.0
    best_achieved_precision = 0.0
    best_achieved_f1 = 0.0

    total_corpus_count = max(1, len(results))

    for th in thresholds:
        th_float = float(th)
        tp = sum(1 for score, label, _ in eval_pairs if score >= th_float and label == 1)
        fp = sum(1 for score, label, _ in eval_pairs if score >= th_float and label == 0)
        fn = sum(1 for score, label, _ in eval_pairs if score < th_float and label == 1)
        tn = sum(1 for score, label, _ in eval_pairs if score < th_float and label == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0

        total_retrieved = sum(1 for r in results if r.aggregate_score >= th_float)
        pct_corpus = total_retrieved / total_corpus_count

        tm = ThresholdMetrics(
            threshold=th_float,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            specificity=specificity,
            accuracy=accuracy,
            total_retrieved=total_retrieved,
            pct_corpus=pct_corpus,
        )
        metrics_list.append(tm)

        # Seleciona o threshold que atende Recall >= minimum_recall e maximiza F1 / Precision
        if recall >= minimum_recall:
            if f1 > highest_f1_for_recall:
                highest_f1_for_recall = f1
                best_calibrated_threshold = th_float
                best_achieved_recall = recall
                best_achieved_precision = precision
                best_achieved_f1 = f1

    # Fallback se nenhum threshold atingir a meta estrita de recall
    if highest_f1_for_recall < 0 and metrics_list:
        max_rec_metric = max(metrics_list, key=lambda m: m.recall)
        best_calibrated_threshold = max_rec_metric.threshold
        best_achieved_recall = max_rec_metric.recall
        best_achieved_precision = max_rec_metric.precision
        best_achieved_f1 = max_rec_metric.f1_score

    # Calcula métricas de IR Ranking (P@k, MAP, nDCG@k)
    ir_metrics = compute_ir_ranking_metrics(results, gold_map, k=10)

    summary = EvaluationSummary(
        calibrated_threshold=best_calibrated_threshold,
        calibration_criterion=f"Mínimo Recall Target >= {minimum_recall:.2f}",
        minimum_recall_target=minimum_recall,
        achieved_recall=best_achieved_recall,
        achieved_precision=best_achieved_precision,
        achieved_f1=best_achieved_f1,
        total_gold_standard_samples=total_samples,
        total_relevant_samples=total_relevant,
        metrics_per_threshold=metrics_list,
        ranking_metrics=ir_metrics,
    )

    return summary, {}


def compute_ir_ranking_metrics(
    results: List[SearchResult],
    gold_map: Dict[str, int],
    k: int = 10
) -> IRRankingMetrics:
    """Calcula métricas formais de sistemas de Information Retrieval (P@k, R@k, MRR, MAP, nDCG@k)."""
    annotated_results = [r for r in results if r.paragraph_id in gold_map]
    if not annotated_results:
        return IRRankingMetrics(k=k, precision_at_k=0.0, recall_at_k=0.0, mrr=0.0, map_score=0.0, ndcg_at_k=0.0)

    top_k_res = annotated_results[:k]
    total_relevant = sum(1 for label in gold_map.values() if label == 1)

    rel_k = [gold_map[r.paragraph_id] for r in top_k_res]

    tp_k = sum(rel_k)
    precision_at_k = tp_k / len(top_k_res) if top_k_res else 0.0
    recall_at_k = tp_k / total_relevant if total_relevant > 0 else 0.0

    # Mean Reciprocal Rank (MRR)
    first_rel_rank = 0
    for idx, rel in enumerate(annotated_results, start=1):
        if gold_map[rel.paragraph_id] == 1:
            first_rel_rank = idx
            break
    mrr = (1.0 / first_rel_rank) if first_rel_rank > 0 else 0.0

    # Mean Average Precision (MAP)
    num_rel_seen = 0
    prec_sum = 0.0
    for idx, rel in enumerate(annotated_results, start=1):
        if gold_map[rel.paragraph_id] == 1:
            num_rel_seen += 1
            prec_sum += num_rel_seen / idx
    map_score = (prec_sum / total_relevant) if total_relevant > 0 else 0.0

    # nDCG@k
    dcg = sum((2**rel - 1) / np.log2(rank + 1) for rank, rel in enumerate(rel_k, start=1))

    # Ideal DCG
    ideal_rel = sorted([gold_map[r.paragraph_id] for r in annotated_results], reverse=True)[:k]
    idcg = sum((2**rel - 1) / np.log2(rank + 1) for rank, rel in enumerate(ideal_rel, start=1))

    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return IRRankingMetrics(
        k=k,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        mrr=mrr,
        map_score=map_score,
        ndcg_at_k=ndcg,
    )


def _generate_uncalibrated_summary(results: List[SearchResult], minimum_recall: float) -> EvaluationSummary:
    """Gera um sumário para quando não há dataset de validação humana anotado."""
    return EvaluationSummary(
        calibrated_threshold=0.50,
        calibration_criterion="Limiar Exploratório (Sem Gold Standard para Calibração)",
        minimum_recall_target=minimum_recall,
        achieved_recall=0.0,
        achieved_precision=0.0,
        achieved_f1=0.0,
        total_gold_standard_samples=0,
        total_relevant_samples=0,
        metrics_per_threshold=[],
        ranking_metrics=None,
    )
