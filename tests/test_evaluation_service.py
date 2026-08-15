"""
Testes unitários para calibração de limiar por Mínimo Recall, Gold Standard e métricas IR.
"""

from pathlib import Path
import pytest
from src.sld.models.search_result import SearchResult
from src.sld.models.evaluation import GoldStandardAnnotation
from src.sld.semantic.evaluation_service import (
    save_gold_standard,
    load_gold_standard,
    calibrate_threshold_by_minimum_recall,
    compute_ir_ranking_metrics,
)


def test_save_and_load_gold_standard(tmp_path):
    """Testa persistência e leitura de anotações humanas do Gold Standard."""
    gs_file = tmp_path / "gold_standard.jsonl"

    annotations = [
        GoldStandardAnnotation(paragraph_id="P0001", human_label=1, annotator="doutorando", article_id="SLD-001"),
        GoldStandardAnnotation(paragraph_id="P0002", human_label=0, annotator="doutorando", article_id="SLD-001"),
    ]

    save_gold_standard(annotations, gs_file)
    loaded = load_gold_standard(gs_file)

    assert len(loaded) == 2
    assert loaded[0].paragraph_id == "P0001"
    assert loaded[0].human_label == 1
    assert loaded[1].human_label == 0


def test_calibrate_threshold_by_minimum_recall():
    """Testa seleção empírica de limiar satisfazendo a restrição Recall >= 0.90."""
    results = [
        SearchResult(rank=1, aggregate_score=0.85, article_id="A1", paragraph_id="P0001", paragraph_hash="h1", title="T1", authors=[], source_pdf="a.pdf", section="S", subsection="", page_range="1", segment_id="S1", chunk_id=None, text="texto 1"),
        SearchResult(rank=2, aggregate_score=0.75, article_id="A1", paragraph_id="P0002", paragraph_hash="h2", title="T1", authors=[], source_pdf="a.pdf", section="S", subsection="", page_range="1", segment_id="S2", chunk_id=None, text="texto 2"),
        SearchResult(rank=3, aggregate_score=0.60, article_id="A2", paragraph_id="P0003", paragraph_hash="h3", title="T2", authors=[], source_pdf="b.pdf", section="S", subsection="", page_range="1", segment_id="S3", chunk_id=None, text="texto 3"),
        SearchResult(rank=4, aggregate_score=0.30, article_id="A2", paragraph_id="P0004", paragraph_hash="h4", title="T2", authors=[], source_pdf="b.pdf", section="S", subsection="", page_range="1", segment_id="S4", chunk_id=None, text="texto 4"),
    ]

    gold_standard = [
        GoldStandardAnnotation(paragraph_id="P0001", human_label=1),
        GoldStandardAnnotation(paragraph_id="P0002", human_label=1),
        GoldStandardAnnotation(paragraph_id="P0003", human_label=0),
        GoldStandardAnnotation(paragraph_id="P0004", human_label=0),
    ]

    summary, _ = calibrate_threshold_by_minimum_recall(results, gold_standard, minimum_recall=0.90)

    assert summary.calibrated_threshold <= 0.75
    assert summary.achieved_recall >= 0.90
    assert summary.total_gold_standard_samples == 4
    assert summary.total_relevant_samples == 2


def test_compute_ir_ranking_metrics():
    """Testa cálculo de métricas de ranking IR (P@k, R@k, MRR, nDCG@k)."""
    results = [
        SearchResult(rank=1, aggregate_score=0.9, article_id="A1", paragraph_id="P0001", paragraph_hash="h1", title="T", authors=[], source_pdf="a.pdf", section="S", subsection="", page_range="1", segment_id="S1", chunk_id=None, text="t1"),
        SearchResult(rank=2, aggregate_score=0.8, article_id="A1", paragraph_id="P0002", paragraph_hash="h2", title="T", authors=[], source_pdf="a.pdf", section="S", subsection="", page_range="1", segment_id="S2", chunk_id=None, text="t2"),
    ]
    gold_map = {"P0001": 1, "P0002": 0}

    metrics = compute_ir_ranking_metrics(results, gold_map, k=2)

    assert metrics.precision_at_k == 0.5
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 1.0
    assert metrics.ndcg_at_k > 0.0
