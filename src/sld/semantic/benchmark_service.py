"""
Serviço de benchmark comparativo entre múltiplos modelos de Sentence Embeddings.
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import psutil
from src.sld.models.evaluation import GoldStandardAnnotation, EvaluationSummary
from src.sld.models.search_result import Segment, SearchResult
from src.sld.semantic.embedding_service import EmbeddingService
from src.sld.semantic.vector_index import VectorIndex
from src.sld.semantic.semantic_reference import SemanticReferenceSet
from src.sld.semantic.semantic_search import perform_multi_anchor_search
from src.sld.semantic.evaluation_service import calibrate_threshold_by_minimum_recall


@dataclass
class ModelBenchmarkResult:
    """Resultado do benchmark empírico para um modelo de embeddings."""
    model_name: str
    embedding_dim: int
    device: str
    inference_time_seconds: float
    paragraphs_per_second: float
    memory_mb: float
    best_threshold: float
    recall: float
    precision: float
    f1_score: float
    ndcg_at_10: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Modelo": self.model_name,
            "Dimensão": self.embedding_dim,
            "Dispositivo": self.device.upper(),
            "Tempo Inferência (s)": round(self.inference_time_seconds, 2),
            "Parágrafos / seg": round(self.paragraphs_per_second, 1),
            "Memória (MB)": round(self.memory_mb, 1),
            "Limiar Calibrado": round(self.best_threshold, 4),
            "Recall": round(self.recall, 4),
            "Precision": round(self.precision, 4),
            "F1-Score": round(self.f1_score, 4),
            "nDCG@10": round(self.ndcg_at_10, 4),
        }


def run_model_benchmark(
    models_to_compare: List[str],
    segments: List[Segment],
    gold_standard: List[GoldStandardAnnotation],
    reference_set: SemanticReferenceSet,
    minimum_recall: float = 0.90,
    aggregation_strategy: str = "maximum",
    batch_size: int = 32
) -> List[ModelBenchmarkResult]:
    """
    Executa o benchmark empírico comparativo entre múltiplos modelos exatamente sobre o mesmo Gold Standard.
    """
    benchmark_results: List[ModelBenchmarkResult] = []
    valid_segments = [s for s in segments if s.status == "valid_paragraph"]

    if not valid_segments or not gold_standard:
        return []

    process = psutil.Process()

    for model_name in models_to_compare:
        start_mem = process.memory_info().rss / (1024 * 1024)
        start_time = time.time()

        emb_service = EmbeddingService(model_name=model_name)
        texts = [s.text for s in valid_segments]

        # Gera embeddings em batch
        embeddings = emb_service.encode(texts, batch_size=batch_size, normalize=True)
        infer_time = time.time() - start_time
        mem_used = (process.memory_info().rss / (1024 * 1024)) - start_mem

        para_per_sec = len(valid_segments) / infer_time if infer_time > 0 else 0.0

        # Constrói índice em memória
        v_index = VectorIndex(Path("/tmp/benchmark_temp_index"))
        v_index.embeddings = embeddings
        v_index.segments = valid_segments

        # Busca semântica multi-âncora
        results = perform_multi_anchor_search(
            vector_index=v_index,
            embedding_service=emb_service,
            reference_set=reference_set,
            aggregation_strategy=aggregation_strategy,
            threshold=0.0,
            run_id="benchmark"
        )

        # Calibração e métricas
        eval_summary, _ = calibrate_threshold_by_minimum_recall(
            results=results,
            gold_standard=gold_standard,
            minimum_recall=minimum_recall
        )

        ndcg = eval_summary.ranking_metrics.ndcg_at_k if eval_summary.ranking_metrics else 0.0

        benchmark_results.append(
            ModelBenchmarkResult(
                model_name=model_name,
                embedding_dim=emb_service.get_vector_dimension(),
                device=emb_service.device,
                inference_time_seconds=infer_time,
                paragraphs_per_second=para_per_sec,
                memory_mb=max(0.0, mem_used),
                best_threshold=eval_summary.calibrated_threshold,
                recall=eval_summary.achieved_recall,
                precision=eval_summary.achieved_precision,
                f1_score=eval_summary.achieved_f1,
                ndcg_at_10=ndcg,
            )
        )

    return benchmark_results
