"""
Mecanismo otimizado de recuperação semântica multidimensional multi-âncora por Cosine Similarity.
Suporta processamento vetorizado em batches/chunks para baixo consumo de memória RAM em grandes corpora (1.16M+ parágrafos).
"""

import time
import gc
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any, Callable, Union
import numpy as np
import psutil

from src.sld.models.search_result import SearchResult, Segment
from src.sld.semantic.embedding_service import EmbeddingService
from src.sld.semantic.vector_index import VectorIndex
from src.sld.semantic.semantic_reference import SemanticReferenceSet, SemanticAnchor
from config.settings import DEFAULT_SEMANTIC_BATCH_SIZE, DEFAULT_TOP_K_ANCHORS, DEFAULT_SEMANTIC_DTYPE


@dataclass
class SemanticSearchSummary:
    """Estatísticas descritivas e instrumentação de desempenho da busca semântica em lote."""
    total_paragraphs: int = 0
    retained_paragraphs: int = 0
    discarded_paragraphs: int = 0
    retention_rate: float = 0.0
    threshold_used: float = 0.50
    anchor_count: int = 0
    batch_size: int = DEFAULT_SEMANTIC_BATCH_SIZE
    top_k_anchors: int = DEFAULT_TOP_K_ANCHORS
    total_duration_seconds: float = 0.0
    avg_batch_duration_seconds: float = 0.0
    paragraphs_per_second: float = 0.0
    initial_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    final_memory_mb: float = 0.0
    mean_similarity_retained: float = 0.0
    median_similarity_retained: float = 0.0
    min_similarity_retained: float = 0.0
    max_similarity_retained: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Total de Parágrafos Analisados": self.total_paragraphs,
            "Parágrafos Retidos": self.retained_paragraphs,
            "Parágrafos Descartados": self.discarded_paragraphs,
            "Taxa de Retenção (%)": f"{self.retention_rate:.2f}%",
            "Limiar Semântico (θ_s)": round(self.threshold_used, 4),
            "Número de Sentenças-Âncoras": self.anchor_count,
            "Tamanho do Batch (Batch Size)": self.batch_size,
            "Top-K Âncoras": self.top_k_anchors,
            "Tempo Total (s)": round(self.total_duration_seconds, 3),
            "Tempo Médio por Batch (s)": round(self.avg_batch_duration_seconds, 4),
            "Vazão (parágrafos/s)": round(self.paragraphs_per_second, 1),
            "Uso Memória Inicial (MB)": round(self.initial_memory_mb, 1),
            "Pico de Memória (MB)": round(self.peak_memory_mb, 1),
            "Uso Memória Final (MB)": round(self.final_memory_mb, 1),
            "Similaridade Média (Retidos)": round(self.mean_similarity_retained, 4),
            "Similaridade Mediana (Retidos)": round(self.median_similarity_retained, 4),
            "Similaridade Mínima (Retidos)": round(self.min_similarity_retained, 4),
            "Similaridade Máxima (Retidos)": round(self.max_similarity_retained, 4),
        }


def perform_multi_anchor_search(
    vector_index: VectorIndex,
    embedding_service: EmbeddingService,
    reference_set: SemanticReferenceSet,
    aggregation_strategy: str = "maximum",
    threshold: float = 0.50,
    chunk_aggregation: str = "maximum",
    selected_article_ids: Optional[List[str]] = None,
    run_id: str = "exp_default",
    batch_size: int = DEFAULT_SEMANTIC_BATCH_SIZE,
    top_k_anchors: int = DEFAULT_TOP_K_ANCHORS,
    progress_callback: Optional[Callable[[int, int, int, int, int, int, float], None]] = None,
    return_summary: bool = False,
    only_retained: bool = False,
) -> Union[List[SearchResult], Tuple[List[SearchResult], SemanticSearchSummary]]:
    """
    Executa a busca semântica multi-âncora por Cosine Similarity em blocos/chunks.

    Fluxo de Execução Otimizado:
      1. Leitura incremental dos embeddings persistidos (via memory-mapping npy se disponível).
      2. Vetorização e normalização L2 das sentenças-âncoras (uma única vez).
      3. Iteração em lotes de tamanho `batch_size`.
      4. Produto matricial vetorizado: batch_embeddings @ anchor_embeddings.T em float32.
      5. Extração imediata do maior score por parágrafo S_i = max_j S(P_i, A_j).
      6. Aplicação do limiar semântico θ_s (R_i = I[S_i >= θ_s]).
      7. Descarte imediato da matriz intermediária de similaridade (batch_size x M).
      8. Atualização de estatísticas descritivas e progresso por batch.

    Parâmetros:
        vector_index: Índice vetorial contendo embeddings e segmentos.
        embedding_service: Serviço de modelos de embedding.
        reference_set: Conjunto de sentenças-âncoras de referência conceitual.
        aggregation_strategy: Estratégia de agregação ("maximum", "mean", "weighted_mean", "centroid").
        threshold: Limiar de retenção semântica (θ_s).
        chunk_aggregation: Agregação de múltiplos chunks por parágrafo ("maximum" ou "mean").
        selected_article_ids: Lista opcional de artigos a filtrar.
        run_id: Identificador da execução.
        batch_size: Tamanho do lote de processamento (default: 8192).
        top_k_anchors: Número de melhores âncoras a preservar por parágrafo (default: 1).
        progress_callback: Callback para feedback da barra de progresso por batch.
        return_summary: Se True, retorna a tupla (resultados, summary).
        only_retained: Se True, preserva em memória apenas os resultados com S_i >= θ_s.
    """
    process = psutil.Process()
    start_memory_mb = process.memory_info().rss / (1024 * 1024)
    peak_memory_mb = start_memory_mb
    start_time = time.time()

    if vector_index.embeddings is None or not vector_index.segments:
        empty_summary = SemanticSearchSummary(
            threshold_used=threshold,
            batch_size=batch_size,
            top_k_anchors=top_k_anchors,
            initial_memory_mb=start_memory_mb,
            peak_memory_mb=start_memory_mb,
            final_memory_mb=start_memory_mb,
        )
        return ([], empty_summary) if return_summary else []

    anchors = reference_set.anchors
    if not anchors:
        empty_summary = SemanticSearchSummary(
            threshold_used=threshold,
            batch_size=batch_size,
            top_k_anchors=top_k_anchors,
            initial_memory_mb=start_memory_mb,
            peak_memory_mb=start_memory_mb,
            final_memory_mb=start_memory_mb,
        )
        return ([], empty_summary) if return_summary else []

    # 1. Filtra índices válidos para a busca semântica
    valid_indices = []
    valid_segments: List[Segment] = []
    for idx, seg in enumerate(vector_index.segments):
        # Exclui apenas se explicitamente marcado como inválido ou deletado
        if getattr(seg, "status", "") in ["error", "deleted", "invalid_segment", "failed"]:
            continue
        if selected_article_ids and seg.article_id not in selected_article_ids:
            continue
        valid_indices.append(idx)
        valid_segments.append(seg)

    total_valid = len(valid_segments)
    if total_valid == 0:
        empty_summary = SemanticSearchSummary(
            threshold_used=threshold,
            anchor_count=len(anchors),
            batch_size=batch_size,
            top_k_anchors=top_k_anchors,
            initial_memory_mb=start_memory_mb,
            peak_memory_mb=start_memory_mb,
            final_memory_mb=start_memory_mb,
        )
        return ([], empty_summary) if return_summary else []

    # 2. Gera embeddings das sentenças-âncoras (M, D) normalizados L2 uma única vez por execução
    anchor_texts = [a.text for a in anchors]
    anchor_vecs = embedding_service.encode_queries(anchor_texts, normalize=True).astype(np.float32)  # (M, D)
    num_anchors = len(anchors)
    anchor_weights = {a.id: a.weight for a in anchors}

    # Vetor centroide se a estratégia for "centroid"
    centroid_vec = None
    if aggregation_strategy == "centroid":
        centroid_vec = np.mean(anchor_vecs, axis=0)
        c_norm = np.linalg.norm(centroid_vec)
        if c_norm > 0:
            centroid_vec = (centroid_vec / c_norm).astype(np.float32)

    # Mapeamento rápido para contextos adjacentes
    doc_segments_map: Dict[str, Dict[int, Segment]] = {}
    for seg in vector_index.segments:
        if seg.article_id not in doc_segments_map:
            doc_segments_map[seg.article_id] = {}
        doc_segments_map[seg.article_id][seg.segment_index_in_doc] = seg

    # 3. Processamento em Batches / Chunks
    candidate_results: List[SearchResult] = []
    retained_scores: List[float] = []
    retained_count = 0
    discarded_count = 0

    num_batches = (total_valid + batch_size - 1) // batch_size
    batch_durations: List[float] = []

    for batch_idx in range(num_batches):
        batch_start_time = time.time()
        start_i = batch_idx * batch_size
        end_i = min(start_i + batch_size, total_valid)
        b_indices = valid_indices[start_i:end_i]
        b_segments = valid_segments[start_i:end_i]

        # Extrai fatia de embeddings do lote atual
        batch_emb = vector_index.embeddings[b_indices]  # (B, D)

        # NOTA TÉCNICA: Verificação da precisão numérica e da Normalização L2
        # Garantimos que a representação seja float32 (evitando conversão para float64).
        # Os embeddings gerados pelo EmbeddingService (SentenceTransformers / Ollama)
        # já são produzidos com norma L2 unitária (|E(P_i)|_2 = 1.0).
        # Para evitar re-normalizações custosas a cada comparação em 1.16M de parágrafos,
        # checamos a norma L2 e aplicamos a normalização somente se necessário.
        if batch_emb.dtype != np.float32:
            batch_emb = batch_emb.astype(np.float32, copy=False)

        norm_sample = float(np.linalg.norm(batch_emb[0])) if len(batch_emb) > 0 else 1.0
        if not np.isclose(norm_sample, 1.0, atol=1e-3):
            norms = np.linalg.norm(batch_emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            batch_emb = batch_emb / norms

        # 4. Produto matricial vetorizado: batch_size x M (float32)
        # sim_matrix[i, j] = Cosine Similarity entre o parágrafo i do batch e a âncora j
        sim_matrix = np.dot(batch_emb, anchor_vecs.T)  # (B, M)

        centroid_sims_batch = None
        if centroid_vec is not None:
            centroid_sims_batch = np.dot(batch_emb, centroid_vec)  # (B,)

        # 5. Processamento dos resultados por parágrafo dentro do batch
        for idx_in_batch, seg in enumerate(b_segments):
            row_sims = sim_matrix[idx_in_batch]
            scores_per_anchor: Dict[str, float] = {
                anchors[j].id: float(row_sims[j]) for j in range(num_anchors)
            }
            best_anchor_idx = int(np.argmax(row_sims))
            best_score = float(row_sims[best_anchor_idx])
            best_anchor_id = anchors[best_anchor_idx].id
            best_anchor_text = anchors[best_anchor_idx].text

            if aggregation_strategy == "maximum":
                agg_score = best_score
            else:
                cent_val = float(centroid_sims_batch[idx_in_batch]) if centroid_sims_batch is not None else None
                agg_score = SemanticReferenceSet.aggregate(
                    anchor_sims=scores_per_anchor,
                    strategy=aggregation_strategy,
                    weights=anchor_weights,
                    centroid_sim=cent_val,
                )

            # Aplicação do limiar semântico θ_s
            is_selected = bool(agg_score >= threshold)

            if is_selected:
                retained_count += 1
                retained_scores.append(agg_score)
            else:
                discarded_count += 1

            # Se retido (ou se only_retained=False), constrói o SearchResult
            if is_selected or not only_retained:
                doc_map = doc_segments_map.get(seg.article_id, {})
                seg_idx = seg.segment_index_in_doc

                prev_seg = doc_map.get(seg_idx - 1)
                next_seg = doc_map.get(seg_idx + 1)

                context_before = prev_seg.text if prev_seg else None
                context_after = next_seg.text if next_seg else None

                page_range = (
                    str(seg.page_start)
                    if seg.page_start == seg.page_end
                    else f"{seg.page_start}-{seg.page_end}"
                )

                candidate_results.append(
                    SearchResult(
                        rank=0,
                        aggregate_score=agg_score,
                        article_id=seg.article_id,
                        paragraph_id=seg.paragraph_id,
                        paragraph_hash=seg.text_sha256,
                        title=seg.title,
                        authors=[],
                        source_pdf=seg.source_pdf,
                        section=seg.section,
                        subsection=seg.subsection,
                        page_range=page_range,
                        segment_id=seg.segment_id,
                        chunk_id=seg.chunk_id,
                        text=seg.text,
                        anchor_scores=scores_per_anchor,
                        best_anchor_id=best_anchor_id,
                        best_anchor_text=best_anchor_text,
                        context_before=context_before,
                        context_after=context_after,
                        markdown_path=seg.markdown_path,
                        run_id=run_id,
                        threshold_used=threshold,
                        selected=is_selected,
                    )
                )

        # 6. Descarte explícito das matrizes intermediárias do lote para liberar memória RAM
        del batch_emb, sim_matrix
        if centroid_sims_batch is not None:
            del centroid_sims_batch

        batch_duration = time.time() - batch_start_time
        batch_durations.append(batch_duration)

        # Monitoramento de memória
        current_mem = process.memory_info().rss / (1024 * 1024)
        if current_mem > peak_memory_mb:
            peak_memory_mb = current_mem

        # Callback de progresso atualizado por batch
        if progress_callback:
            progress_callback(
                end_i,
                total_valid,
                retained_count,
                discarded_count,
                batch_idx + 1,
                num_batches,
                batch_duration,
            )

    # 7. Consolidação de chunks se necessário
    consolidated_results = _consolidate_chunks_if_needed(candidate_results, chunk_aggregation)

    # 8. Ordenação decrescente por score agregado e atribuição de Ranks
    consolidated_results.sort(key=lambda r: r.aggregate_score, reverse=True)
    for rank, r in enumerate(consolidated_results, start=1):
        r.rank = rank

    # 9. Cálculo de Estatísticas Descritivas Finais
    total_time = time.time() - start_time
    final_memory_mb = process.memory_info().rss / (1024 * 1024)

    mean_sim = float(np.mean(retained_scores)) if retained_scores else 0.0
    median_sim = float(np.median(retained_scores)) if retained_scores else 0.0
    min_sim = float(np.min(retained_scores)) if retained_scores else 0.0
    max_sim = float(np.max(retained_scores)) if retained_scores else 0.0
    ret_rate = (retained_count / total_valid * 100.0) if total_valid > 0 else 0.0

    summary = SemanticSearchSummary(
        total_paragraphs=total_valid,
        retained_paragraphs=retained_count,
        discarded_paragraphs=discarded_count,
        retention_rate=ret_rate,
        threshold_used=threshold,
        anchor_count=num_anchors,
        batch_size=batch_size,
        top_k_anchors=top_k_anchors,
        total_duration_seconds=total_time,
        avg_batch_duration_seconds=float(np.mean(batch_durations)) if batch_durations else 0.0,
        paragraphs_per_second=(total_valid / total_time) if total_time > 0 else 0.0,
        initial_memory_mb=start_memory_mb,
        peak_memory_mb=peak_memory_mb,
        final_memory_mb=final_memory_mb,
        mean_similarity_retained=mean_sim,
        median_similarity_retained=median_sim,
        min_similarity_retained=min_sim,
        max_similarity_retained=max_sim,
    )

    if return_summary:
        return consolidated_results, summary
    return consolidated_results


def _consolidate_chunks_if_needed(
    results: List[SearchResult],
    chunk_aggregation: str = "maximum"
) -> List[SearchResult]:
    """
    Consolida múltiplos chunks de um mesmo parágrafo (ex: P0001_C01, P0001_C02)
    em um único resultado representativo por parágrafo.
    """
    grouped_by_para: Dict[Tuple[str, str], List[SearchResult]] = {}

    for r in results:
        key = (r.article_id, r.paragraph_id)
        if key not in grouped_by_para:
            grouped_by_para[key] = []
        grouped_by_para[key].append(r)

    consolidated: List[SearchResult] = []

    for key, chunk_list in grouped_by_para.items():
        if len(chunk_list) == 1:
            consolidated.append(chunk_list[0])
        else:
            if chunk_aggregation == "maximum":
                best_chunk = max(chunk_list, key=lambda c: c.aggregate_score)
                consolidated.append(best_chunk)
            else:  # "mean"
                avg_score = float(np.mean([c.aggregate_score for c in chunk_list]))
                best_chunk = max(chunk_list, key=lambda c: c.aggregate_score)
                best_chunk.aggregate_score = avg_score
                consolidated.append(best_chunk)
    return consolidated


def perform_semantic_search(
    query: str,
    vector_index: VectorIndex,
    embedding_service: EmbeddingService,
    top_k: int = 10,
    similarity_threshold: float = 0.0,
    selected_article_ids: Optional[List[str]] = None,
    batch_size: int = DEFAULT_SEMANTIC_BATCH_SIZE
) -> List[SearchResult]:
    """Alias mantendo compatibilidade com busca por texto único."""
    ref_set = SemanticReferenceSet(anchors=[SemanticAnchor(id="Q1", text=query)])
    results = perform_multi_anchor_search(
        vector_index=vector_index,
        embedding_service=embedding_service,
        reference_set=ref_set,
        aggregation_strategy="maximum",
        threshold=similarity_threshold,
        selected_article_ids=selected_article_ids,
        batch_size=batch_size,
    )
    return results[:top_k]


def compute_per_anchor_statistics(
    results: List[SearchResult],
    reference_set: SemanticReferenceSet,
    threshold: float = 0.50
) -> Any:
    """
    Calcula estatísticas descritivas completas para cada sentença-âncora do conjunto de referência.
    Retorna um pandas DataFrame com contagens de parágrafos, documentos únicos e métricas descritivas.
    """
    import pandas as pd
    rows = []
    if not results or not reference_set.anchors:
        return pd.DataFrame()

    total_paragraphs = len(results)
    total_docs = len(set(r.article_id for r in results))

    for anchor in reference_set.anchors:
        anc_id = anchor.id
        anc_text = anchor.text
        anc_cat = getattr(anchor, "description", "") or "Geral"

        scores = []
        doc_ids_at_th = set()
        p_ids_at_th = set()
        n_best = 0

        for r in results:
            s_val = r.anchor_scores.get(anc_id)
            if s_val is not None:
                scores.append(s_val)
                if s_val >= threshold:
                    doc_ids_at_th.add(r.article_id)
                    p_ids_at_th.add(r.paragraph_id)
            if r.best_anchor_id == anc_id:
                n_best += 1

        if scores:
            arr = np.array(scores)
            mean_s = float(np.mean(arr))
            median_s = float(np.median(arr))
            min_s = float(np.min(arr))
            max_s = float(np.max(arr))
            std_s = float(np.std(arr))
        else:
            mean_s = median_s = min_s = max_s = std_s = 0.0

        rows.append({
            "Âncora ID": anc_id,
            "Categoria": anc_cat,
            "Texto da Âncora": anc_text,
            "Score Médio": round(mean_s, 4),
            "Score Mediano": round(median_s, 4),
            "Score Mínimo": round(min_s, 4),
            "Score Máximo": round(max_s, 4),
            "Desvio Padrão": round(std_s, 4),
            "Parágrafos (≥ θ_s)": len(p_ids_at_th),
            "% Parágrafos": f"{(len(p_ids_at_th) / total_paragraphs * 100):.1f}%" if total_paragraphs > 0 else "0.0%",
            "Documentos Únicos (≥ θ_s)": len(doc_ids_at_th),
            "% Documentos Únicos": f"{(len(doc_ids_at_th) / total_docs * 100):.1f}%" if total_docs > 0 else "0.0%",
            "Melhor Âncora (A*)": n_best
        })

    return pd.DataFrame(rows)
