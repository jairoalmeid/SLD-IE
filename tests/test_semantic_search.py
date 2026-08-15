"""
Suíte completa de testes unitários e de equivalência para a busca semântica multi-âncora por Cosine Similarity.
Valida equivalência numérica, independência de batch size, variação de âncoras, filtragem por limiar e consumo de memória.
"""

from typing import List
from pathlib import Path
import numpy as np
import pytest
from src.sld.models.search_result import Segment
from src.sld.semantic.vector_index import VectorIndex
from src.sld.semantic.semantic_reference import SemanticReferenceSet, SemanticAnchor
from src.sld.semantic.semantic_search import perform_multi_anchor_search, SemanticSearchSummary


class MockEmbeddingService:
    """Mock determinístico de alta performance para testes de grande escala."""

    def __init__(self, dim: int = 128):
        self.model_name = "mock-model"
        self.device = "cpu"
        self.dim = dim

    def encode_queries(self, queries: List[str], normalize: bool = True) -> np.ndarray:
        vecs = []
        for idx, q in enumerate(queries):
            rng = np.random.RandomState(idx + 42)
            v = rng.randn(self.dim).astype(np.float32)
            if normalize:
                norm = np.linalg.norm(v)
                v = v / (norm if norm > 0 else 1.0)
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)


def _generate_synthetic_corpus(num_paragraphs: int = 1000, dim: int = 128):
    """Gera um corpus sintético de parágrafos e embeddings normalizados."""
    segments = []
    rng = np.random.RandomState(123)
    raw_embeddings = rng.randn(num_paragraphs, dim).astype(np.float32)
    norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    embeddings = raw_embeddings / norms

    for i in range(num_paragraphs):
        segments.append(
            Segment(
                segment_id=f"S_{i:05d}",
                article_id=f"A_{i // 10:04d}",
                paragraph_id=f"P_{i:05d}",
                source_pdf=f"doc_{i // 10}.pdf",
                markdown_path=f"/tmp/doc_{i // 10}.md",
                title=f"Título do Artigo {i // 10}",
                section="Métodos",
                subsection="",
                page_start=1,
                page_end=1,
                text=f"Parágrafo de teste de alta escala número {i}.",
                text_sha256=f"hash_{i}",
                status="valid_paragraph",
                segment_index_in_doc=i % 10
            )
        )

    v_index = VectorIndex(Path("/tmp/synthetic_index_dir"))
    v_index.embeddings = embeddings
    v_index.segments = segments
    return v_index, embeddings, segments


def test_multi_anchor_cosine_similarity_and_ranking():
    """Testa busca semântica básica e ordenação por similaridade agregada."""
    segments = [
        Segment(
            segment_id="SLD-1_P0001", article_id="SLD-1", paragraph_id="P0001", source_pdf="artigo1.pdf",
            markdown_path="/tmp/artigo1.md", title="Artigo Biologia", section="Introdução", subsection="",
            page_start=1, page_end=1, text="Estudo sobre biologia celular e genômica.", text_sha256="hash1",
            status="valid_paragraph", segment_index_in_doc=0
        ),
        Segment(
            segment_id="SLD-2_P0001", article_id="SLD-2", paragraph_id="P0001", source_pdf="artigo2.pdf",
            markdown_path="/tmp/artigo2.md", title="Artigo Computação", section="Introdução", subsection="",
            page_start=1, page_end=1, text="Estudo sobre sistemas distribuídos e computação.", text_sha256="hash2",
            status="valid_paragraph", segment_index_in_doc=0
        )
    ]

    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float32)

    v_index = VectorIndex(Path("/tmp/test_index_dir"))
    v_index.embeddings = embeddings
    v_index.segments = segments

    mock_emb = MockEmbeddingService(dim=3)
    # Sobrescreve para valores conhecidos
    mock_emb.encode_queries = lambda q_list, normalize=True: np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    ref_set = SemanticReferenceSet(anchors=[
        SemanticAnchor(id="Q1", text="Pesquisa em biologia", weight=1.0)
    ])

    results = perform_multi_anchor_search(
        vector_index=v_index,
        embedding_service=mock_emb,
        reference_set=ref_set,
        aggregation_strategy="maximum",
        threshold=0.0
    )

    assert len(results) == 2
    assert results[0].article_id == "SLD-1"
    assert results[0].rank == 1
    assert pytest.approx(results[0].aggregate_score, 0.01) == 1.0
    assert results[0].best_anchor_id == "Q1"


def test_1_numerical_equivalence():
    """Teste 1: Equivalência numérica entre matriz ingênua e busca por batches."""
    v_index, embeddings, segments = _generate_synthetic_corpus(num_paragraphs=500, dim=64)
    mock_emb = MockEmbeddingService(dim=64)

    ref_set = SemanticReferenceSet(anchors=[
        SemanticAnchor(id="Q1", text="Conceito 1"),
        SemanticAnchor(id="Q2", text="Conceito 2"),
        SemanticAnchor(id="Q3", text="Conceito 3"),
    ])

    anchor_vecs = mock_emb.encode_queries([a.text for a in ref_set.anchors], normalize=True)
    sim_naive = np.dot(embeddings, anchor_vecs.T)
    naive_max_scores = np.max(sim_naive, axis=1)

    results_batched = perform_multi_anchor_search(
        vector_index=v_index,
        embedding_service=mock_emb,
        reference_set=ref_set,
        aggregation_strategy="maximum",
        threshold=0.0,
        batch_size=64,
        only_retained=False
    )

    res_dict = {r.paragraph_id: r.aggregate_score for r in results_batched}
    for idx, seg in enumerate(segments):
        expected_score = float(naive_max_scores[idx])
        actual_score = res_dict[seg.paragraph_id]
        assert pytest.approx(actual_score, abs=1e-5) == expected_score


def test_2_batch_size_independence():
    """Teste 2: Os resultados de similaridade devem ser idênticos independentemente do batch_size."""
    v_index, _, _ = _generate_synthetic_corpus(num_paragraphs=1200, dim=64)
    mock_emb = MockEmbeddingService(dim=64)

    ref_set = SemanticReferenceSet(anchors=[
        SemanticAnchor(id="Q1", text="Frase âncora principal"),
        SemanticAnchor(id="Q2", text="Frase âncora secundária")
    ])

    batch_sizes = [128, 256, 512, 1024]
    results_by_batch = {}

    for b_size in batch_sizes:
        res = perform_multi_anchor_search(
            vector_index=v_index,
            embedding_service=mock_emb,
            reference_set=ref_set,
            aggregation_strategy="maximum",
            threshold=0.30,
            batch_size=b_size,
            only_retained=False
        )
        results_by_batch[b_size] = res

    base_results = results_by_batch[128]
    for b_size in [256, 512, 1024]:
        comp_results = results_by_batch[b_size]
        assert len(comp_results) == len(base_results)
        for r_base, r_comp in zip(base_results, comp_results):
            assert r_base.paragraph_id == r_comp.paragraph_id
            assert pytest.approx(r_base.aggregate_score, abs=1e-5) == r_comp.aggregate_score
            assert r_base.selected == r_comp.selected


def test_3_anchor_counts():
    """Teste 3: O algoritmo deve funcionar de forma genérica para 5, 10, 25 e 50 âncoras."""
    v_index, _, _ = _generate_synthetic_corpus(num_paragraphs=200, dim=64)
    mock_emb = MockEmbeddingService(dim=64)

    for n_anchors in [5, 10, 25, 50]:
        anchors = [SemanticAnchor(id=f"Q{i+1}", text=f"Âncora de teste {i+1}") for i in range(n_anchors)]
        ref_set = SemanticReferenceSet(anchors=anchors)

        results = perform_multi_anchor_search(
            vector_index=v_index,
            embedding_service=mock_emb,
            reference_set=ref_set,
            aggregation_strategy="maximum",
            threshold=0.0,
            batch_size=64
        )

        assert len(results) == 200
        assert results[0].best_anchor_id in [a.id for a in anchors]


def test_4_threshold_filtering():
    """Teste 4: Valida retenção/descarte com limiares abaixo, iguais e acima do θ_s."""
    v_index, _, _ = _generate_synthetic_corpus(num_paragraphs=500, dim=64)
    mock_emb = MockEmbeddingService(dim=64)
    ref_set = SemanticReferenceSet(anchors=[SemanticAnchor(id="Q1", text="Teste Limiar")])

    res_all, sum_all = perform_multi_anchor_search(
        vector_index=v_index, embedding_service=mock_emb, reference_set=ref_set,
        threshold=-1.0, return_summary=True, only_retained=True
    )
    assert len(res_all) == 500
    assert sum_all.retained_paragraphs == 500
    assert sum_all.discarded_paragraphs == 0

    res_strict, sum_strict = perform_multi_anchor_search(
        vector_index=v_index, embedding_service=mock_emb, reference_set=ref_set,
        threshold=0.99, return_summary=True, only_retained=True
    )
    assert len(res_strict) < 500
    assert sum_strict.discarded_paragraphs > 0
    assert sum_strict.retained_paragraphs == len(res_strict)


def test_5_memory_efficiency_and_summary():
    """Teste 5: Instrumentação de memória e resumo estatístico."""
    v_index, _, _ = _generate_synthetic_corpus(num_paragraphs=2000, dim=128)
    mock_emb = MockEmbeddingService(dim=128)
    ref_set = SemanticReferenceSet(anchors=[SemanticAnchor(id="Q1", text="Teste Memória")])

    results, summary = perform_multi_anchor_search(
        vector_index=v_index,
        embedding_service=mock_emb,
        reference_set=ref_set,
        threshold=0.10,
        batch_size=512,
        return_summary=True
    )

    assert isinstance(summary, SemanticSearchSummary)
    assert summary.total_paragraphs == 2000
    assert summary.batch_size == 512
    assert summary.peak_memory_mb >= summary.initial_memory_mb
    assert summary.paragraphs_per_second > 0
    assert 0.0 <= summary.retention_rate <= 100.0


def test_vector_index_invalidation(tmp_path):
    """Testa invalidação do índice vetorial quando a configuração muda."""
    vector_index = VectorIndex(tmp_path)

    config_v1 = {
        "embedding_model": "model-v1",
        "min_words": 10,
        "max_characters": 1500,
        "long_text_strategy": "chunk"
    }

    segments = [
        Segment(
            segment_id="S1", article_id="A1", paragraph_id="P01", source_pdf="a.pdf",
            markdown_path="a.md", title="T", section="S", subsection="", page_start=1,
            page_end=1, text="Texto válido para o teste.", text_sha256="h1", status="valid_paragraph"
        )
    ]
    embeddings = np.array([[0.1, 0.2]], dtype=np.float32)

    vector_index.build_and_save(embeddings, segments, "model-v1", "cpu", config_v1)
    assert vector_index.is_valid(config_v1) is True

    config_v2 = config_v1.copy()
    config_v2["embedding_model"] = "model-v2"
    assert vector_index.is_valid(config_v2) is False
