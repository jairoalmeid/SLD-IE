"""
Testes unitários rigorosos para o Índice de Recuperação do Corpus Refinado (FAISS + Parquet).
Cobre os 7 testes obrigatórios de integridade, correspondência, persistência, classes e IDs.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import faiss

from src.sld.models.classification import ParagraphRecord
from src.sld.rag_index.models import RAGIndexConfig
from src.sld.rag_index.builder import (
    RAGIndexBuilder,
    compute_corpus_distribution_stats,
    compute_coverage_stats,
)
from src.sld.rag_index.retriever import RAGIndexRetriever


class MockEmbeddingService:
    """Mock do EmbeddingService para testar consultas vetoriais sem baixar modelos externos."""

    def __init__(self, dim=384):
        self.dim = dim

    def encode_queries(self, queries, normalize=True):
        np.random.seed(42)
        vecs = np.random.randn(len(queries), self.dim).astype(np.float32)
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs


@pytest.fixture
def sample_corpus_data():
    """Gera dados sintéticos de corpus com parágrafos multilabel, Classe 0 e duplicações."""
    np.random.seed(123)
    num_paras = 20
    dim = 64

    records = []
    # Parágrafo 1: Definição (Classe 1)
    records.append(
        ParagraphRecord(
            paragraph_id="P0001",
            article_id="ART_01",
            text="Resiliência é a capacidade de um sistema absorver impactos.",
            status="MODEL_RELEVANT",
            semantic_score=0.85,
            predicted_labels=["definition"],
        )
    )
    # Parágrafo 2: Multilabel (Classes 1, 4 e 5)
    records.append(
        ParagraphRecord(
            paragraph_id="P0002",
            article_id="ART_01",
            text="A governança adaptativa atua como mecanismo de mediação e possui flexibilidade.",
            status="MODEL_RELEVANT",
            semantic_score=0.92,
            predicted_labels=["definition", "causal_relation", "property"],
        )
    )
    # Parágrafo 3: Classe 0 — Não Relevante (MODEL_NOT_RELEVANT, sem labels)
    records.append(
        ParagraphRecord(
            paragraph_id="P0003",
            article_id="ART_01",
            text="Os autores agradecem o apoio financeiro do CNPq.",
            status="MODEL_NOT_RELEVANT",
            semantic_score=0.15,
            predicted_labels=[],
        )
    )
    # Parágrafo 4: Fator e Dimensão (Classes 2 e 3)
    records.append(
        ParagraphRecord(
            paragraph_id="P0004",
            article_id="ART_02",
            text="Fatores determinantes incluem a liderança e dimensões operacionais.",
            status="MODEL_RELEVANT",
            semantic_score=0.78,
            predicted_labels=["determinant", "type_dimension"],
        )
    )
    # Parágrafo 5: Duplicata de P0002 para testar deduplicação
    records.append(
        ParagraphRecord(
            paragraph_id="P0002",
            article_id="ART_01",
            text="A governança adaptativa atua como mecanismo de mediação e possui flexibilidade.",
            status="MODEL_RELEVANT",
            semantic_score=0.92,
            predicted_labels=["definition", "causal_relation", "property"],
        )
    )
    # Parágrafo 6: Outro não relevante
    records.append(
        ParagraphRecord(
            paragraph_id="P0006",
            article_id="ART_03",
            text="Tabela 1: Resumo das variáveis.",
            status="MODEL_NOT_RELEVANT",
            semantic_score=0.20,
            predicted_labels=[],
        )
    )

    # Gera matriz de embeddings normalizados correspondente
    embeddings_matrix = np.random.randn(len(records), dim).astype(np.float32)
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    embeddings_matrix = embeddings_matrix / norms

    return records, embeddings_matrix, dim


def test_1_correspondence(tmp_path, sample_corpus_data):
    """
    Teste 1 — Correspondência:
    Cada vetor deve possuir exatamente um registro correspondente no metadata.parquet.
    """
    records, embeddings, dim = sample_corpus_data
    builder = RAGIndexBuilder(output_dir=tmp_path, config=RAGIndexConfig(dimension=dim, index_type="FlatIP"))

    faiss_p, parquet_p, manifest_p, zip_p, stats, _ = builder.build(
        corpus_records=records,
        embeddings_matrix=embeddings,
    )

    index = faiss.read_index(str(faiss_p))
    df = pd.read_parquet(parquet_p)

    assert index.ntotal == len(df), "Número de vetores no FAISS difere do número de linhas no Parquet."
    assert list(df["faiss_id"]) == list(range(len(df))), "Os faiss_ids devem corresponder exatamente aos índices 0 a N-1."


def test_2_persistence_and_query_equivalence(tmp_path, sample_corpus_data):
    """
    Teste 2 — Persistência:
    Criar o índice, salvar, descarregar da memória, carregar novamente e executar a mesma consulta.
    Os resultados devem permanecer equivalentes.
    """
    records, embeddings, dim = sample_corpus_data
    builder = RAGIndexBuilder(output_dir=tmp_path, config=RAGIndexConfig(dimension=dim, index_type="HNSW", M=16))

    faiss_p, parquet_p, manifest_p, zip_p, stats, _ = builder.build(
        corpus_records=records,
        embeddings_matrix=embeddings,
    )

    # Carrega via RAGIndexRetriever
    retriever = RAGIndexRetriever()
    retriever.load_from_dir(tmp_path / "rag_index")

    mock_emb = MockEmbeddingService(dim=dim)
    query_res_1 = retriever.query("resiliência de sistemas", embedding_service=mock_emb, top_k=3)

    # Simula descarga da memória e recarregamento a partir do ZIP
    retriever_from_zip = RAGIndexRetriever()
    retriever_from_zip.load_from_zip(zip_p)

    query_res_2 = retriever_from_zip.query("resiliência de sistemas", embedding_service=mock_emb, top_k=3)

    assert len(query_res_1) == len(query_res_2)
    for r1, r2 in zip(query_res_1, query_res_2):
        assert r1.faiss_id == r2.faiss_id
        assert r1.paragraph_id == r2.paragraph_id
        assert np.isclose(r1.score, r2.score, atol=1e-4)


def test_3_class_0_exclusion(tmp_path, sample_corpus_data):
    """
    Teste 3 — Classes:
    Verificar que Classe 0 (Não relevante) não foi incluída no índice.
    """
    records, embeddings, dim = sample_corpus_data
    builder = RAGIndexBuilder(output_dir=tmp_path, config=RAGIndexConfig(dimension=dim))

    faiss_p, parquet_p, _, _, stats, _ = builder.build(
        corpus_records=records,
        embeddings_matrix=embeddings,
    )

    df = pd.read_parquet(parquet_p)

    # Parágrafos P0003 e P0006 pertencem à Classe 0 (MODEL_NOT_RELEVANT)
    assert "P0003" not in df["paragraph_id"].values
    assert "P0006" not in df["paragraph_id"].values


def test_4_multilabel_deduplication(tmp_path, sample_corpus_data):
    """
    Teste 4 — Multilabel e Deduplicação:
    Verificar que um parágrafo pertencente a múltiplas classes (P0002) é indexado apenas uma vez.
    """
    records, embeddings, dim = sample_corpus_data
    builder = RAGIndexBuilder(output_dir=tmp_path, config=RAGIndexConfig(dimension=dim))

    faiss_p, parquet_p, _, _, stats, _ = builder.build(
        corpus_records=records,
        embeddings_matrix=embeddings,
    )

    df = pd.read_parquet(parquet_p)

    # P0002 deve aparecer exatamente uma vez
    p0002_rows = df[df["paragraph_id"] == "P0002"]
    assert len(p0002_rows) == 1, "Parágrafo multilabel P0002 foi duplicado no índice."

    # Verifica se as classes multilabel foram devidamente atribuídas
    row = p0002_rows.iloc[0]
    assert bool(row["class_1"]) is True   # definition
    assert bool(row["class_2"]) is False  # determinant
    assert bool(row["class_3"]) is False  # type_dimension
    assert bool(row["class_4"]) is True   # causal_relation
    assert bool(row["class_5"]) is True   # property


def test_5_embeddings_not_recomputed(tmp_path, sample_corpus_data):
    """
    Teste 5 — Embeddings:
    Garantir que nenhum embedding seja recalculado durante a criação do índice
    (utiliza fatias da matriz existente).
    """
    records, embeddings, dim = sample_corpus_data

    # Altera propositalmente uma linha específica para identificação única
    embeddings[1, :] = 0.5  # Modifica vetor de P0002
    embeddings[1, :] /= np.linalg.norm(embeddings[1, :])

    builder = RAGIndexBuilder(output_dir=tmp_path, config=RAGIndexConfig(dimension=dim, index_type="FlatIP"))
    faiss_p, parquet_p, _, _, _, _ = builder.build(
        corpus_records=records,
        embeddings_matrix=embeddings,
    )

    index = faiss.read_index(str(faiss_p))
    df = pd.read_parquet(parquet_p)

    # Identifica a linha correspondente a P0002 no índice FAISS
    p0002_faiss_id = df[df["paragraph_id"] == "P0002"].iloc[0]["faiss_id"]

    # Reconstitui vetor do FAISS para verificar se corresponde ao vetor da matriz
    reconstructed = index.reconstruct(int(p0002_faiss_id))
    assert np.allclose(reconstructed, embeddings[1, :], atol=1e-4)


def test_6_id_mapping(tmp_path, sample_corpus_data):
    """
    Teste 6 — IDs:
    Verificar que o faiss_id recuperado retorna o paragraph_id correto.
    """
    records, embeddings, dim = sample_corpus_data
    builder = RAGIndexBuilder(output_dir=tmp_path, config=RAGIndexConfig(dimension=dim, index_type="FlatIP"))

    faiss_p, parquet_p, _, _, _, _ = builder.build(
        corpus_records=records,
        embeddings_matrix=embeddings,
    )

    df = pd.read_parquet(parquet_p)

    for idx, row in df.iterrows():
        faiss_id = row["faiss_id"]
        expected_p_id = row["paragraph_id"]
        assert df.loc[df["faiss_id"] == faiss_id, "paragraph_id"].iloc[0] == expected_p_id


def test_7_integrity_checks(tmp_path, sample_corpus_data):
    """
    Teste 7 — Integridade Geral:
    FAISS ntotal == linhas do metadata == parágrafos únicos indexados.
    """
    records, embeddings, dim = sample_corpus_data
    builder = RAGIndexBuilder(output_dir=tmp_path, config=RAGIndexConfig(dimension=dim))

    faiss_p, parquet_p, manifest_p, zip_p, stats, manifest = builder.build(
        corpus_records=records,
        embeddings_matrix=embeddings,
        total_original_articles=5,
    )

    index = faiss.read_index(str(faiss_p))
    df = pd.read_parquet(parquet_p)

    assert index.ntotal == len(df) == stats.total_vectors == 3
    assert stats.unique_paragraphs == 3
    assert stats.represented_documents == 2
    assert zip_p.exists() and zip_p.stat().st_size > 0
    assert manifest_p.exists()
    assert manifest.indexed_paragraphs == 3


def test_8_min_max_score_threshold_filter(tmp_path, sample_corpus_data):
    """
    Teste 8 — Filtragem por Limiar Mínimo e Máximo de Similaridade:
    Garante que os resultados retornados estejam dentro do intervalo [min_score, max_score].
    """
    records, embeddings, dim = sample_corpus_data
    builder = RAGIndexBuilder(output_dir=tmp_path, config=RAGIndexConfig(dimension=dim))

    builder.build(
        corpus_records=records,
        embeddings_matrix=embeddings,
    )

    retriever = RAGIndexRetriever()
    retriever.load_from_dir(tmp_path / "rag_index")

    mock_emb = MockEmbeddingService(dim=dim)

    # Consulta sem restrições
    all_res = retriever.query("resiliência de sistemas", embedding_service=mock_emb, top_k=10)
    assert len(all_res) > 0

    scores = [r.score for r in all_res]
    median_score = float(np.median(scores))

    # Filtra com min_score acima da mediana
    filtered_min = retriever.query("resiliência de sistemas", embedding_service=mock_emb, top_k=10, min_score=median_score)
    for r in filtered_min:
        assert r.score >= median_score

    # Filtra com max_score abaixo da mediana
    filtered_max = retriever.query("resiliência de sistemas", embedding_service=mock_emb, top_k=10, max_score=median_score)
    for r in filtered_max:
        assert r.score <= median_score

