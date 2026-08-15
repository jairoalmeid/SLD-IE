"""
Testes unitários e de integração para o Módulo de Persistência Completa de Sessão e Desacoplamento de Memória.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.sld.corpus.analysis_project import AnalysisProject
from src.sld.corpus.session_manager import save_full_session_state, restore_full_session_state
from src.sld.models.article import ArticleMetadata, ProcessedArticle
from src.sld.models.classification import ParagraphRecord, AnnotationRecord
from src.sld.models.search_result import Segment
from src.sld.annotation.annotation_service import AnnotationService
from src.sld.semantic.vector_index import VectorIndex
from src.sld.classification.baseline_classifier import MultilabelLogisticClassifier
from src.sld.rag_index import (
    RAGIndexBuilder,
    RAGIndexRetriever,
    RAGIndexConfig,
    IndexStats,
)


class MockSessionState:
    """Mock do st.session_state para testes de persistência e restauração."""
    def __init__(self):
        self.run_id = "test_run_123"
        self.completed_steps = []
        self.config = {"embedding_model": "all-MiniLM-L6-v2", "similarity_threshold": 0.65}
        self.funnel_counts = {"1_Ingestao": {"n_in": 10, "n_out": 10, "duration_sec": 1.0}}
        self.articles_records = []
        self.corpus_records = []
        self.embeddings_matrix = None
        self.semantic_reference = None
        self.semantic_candidates = []
        self.semantic_scores_map = {}
        self.gold_annotations = []
        self.logistic_classifier = None
        self.optimal_thresholds = None
        self.classified_records = []
        self.rag_index_stats = None
        self.rag_index_manifest = None
        self.rag_index_zip_path = None
        self.rag_retriever = None


def test_session_persistence_roundtrip(tmp_path: Path):
    """Testa o ciclo completo de salvar a sessão com múltiplos artefatos e restaurá-la em uma nova sessão limpa."""
    project = AnalysisProject(tmp_path / "output")
    project.initialize_new_project({"embedding_model": "all-MiniLM-L6-v2"})

    # 1. Configura estado inicial
    session_1 = MockSessionState()
    session_1.completed_steps = [1, 2, 4, 5, 6]

    # Artigos e Markdowns
    md_file = project.markdown_dir / "doc1.md"
    md_file.write_text("# Artigo 1\nConteúdo de teste", encoding="utf-8")
    from src.sld.corpus.duplicate_controller import ArticleRegistryRecord
    project.registry.upsert_record(
        ArticleRegistryRecord(
            article_id="doc1",
            source_filename="doc1.pdf",
            source_path=str(tmp_path / "doc1.pdf"),
            pdf_sha256="abc123hash",
            markdown_path=str(md_file.relative_to(project.output_dir)),
            character_count=100,
            page_count=2,
            processing_status="completed"
        )
    )

    # Embeddings e Vetores
    vec_index = VectorIndex(project.index_dir)
    embeddings = np.random.randn(5, 64).astype(np.float32)
    segments = [
        Segment(
            segment_id=f"doc1_p{i}",
            article_id="doc1",
            paragraph_id=f"doc1_p{i}",
            source_pdf="doc1.pdf",
            markdown_path=str(md_file),
            title="Artigo 1",
            section="Intro",
            subsection="",
            page_start=1,
            page_end=1,
            text=f"Texto do parágrafo {i}",
            text_sha256=f"hash_{i}",
            status="valid_paragraph"
        )
        for i in range(5)
    ]
    vec_index.build_and_save(embeddings, segments, "all-MiniLM-L6-v2", "cpu", {"embedding_dim": 64})

    # Candidatos semânticos
    session_1.semantic_candidates = [
        ParagraphRecord(paragraph_id=f"doc1_p{i}", article_id="doc1", text=f"Texto {i}", semantic_score=0.85)
        for i in range(3)
    ]

    # Gold Standard
    session_1.gold_annotations = [
        AnnotationRecord(
            annotation_id=f"ANN_p{i}",
            dataset_id="TEST_SET",
            paragraph_id=f"doc1_p{i}",
            document_id="doc1",
            annotator_id="ANN_001",
            label_1=(i == 0),
            label_2=(i == 1),
            label_0=(i == 2),
            annotation_status="valid",
            included_in_gold_standard=True
        )
        for i in range(3)
    ]

    # Modelo treinado
    clf = MultilabelLogisticClassifier(C=1.0)
    X_dummy = np.random.randn(6, 64)
    y_dummy = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ])
    clf.fit(X_dummy, y_dummy)
    session_1.logistic_classifier = clf
    session_1.optimal_thresholds = {"class_1": 0.45, "class_2": 0.50}

    # Classificados
    session_1.classified_records = [
        ParagraphRecord(
            paragraph_id="doc1_p0",
            article_id="doc1",
            text="Texto 0",
            status="MODEL_RELEVANT",
            semantic_score=0.88,
            predicted_labels=["class_1"],
            predicted_probabilities={"class_1": 0.92}
        ),
        ParagraphRecord(
            paragraph_id="doc1_p1",
            article_id="doc1",
            text="Texto 1",
            status="MODEL_RELEVANT",
            semantic_score=0.82,
            predicted_labels=["class_2"],
            predicted_probabilities={"class_2": 0.78}
        ),
    ]

    # 2. Salva o snapshot da sessão
    save_res = save_full_session_state(project, session_1)
    assert project.session_snapshot_path.exists()
    assert (project.semantic_dir / "candidates.parquet").exists()
    assert (project.classification_dir / "classified_corpus.parquet").exists()
    assert (project.classification_dir / "logistic_classifier.joblib").exists()
    assert (project.annotations_dir / "gold_standard.jsonl").exists()

    # 3. Cria uma sessão vazia e restaura
    session_2 = MockSessionState()
    restore_res = restore_full_session_state(project, session_2)

    assert restore_res["restored_count"] >= 5
    assert len(session_2.articles_records) == 1
    assert len(session_2.corpus_records) == 5
    assert session_2.embeddings_matrix is not None
    assert len(session_2.semantic_candidates) == 3
    assert len(session_2.gold_annotations) == 3
    assert session_2.logistic_classifier is not None
    assert session_2.logistic_classifier.is_fitted
    assert len(session_2.classified_records) == 2
    assert session_2.classified_records[0].predicted_labels == ["class_1"]
    assert 1 in session_2.completed_steps
    assert 2 in session_2.completed_steps
    assert 4 in session_2.completed_steps
    assert 5 in session_2.completed_steps
    assert 6 in session_2.completed_steps


def test_memory_no_vector_duplication(tmp_path: Path):
    """Garante que ao carregar 1000 segmentos, não ocorra duplicação de instâncias numpy dentro de cada ParagraphRecord."""
    project = AnalysisProject(tmp_path / "output_mem")
    project.initialize_new_project({"embedding_model": "all-MiniLM-L6-v2"})

    N = 100
    D = 32
    embeddings = np.random.randn(N, D).astype(np.float32)
    segments = [
        Segment(
            segment_id=f"doc_{i}",
            article_id=f"doc_{i}",
            paragraph_id=f"p_{i}",
            source_pdf="test.pdf",
            markdown_path="test.md",
            title="Test",
            section="Sec",
            subsection="",
            page_start=1,
            page_end=1,
            text=f"Paragraph text {i}",
            text_sha256=f"hash_{i}",
            status="valid_paragraph"
        )
        for i in range(N)
    ]
    vec_index = VectorIndex(project.index_dir)
    vec_index.build_and_save(embeddings, segments, "all-MiniLM-L6-v2", "cpu", {"embedding_dim": D})

    session = MockSessionState()
    restore_full_session_state(project, session)

    assert len(session.corpus_records) == N
    # Verifica que cada ParagraphRecord é leve e não possui p.embedding individual duplicado
    for p in session.corpus_records:
        assert p.embedding is None, "p.embedding não deve duplicar arrays para economizar RAM"
    # A matriz centralizada deve estar presente
    assert session.embeddings_matrix is not None
    assert session.embeddings_matrix.shape == (N, D)


def test_rag_index_roundtrip_persistence(tmp_path: Path):
    """Garante que o índice FAISS e o retriever sejam restaurados fielmente."""
    project = AnalysisProject(tmp_path / "output_rag")
    project.initialize_new_project()

    builder = RAGIndexBuilder(
        output_dir=project.rag_index_dir,
        config=RAGIndexConfig(index_type="FlatIP", version="v001")
    )
    embeddings = np.random.randn(4, 32).astype(np.float32)

    records = [
        ParagraphRecord(
            paragraph_id=f"p_{i}",
            article_id="doc1",
            text=f"Texto RAG {i}",
            status="MODEL_RELEVANT",
            predicted_labels=["class_1"]
        )
        for i in range(4)
    ]

    faiss_p, parq_p, man_p, zip_p, stats, manifest = builder.build(
        corpus_records=records,
        embeddings_matrix=embeddings,
        all_corpus_records=records,
        total_original_articles=1,
        embedding_model_name="all-MiniLM-L6-v2"
    )

    session_1 = MockSessionState()
    session_1.completed_steps = [1, 2, 6, 7]
    session_1.rag_index_stats = stats
    session_1.rag_index_manifest = manifest
    session_1.rag_index_zip_path = str(zip_p)

    save_full_session_state(project, session_1)

    session_2 = MockSessionState()
    restore_full_session_state(project, session_2)

    assert session_2.rag_retriever is not None
    assert session_2.rag_retriever.faiss_index.ntotal == 4
    assert 7 in session_2.completed_steps

