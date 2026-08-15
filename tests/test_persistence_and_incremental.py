"""
Suíte de Testes Automatizados para a Persistência de Análise, Processamento Incremental,
Checkpoints, Controle de Duplicidades por SHA-256 e Gravação Atômica do SLD.
"""

import json
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any

from src.sld.corpus.analysis_project import AnalysisProject, AnalysisMetadata
from src.sld.corpus.duplicate_controller import ArticleRegistry, ArticleRegistryRecord, DuplicateSummary
from src.sld.corpus.checkpoint_manager import CheckpointManager, OperationCheckpoint
from src.sld.corpus.integrity_checker import AnalysisIntegrityChecker, IntegrityReport
from src.sld.semantic.vector_index import VectorIndex
from src.sld.models.search_result import Segment
from src.sld.utils.atomic import atomic_write_json, atomic_write_numpy, atomic_write_text, validate_vector_index_files
from src.sld.utils.hashing import calculate_text_sha256, calculate_file_sha256


@pytest.fixture
def tmp_project_dir(tmp_path) -> Path:
    """Fixture que retorna um diretório de teste isolado."""
    proj_dir = tmp_path / "test_analysis_project"
    proj_dir.mkdir(parents=True, exist_ok=True)
    return proj_dir


def test_atomic_file_operations(tmp_project_dir):
    """Testa escritas atômicas de JSON, Texto e matriz NumPy."""
    json_path = tmp_project_dir / "test.json"
    txt_path = tmp_project_dir / "test.txt"
    npy_path = tmp_project_dir / "test.npy"

    # JSON
    sample_data = {"key": "value", "count": 42}
    atomic_write_json(json_path, sample_data)
    assert json_path.exists()
    assert json.load(open(json_path)) == sample_data

    # Text
    atomic_write_text(txt_path, "Conteúdo Atômico")
    assert txt_path.exists()
    assert txt_path.read_text(encoding="utf-8") == "Conteúdo Atômico"

    # NumPy
    arr = np.ones((5, 10), dtype=np.float32)
    atomic_write_numpy(npy_path, arr)
    assert npy_path.exists()
    loaded_arr = np.load(npy_path)
    assert loaded_arr.shape == (5, 10)


def test_analysis_project_initialization_and_loading(tmp_project_dir):
    """Testa a inicialização e carregamento de metadados em analysis.json."""
    project = AnalysisProject(tmp_project_dir)
    assert not project.is_existing_project()

    meta = project.initialize_new_project()
    assert project.is_existing_project()
    assert meta.status == "new"
    assert (tmp_project_dir / "analysis.json").exists()

    loaded_meta = project.load_metadata()
    assert loaded_meta.analysis_id == meta.analysis_id
    assert loaded_meta.embedding_model == "nomic-embed-text"

    project.update_status("ready")
    assert project.load_metadata().status == "ready"


def test_legacy_folder_migration(tmp_project_dir):
    """Testa a migração de pastas da versão anterior do SLD."""
    # Simula arquivos de uma pasta legada
    legacy_md = tmp_project_dir / "processed" / "artigo_01.md"
    legacy_md.parent.mkdir(parents=True, exist_ok=True)
    legacy_md.write_text("# Artigo Legado\nConteúdo anterior.", encoding="utf-8")

    project = AnalysisProject(tmp_project_dir)
    assert project.is_legacy_sld_folder()
    assert not project.is_existing_project()

    migrated_meta = project.migrate_legacy_folder()
    assert project.is_existing_project()
    assert migrated_meta.migrated_from_legacy is True
    assert (tmp_project_dir / "markdown" / "artigo_01.md").exists()


def test_duplicate_controller_and_sha256_summary(tmp_project_dir):
    """Testa o registro consolidado e a análise de duplicidades por hash SHA-256."""
    manifests_dir = tmp_project_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    reg = ArticleRegistry(manifests_dir)

    # Cria arquivo PDF simulado
    pdf1 = tmp_project_dir / "artigo_A.pdf"
    pdf1.write_bytes(b"CONTEUDO_PDF_A_SIMULADO")
    sha1 = calculate_file_sha256(pdf1)

    rec1 = ArticleRegistryRecord(
        article_id="SLD-A",
        source_filename=pdf1.name,
        source_path=str(pdf1),
        pdf_sha256=sha1,
        processing_status="completed"
    )
    reg.upsert_record(rec1)

    # Analisa lote contendo o mesmo PDF e um novo PDF
    pdf2 = tmp_project_dir / "artigo_B.pdf"
    pdf2.write_bytes(b"CONTEUDO_PDF_B_NOVO")

    summary = reg.analyze_batch([pdf1, pdf2])
    assert summary.total_found == 2
    assert summary.duplicate_content == 1
    assert summary.new_files == 1


def test_checkpoint_manager_lifecycle(tmp_project_dir):
    """Testa a criação, atualização e encerramento de checkpoints."""
    manifests_dir = tmp_project_dir / "manifests"
    chk_mgr = CheckpointManager(manifests_dir)

    assert chk_mgr.get_active_checkpoint() is None

    chk = chk_mgr.create_checkpoint(
        operation_type="ingestion",
        total_items=10,
        all_item_keys=[f"item_{i}" for i in range(10)],
        config={"model": "nomic-embed-text"}
    )

    active = chk_mgr.get_active_checkpoint()
    assert active is not None
    assert active.operation_id == chk.operation_id
    assert active.status == "active"

    chk_mgr.update_item_success(chk, "item_0", current_item_name="artigo_0")
    assert chk.completed_items == 1

    chk_mgr.mark_completed(chk)
    assert chk_mgr.get_active_checkpoint() is None


def test_incremental_vector_index_append(tmp_project_dir):
    """Testa o append incremental de vetores e segmentos no índice vetorial."""
    index_dir = tmp_project_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    vec_index = VectorIndex(index_dir)

    # Lote 1
    emb1 = np.random.rand(3, 768).astype(np.float32)
    seg1 = [
        Segment.from_dict({
            "segment_id": f"s_{i}", "article_id": "art1", "paragraph_id": f"P{i}",
            "source_pdf": "art1.pdf", "markdown_path": "art1.md", "text": f"Texto {i}", "text_sha256": f"hash_{i}"
        })
        for i in range(3)
    ]
    vec_index.build_and_save(emb1, seg1, model_name="nomic-embed-text", device="cpu", config={})

    assert vec_index.load()
    assert vec_index.embeddings.shape == (3, 768)
    assert len(vec_index.segments) == 3

    # Lote 2 (Incremental)
    emb2 = np.random.rand(2, 768).astype(np.float32)
    seg2 = [
        Segment.from_dict({
            "segment_id": f"s_{i}", "article_id": "art2", "paragraph_id": f"P{i}",
            "source_pdf": "art2.pdf", "markdown_path": "art2.md", "text": f"Texto {i}", "text_sha256": f"hash2_{i}"
        })
        for i in range(3, 5)
    ]

    vec_index.append_vectors_and_segments(emb2, seg2, model_name="nomic-embed-text", device="cpu", config={})

    assert vec_index.embeddings.shape == (5, 768)
    assert len(vec_index.segments) == 5


def test_vector_index_incompatible_model_rejection(tmp_project_dir):
    """Testa se o índice rejeita append incremental quando o modelo é incompatível."""
    index_dir = tmp_project_dir / "index"
    vec_index = VectorIndex(index_dir)

    emb1 = np.random.rand(2, 384).astype(np.float32)
    seg1 = [Segment.from_dict({"segment_id": "s1", "article_id": "art1", "source_pdf": "a.pdf", "markdown_path": "a.md", "text": "t1", "text_sha256": "h1"})]
    vec_index.build_and_save(emb1, seg1, model_name="all-MiniLM-L6-v2", device="cpu", config={})

    emb2 = np.random.rand(2, 768).astype(np.float32)
    seg2 = [Segment.from_dict({"segment_id": "s2", "article_id": "art2", "source_pdf": "b.pdf", "markdown_path": "b.md", "text": "t2", "text_sha256": "h2"})]

    with pytest.raises(ValueError, match="incompatível"):
        vec_index.append_vectors_and_segments(emb2, seg2, model_name="nomic-embed-text", device="cpu", config={})


def test_analysis_integrity_checker(tmp_project_dir):
    """Testa a checagem não destrutiva de integridade da análise."""
    project = AnalysisProject(tmp_project_dir)
    project.initialize_new_project()

    checker = AnalysisIntegrityChecker(tmp_project_dir)
    report = checker.run_full_check(expected_model="nomic-embed-text")

    assert isinstance(report, IntegrityReport)
    assert report.status in ["valid", "warning"]
    assert report.project_dir == str(tmp_project_dir)


def test_embeddings_tracker_per_article(tmp_project_dir):
    """Testa a geração de arquivos .npy por artigo e o manifesto embeddings_tracker.md."""
    from src.sld.semantic.embeddings_tracker import EmbeddingsTracker

    index_dir = tmp_project_dir / "index"
    tracker = EmbeddingsTracker(index_dir)

    assert not tracker.has_article_embedding("SLD-TEST-ART1")

    emb = np.random.rand(4, 768).astype(np.float32)
    npy_path = tracker.save_article_embedding(
        article_id="SLD-TEST-ART1",
        source_pdf="artigo_teste.pdf",
        embeddings=emb,
        model_name="nomic-embed-text",
        paragraph_count=4
    )

    assert npy_path.exists()
    assert tracker.has_article_embedding("SLD-TEST-ART1")
    assert tracker.tracker_md_path.exists()

    md_content = tracker.tracker_md_path.read_text(encoding="utf-8")
    assert "SLD-TEST-ART1" in md_content
    assert "artigo_teste.pdf" in md_content

    loaded_emb = tracker.load_article_embedding("SLD-TEST-ART1")
    assert loaded_emb is not None
    assert loaded_emb.shape == (4, 768)
