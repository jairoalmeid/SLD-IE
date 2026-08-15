"""
Testes unitários para persistência de experimentos e geração do relatório METHODS.md.
"""

from pathlib import Path
from src.sld.models.experiment import ExperimentConfig
from src.sld.models.search_result import SearchResult, Segment
from src.sld.semantic.semantic_reference import SemanticReferenceSet
from src.sld.semantic.embedding_service import EmbeddingService
from src.sld.services.experiment_service import ExperimentService, generate_methods_markdown_report


def test_experiment_service_save_run(tmp_path):
    """Testa criação da estrutura de experimento output/experiments/<run_id>/ e geração do METHODS.md."""
    exp_service = ExperimentService(base_output_dir=tmp_path)
    run_id = exp_service.create_run_id()

    config = ExperimentConfig(
        run_id=run_id,
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        similarity_metric="cosine",
        aggregation_strategy="maximum",
        threshold=0.50
    )

    emb_service = EmbeddingService(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    ref_set = SemanticReferenceSet()

    segments = [
        Segment(
            segment_id="SLD-001_P0001",
            article_id="SLD-001",
            paragraph_id="P0001",
            source_pdf="artigo.pdf",
            markdown_path="/tmp/artigo.md",
            title="Artigo de Teste",
            section="Introdução",
            subsection="",
            page_start=1,
            page_end=1,
            text="Parágrafo de teste sobre vulnerabilidade em desastres.",
            text_sha256="hash123",
            word_count=8,
            char_count=52,
            status="valid_paragraph"
        )
    ]

    results = [
        SearchResult(
            rank=1,
            aggregate_score=0.78,
            article_id="SLD-001",
            paragraph_id="P0001",
            paragraph_hash="hash123",
            title="Artigo de Teste",
            authors=["Autor A"],
            source_pdf="artigo.pdf",
            section="Introdução",
            subsection="",
            page_range="1",
            segment_id="SLD-001_P0001",
            chunk_id=None,
            text="Parágrafo de teste sobre vulnerabilidade em desastres.",
            anchor_scores={"Q1": 0.78, "Q2": 0.40},
            best_anchor_id="Q1",
            best_anchor_text="Definição geral de vulnerabilidade",
            context_before=None,
            context_after=None,
            markdown_path="/tmp/artigo.md",
            run_id=run_id,
            threshold_used=0.50,
            selected=True
        )
    ]

    exp_dir = exp_service.save_experiment_run(
        config=config,
        embedding_service=emb_service,
        reference_set=ref_set,
        results=results,
        segments=segments
    )

    assert exp_dir.exists()
    assert (exp_dir / "config.json").exists()
    assert (exp_dir / "environment.json").exists()
    assert (exp_dir / "manifest.json").exists()
    assert (exp_dir / "results.csv").exists()
    assert (exp_dir / "results.md").exists()
    assert (exp_dir / "METHODS.md").exists()

    with open(exp_dir / "METHODS.md", "r", encoding="utf-8") as f:
        methods_content = f.read()

    assert "Methodology Report" in methods_content
    assert run_id in methods_content
    assert "paraphrase-multilingual-mpnet-base-v2" in methods_content
    assert "Cosine Similarity" in methods_content
