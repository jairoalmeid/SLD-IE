"""
Serviço orquestrador do pipeline de ingestão, indexação e experimentos do SLD.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
import fitz
import yaml
import pandas as pd

from config.settings import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_MARKDOWN_DIR,
    DEFAULT_INDEX_DIR,
    DEFAULT_MANIFESTS_DIR,
    DEFAULT_EXPORTS_DIR,
    DEFAULT_LOGS_DIR,
    get_default_config,
)
from src.sld.models.article import ArticleMetadata, ProcessedArticle
from src.sld.models.search_result import Segment, SearchResult
from src.sld.models.experiment import ExperimentConfig
from src.sld.ingestion.pdf_reader import read_pdf, PDFReaderError
from src.sld.ingestion.metadata_extractor import extract_metadata
from src.sld.ingestion.text_cleaner import clean_pages
from src.sld.ingestion.reference_remover import remove_references
from src.sld.ingestion.markdown_writer import write_markdown_file
from src.sld.semantic.segmenter import segment_markdown
from src.sld.semantic.embedding_service import EmbeddingService
from src.sld.semantic.vector_index import VectorIndex
from src.sld.semantic.semantic_reference import SemanticReferenceSet
from src.sld.semantic.semantic_search import perform_multi_anchor_search
from src.sld.services.experiment_service import ExperimentService
from src.sld.utils.files import ensure_directory
from src.sld.utils.hashing import calculate_file_sha256
from src.sld.utils.logging_config import setup_logger


class PipelineService:
    """Orquestra o ciclo de vida: PDF -> Markdown -> Embeddings -> Experimento Reprodutível."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_default_config()
        self.logger = setup_logger(DEFAULT_LOGS_DIR)

        self.markdown_dir = DEFAULT_MARKDOWN_DIR
        self.index_dir = DEFAULT_INDEX_DIR
        self.manifests_dir = DEFAULT_MANIFESTS_DIR
        self.exports_dir = DEFAULT_EXPORTS_DIR

        ensure_directory(self.markdown_dir)
        ensure_directory(self.index_dir)
        ensure_directory(self.manifests_dir)
        ensure_directory(self.exports_dir)

        self.vector_index = VectorIndex(self.index_dir)
        self.experiment_service = ExperimentService(DEFAULT_OUTPUT_DIR)

    def process_pdf_batch(
        self,
        pdf_paths: List[Path],
        output_markdown_dir: Optional[Path] = None,
        overwrite_policy: str = "skip",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[List[ProcessedArticle], Path, Path]:
        """Processa individualmente cada arquivo PDF do lote fornecido."""
        target_md_dir = output_markdown_dir or self.markdown_dir
        ensure_directory(target_md_dir)

        results: List[ProcessedArticle] = []
        manifest_entries: List[Dict[str, Any]] = []
        total_files = len(pdf_paths)
        start_time_batch = time.time()

        for idx, pdf_path in enumerate(pdf_paths, start=1):
            if progress_callback:
                progress_callback(idx, total_files, pdf_path.name)

            start_file_time = time.time()
            try:
                pdf_sha256 = calculate_file_sha256(pdf_path)
                pages_data, raw_pdf_meta = read_pdf(pdf_path)
                metadata = extract_metadata(pdf_path, pdf_sha256, pages_data, raw_pdf_meta)
                cleaned_pages = clean_pages(pages_data)

                ref_threshold = self.config.get("ref_confidence_threshold", 0.60)
                filtered_pages, ref_decision = remove_references(cleaned_pages, ref_threshold)

                metadata.references_removed = ref_decision.references_removed
                metadata.reference_start_page = ref_decision.start_page
                if ref_decision.warnings:
                    metadata.metadata_warnings.extend(ref_decision.warnings)

                md_path, written = write_markdown_file(
                    metadata, filtered_pages, target_md_dir, overwrite_policy
                )

                with open(md_path, "r", encoding="utf-8") as f:
                    md_content = f.read()

                duration = time.time() - start_file_time
                char_count = sum(len(p["text"]) for p in filtered_pages)
                status = "warning" if metadata.metadata_warnings else "success"

                processed_article = ProcessedArticle(
                    metadata=metadata,
                    markdown_content=md_content,
                    markdown_path=str(md_path.resolve()),
                    page_count=len(pages_data),
                    char_count=char_count,
                    status=status,
                    error_message=None,
                    processing_duration=duration,
                )

                manifest_entry = {
                    "id": metadata.sld_id,
                    "source_pdf": pdf_path.name,
                    "source_path": str(pdf_path.resolve()),
                    "pdf_sha256": pdf_sha256,
                    "processed_at": metadata.processed_at,
                    "status": status,
                    "page_count": len(pages_data),
                    "char_count": char_count,
                    "markdown_path": str(md_path.resolve()),
                    "metadata": metadata.to_yaml_dict(),
                    "reference_decision": {
                        "removed": ref_decision.references_removed,
                        "detected_title": ref_decision.detected_title,
                        "start_page": ref_decision.start_page,
                        "confidence": ref_decision.confidence,
                        "method": ref_decision.method,
                        "warnings": ref_decision.warnings,
                    },
                    "warnings": metadata.metadata_warnings,
                    "errors": [],
                    "duration_seconds": round(duration, 3),
                    "library_versions": {
                        "PyMuPDF": fitz.__version__,
                        "PyYAML": yaml.__version__,
                    }
                }

                results.append(processed_article)
                manifest_entries.append(manifest_entry)
                self.logger.info(f"Processado com sucesso [{status.upper()}]: {pdf_path.name}")

            except PDFReaderError as e:
                duration = time.time() - start_file_time
                err_msg = str(e)
                self.logger.warning(f"Aviso PDF '{pdf_path.name}': {err_msg}")
                dummy_meta = ArticleMetadata(
                    sld_id=f"SLD-ERR-{pdf_path.stem[:8]}",
                    source_pdf=pdf_path.name,
                    source_path=str(pdf_path.resolve()),
                    processed_at=datetime.now().isoformat(),
                    metadata_warnings=[err_msg]
                )
                results.append(ProcessedArticle(
                    metadata=dummy_meta, markdown_content="", markdown_path="",
                    page_count=0, char_count=0, status="error", error_message=err_msg, processing_duration=duration
                ))
                manifest_entries.append({
                    "id": dummy_meta.sld_id, "source_pdf": pdf_path.name, "source_path": str(pdf_path.resolve()),
                    "processed_at": dummy_meta.processed_at, "status": "error", "errors": [err_msg], "duration_seconds": round(duration, 3)
                })
            except Exception as e:
                duration = time.time() - start_file_time
                err_msg = f"Erro inesperado: {e}"
                self.logger.error(f"Erro '{pdf_path.name}': {err_msg}", exc_info=True)
                dummy_meta = ArticleMetadata(
                    sld_id=f"SLD-ERR-{pdf_path.stem[:8]}", source_pdf=pdf_path.name,
                    source_path=str(pdf_path.resolve()), processed_at=datetime.now().isoformat()
                )
                results.append(ProcessedArticle(
                    metadata=dummy_meta, markdown_content="", markdown_path="",
                    page_count=0, char_count=0, status="error", error_message=err_msg, processing_duration=duration
                ))
                manifest_entries.append({
                    "id": dummy_meta.sld_id, "source_pdf": pdf_path.name, "source_path": str(pdf_path.resolve()),
                    "processed_at": dummy_meta.processed_at, "status": "error", "errors": [err_msg], "duration_seconds": round(duration, 3)
                })

        manifest_file, summary_file = self._save_manifests(manifest_entries, time.time() - start_time_batch)
        return results, manifest_file, summary_file

    def _save_manifests(self, entries: List[Dict[str, Any]], total_duration: float) -> Tuple[Path, Path]:
        manifest_path = self.manifests_dir / "ingestion_manifest.jsonl"
        summary_path = self.manifests_dir / "processing_summary.json"

        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        summary = {
            "processed_at": datetime.now().isoformat(),
            "total_files": len(entries),
            "success_count": sum(1 for e in entries if e.get("status") == "success"),
            "warning_count": sum(1 for e in entries if e.get("status") == "warning"),
            "error_count": sum(1 for e in entries if e.get("status") == "error"),
            "total_duration_seconds": round(total_duration, 2),
            "configuration": self.config,
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return manifest_path, summary_path

    def index_markdown_corpus(
        self,
        markdown_dir: Path,
        embedding_service: EmbeddingService,
        force_reindex: bool = False
    ) -> VectorIndex:
        """Segmenta o corpus e constrói o índice vetorial."""
        md_files = list(markdown_dir.glob("*.md"))
        all_segments: List[Segment] = []

        for md_path in md_files:
            try:
                with open(md_path, "r", encoding="utf-8") as f:
                    content = f.read()
                segs = segment_markdown(
                    content,
                    md_path,
                    min_words=self.config.get("min_words", 10),
                    min_characters=self.config.get("min_characters", 50),
                    max_characters=self.config.get("max_characters", 1500),
                    long_text_strategy=self.config.get("long_text_strategy", "chunk"),
                    overlap=self.config.get("chunk_overlap", 100),
                )
                all_segments.extend(segs)
            except Exception as e:
                self.logger.error(f"Erro ao segmentar '{md_path.name}': {e}")

        if not force_reindex and self.vector_index.is_valid(self.config, all_segments):
            self.vector_index.load()
            return self.vector_index

        valid_segs = [s for s in all_segments if s.status == "valid_paragraph"]
        texts = [s.text for s in valid_segs]

        embeddings = embedding_service.encode(texts, batch_size=self.config.get("batch_size", 32))

        self.vector_index.build_and_save(
            embeddings=embeddings,
            segments=valid_segs,
            model_name=embedding_service.model_name,
            device=embedding_service.device,
            config=self.config,
        )
        return self.vector_index

    def run_experiment(
        self,
        markdown_dir: Path,
        embedding_service: EmbeddingService,
        reference_set: SemanticReferenceSet,
        experiment_config: ExperimentConfig,
        selected_article_ids: Optional[List[str]] = None
    ) -> Tuple[List[SearchResult], Path]:
        """
        Executa um experimento científico reprodutivo completo e salva os artefatos em output/experiments/<run_id>/.
        """
        start_time = time.time()

        # 1. Garante indexação do corpus
        vector_index = self.index_markdown_corpus(markdown_dir, embedding_service)

        start_sim_time = time.time()
        # 2. Busca semântica multi-âncora
        results = perform_multi_anchor_search(
            vector_index=vector_index,
            embedding_service=embedding_service,
            reference_set=reference_set,
            aggregation_strategy=experiment_config.aggregation_strategy,
            threshold=experiment_config.threshold,
            chunk_aggregation=experiment_config.chunk_aggregation,
            selected_article_ids=selected_article_ids,
            run_id=experiment_config.run_id
        )
        sim_time = time.time() - start_sim_time
        total_time = time.time() - start_time

        perf = {
            "processing_time": total_time,
            "embedding_time": 0.0,
            "similarity_time": sim_time,
            "paragraphs_per_second": len(vector_index.segments) / total_time if total_time > 0 else 0.0,
        }

        # 3. Salva experimento e gera METHODS.md
        exp_dir = self.experiment_service.save_experiment_run(
            config=experiment_config,
            embedding_service=embedding_service,
            reference_set=reference_set,
            results=results,
            segments=vector_index.segments,
            evaluation_summary=None,
            performance_times=perf
        )

        return results, exp_dir
