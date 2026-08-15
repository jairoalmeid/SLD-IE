"""
Gerenciador do Projeto de Análise Persistente no SLD.
Trata cada pasta de saída como uma unidade persistente e independente de análise.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from config.settings import APP_NAME, APP_VERSION, DEFAULT_EMBEDDING_MODEL, get_default_config
from src.sld.corpus.duplicate_controller import ArticleRegistry, ArticleRegistryRecord
from src.sld.corpus.checkpoint_manager import CheckpointManager
from src.sld.corpus.integrity_checker import AnalysisIntegrityChecker, IntegrityReport
from src.sld.utils.atomic import atomic_write_json
from src.sld.utils.files import ensure_directory


class AnalysisMetadata(BaseModel):
    """Metadados do arquivo persistente analysis.json."""
    analysis_id: str
    application: str = APP_NAME
    application_version: str = APP_VERSION
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: str = "new"  # new, ready, processing, interrupted, partially_completed, completed, inconsistent
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    segmentation_config: Dict[str, Any] = Field(default_factory=dict)
    reference_removal_config: Dict[str, Any] = Field(default_factory=dict)
    article_count: int = 0
    segment_count: int = 0
    index_version: int = 1
    migrated_from_legacy: bool = False


class AnalysisProject:
    """
    Gerencia a pasta de saída como um projeto persistente de análise do SLD.
    Estrutura:
      output_dir/
      ├── analysis.json
      ├── markdown/
      ├── index/
      ├── manifests/
      ├── exports/
      ├── logs/
      └── recovery/
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir).expanduser().resolve()
        ensure_directory(self.output_dir)

        self.markdown_dir = self.output_dir / "markdown"
        self.index_dir = self.output_dir / "index"
        self.manifests_dir = self.output_dir / "manifests"
        self.exports_dir = self.output_dir / "exports"
        self.logs_dir = self.output_dir / "logs"
        self.recovery_dir = self.output_dir / "recovery"

        # Diretórios dedicados aos estágios desacoplados do pipeline
        self.semantic_dir = self.output_dir / "semantic"
        self.annotations_dir = self.output_dir / "annotations"
        self.classification_dir = self.output_dir / "classification"
        self.rag_index_dir = self.output_dir / "rag_index"
        self.llm_dir = self.output_dir / "llm"

        for d in [
            self.markdown_dir,
            self.index_dir,
            self.manifests_dir,
            self.exports_dir,
            self.logs_dir,
            self.recovery_dir,
            self.semantic_dir,
            self.annotations_dir,
            self.classification_dir,
            self.rag_index_dir,
            self.llm_dir,
        ]:
            ensure_directory(d)

        self.analysis_json_path = self.output_dir / "analysis.json"
        self.session_snapshot_path = self.manifests_dir / "session_state.json"

        self.registry = ArticleRegistry(self.manifests_dir)
        self.checkpoint_mgr = CheckpointManager(self.manifests_dir)
        self.integrity_checker = AnalysisIntegrityChecker(self.output_dir)

    def is_existing_project(self) -> bool:
        """Verifica se a pasta já contém um arquivo analysis.json válido."""
        return self.analysis_json_path.exists()

    def is_legacy_sld_folder(self) -> bool:
        """
        Verifica se a pasta pertence a uma versão anterior do SLD (contém arquivos .md,
        manifestos ou parágrafos sem o arquivo analysis.json).
        """
        if self.is_existing_project():
            return False

        # Checa por indicadores conhecidos da versão anterior
        has_md = any(self.output_dir.rglob("*.md"))
        has_manifest = (self.output_dir / "manifests" / "ingestion_manifest.jsonl").exists()
        has_config = any(self.output_dir.rglob("run_config.json"))
        has_paragraphs = (self.output_dir / "paragraphs" / "corpus.jsonl").exists() or (self.output_dir / "corpus.jsonl").exists()

        return has_md or has_manifest or has_config or has_paragraphs

    def initialize_new_project(self, config: Optional[Dict[str, Any]] = None) -> AnalysisMetadata:
        """Inicializa um novo projeto limpo em uma pasta vazia."""
        for d in [self.markdown_dir, self.index_dir, self.manifests_dir, self.exports_dir, self.logs_dir, self.recovery_dir]:
            ensure_directory(d)

        cfg = config or get_default_config()

        meta = AnalysisMetadata(
            analysis_id=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            application=APP_NAME,
            application_version=APP_VERSION,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            status="new",
            embedding_model=cfg.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
            segmentation_config={
                "min_words": cfg.get("min_words", 8),
                "min_characters": cfg.get("min_characters", 40),
                "max_characters": cfg.get("max_characters", 500),
                "long_text_strategy": cfg.get("long_text_strategy", "chunk"),
                "chunk_overlap": cfg.get("chunk_overlap", 100),
                "chunk_aggregation": cfg.get("chunk_aggregation", "maximum")
            },
            reference_removal_config={
                "ref_confidence_threshold": cfg.get("ref_confidence_threshold", 0.60)
            },
            article_count=0,
            segment_count=0,
            index_version=1
        )

        atomic_write_json(self.analysis_json_path, meta.model_dump())
        return meta

    def migrate_legacy_folder(self, config: Optional[Dict[str, Any]] = None) -> AnalysisMetadata:
        """
        Migra assistidamente uma pasta da versão anterior do SLD sem apagar nem alterar arquivos.
        """
        for d in [self.markdown_dir, self.index_dir, self.manifests_dir, self.exports_dir, self.logs_dir, self.recovery_dir]:
            ensure_directory(d)

        cfg = config or get_default_config()

        # Inspeciona Markdowns existentes na pasta
        md_files = list(self.output_dir.rglob("*.md"))
        for mf in md_files:
            if mf.parent != self.markdown_dir:
                try:
                    dest = self.markdown_dir / mf.name
                    if not dest.exists():
                        import shutil
                        shutil.copy2(mf, dest)
                except Exception:
                    pass

        markdowns_count = len(list(self.markdown_dir.glob("*.md")))

        # Constrói o article_registry a partir dos arquivos e manifestos anteriores
        reg = self.registry.load_registry()

        # Backup de manifestos anteriores em recovery/
        ing_manifest_old = self.output_dir / "manifests" / "ingestion_manifest.jsonl"
        if ing_manifest_old.exists():
            import shutil
            shutil.copy2(ing_manifest_old, self.recovery_dir / "legacy_ingestion_manifest.jsonl")

        meta = AnalysisMetadata(
            analysis_id=f"analysis_legacy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            application=APP_NAME,
            application_version=APP_VERSION,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            status="partially_completed" if markdowns_count > 0 else "new",
            embedding_model=cfg.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
            segmentation_config={
                "min_words": cfg.get("min_words", 8),
                "min_characters": cfg.get("min_characters", 40),
                "max_characters": cfg.get("max_characters", 500),
                "long_text_strategy": cfg.get("long_text_strategy", "chunk"),
                "chunk_overlap": cfg.get("chunk_overlap", 100),
                "chunk_aggregation": cfg.get("chunk_aggregation", "maximum")
            },
            reference_removal_config={
                "ref_confidence_threshold": cfg.get("ref_confidence_threshold", 0.60)
            },
            article_count=markdowns_count,
            segment_count=0,
            index_version=1,
            migrated_from_legacy=True
        )

        atomic_write_json(self.analysis_json_path, meta.model_dump())
        return meta

    def load_metadata(self, default_config: Optional[Dict[str, Any]] = None) -> AnalysisMetadata:
        """Carrega os metadados da análise a partir de analysis.json ou inicializa se não existir."""
        if not self.analysis_json_path.exists():
            return self.initialize_new_project(default_config or {})
        try:
            with open(self.analysis_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return AnalysisMetadata(**data)
        except Exception:
            return self.initialize_new_project(default_config or {})

    def save_metadata(self, metadata: AnalysisMetadata) -> None:
        """Salva atomicamente os metadados atualizados."""
        metadata.updated_at = datetime.now().isoformat()
        atomic_write_json(self.analysis_json_path, metadata.model_dump())

    def update_status(self, new_status: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Atualiza o status global da análise."""
        meta = self.load_metadata(config)
        meta.status = new_status
        self.save_metadata(meta)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calcula e retorna as estatísticas consolidadas da análise."""
        meta = self.load_metadata()
        reg_records = self.registry.load_registry()

        completed_count = sum(1 for r in reg_records.values() if r.processing_status == "completed")
        pending_count = sum(1 for r in reg_records.values() if r.processing_status in ["pending", "interrupted"])
        error_count = sum(1 for r in reg_records.values() if r.processing_status == "error")

        index_valid, index_errs = self.integrity_checker.index_dir.exists() and (
            (self.output_dir / "index" / "embeddings.npy").exists(), []
        )

        # Checa status do índice
        index_status = "válido" if (self.output_dir / "index" / "embeddings.npy").exists() else "ausente"

        return {
            "analysis_id": meta.analysis_id,
            "status": meta.status,
            "total_registered": len(reg_records) or meta.article_count,
            "completed_articles": completed_count or meta.article_count,
            "pending_articles": pending_count,
            "error_articles": error_count,
            "total_segments": meta.segment_count,
            "last_updated": meta.updated_at,
            "embedding_model": meta.embedding_model,
            "index_status": index_status,
            "index_version": meta.index_version,
            "migrated_from_legacy": meta.migrated_from_legacy
        }

    def save_session_snapshot(self, session_data: Dict[str, Any]) -> Path:
        """
        Salva atomicamente o estado completo da sessão no arquivo session_state.json.
        """
        data_to_save = dict(session_data)
        data_to_save["snapshot_saved_at"] = datetime.now().isoformat()
        data_to_save["analysis_id"] = self.load_metadata().analysis_id
        atomic_write_json(self.session_snapshot_path, data_to_save)
        return self.session_snapshot_path

    def load_session_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Carrega o snapshot de sessão persistido no projeto, se existente.
        """
        if not self.session_snapshot_path.exists():
            return None
        try:
            with open(self.session_snapshot_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

