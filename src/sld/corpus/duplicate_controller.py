"""
Controle de Duplicidades por SHA-256 e Registro Consolidado de Artigos no SLD.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
from src.sld.utils.atomic import atomic_write_json
from src.sld.utils.files import ensure_directory
from src.sld.utils.hashing import calculate_file_sha256, generate_article_id


class ArticleRegistryRecord(BaseModel):
    """Representa a ficha cadastral completa e persistente de um artigo no acervo."""
    article_id: str
    source_filename: str
    source_path: str
    pdf_sha256: str
    file_size: int = 0
    modified_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    markdown_path: str = ""
    markdown_sha256: str = ""

    processing_status: str = "pending"  # "completed", "pending", "interrupted", "error"
    extraction_status: str = "pending"  # "success", "failed", "skipped"
    reference_removal_status: str = "pending"  # "success", "skipped", "failed"
    segmentation_status: str = "pending"  # "completed", "pending"
    embedding_status: str = "pending"  # "completed", "pending"

    page_count: int = 0
    character_count: int = 0
    segment_count: int = 0

    processing_started_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    processing_completed_at: Optional[str] = None
    last_updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    error_message: Optional[str] = None
    retry_count: int = 0

    previous_article_id: Optional[str] = None
    version_number: int = 1
    supersedes: Optional[str] = None


class DuplicateSummary(BaseModel):
    """Resumo quantitativo da pré-análise de duplicidades de um lote de PDFs."""
    total_found: int = 0
    new_files: int = 0
    duplicate_content: int = 0
    same_name_different_hash: int = 0
    previously_interrupted: int = 0
    previous_error: int = 0
    categorized_items: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)


class ArticleRegistry:
    """Gerencia o manifesto consolidado de artigos em manifests/article_registry.json."""

    def __init__(self, manifests_dir: Path):
        self.manifests_dir = Path(manifests_dir).expanduser().resolve()
        ensure_directory(self.manifests_dir)
        self.registry_path = self.manifests_dir / "article_registry.json"

    def load_registry(self) -> Dict[str, ArticleRegistryRecord]:
        """Carrega todos os registros de artigos ativos do arquivo JSON."""
        if not self.registry_path.exists():
            return {}
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            records = {}
            for k, v in raw_data.items():
                records[k] = ArticleRegistryRecord(**v)
            return records
        except Exception:
            return {}

    def save_registry(self, registry: Dict[str, ArticleRegistryRecord]) -> None:
        """Salva atomicamente o dicionário completo de registros no disco."""
        data = {k: v.model_dump() for k, v in registry.items()}
        atomic_write_json(self.registry_path, data)

    def get_by_sha256(self, sha256_hash: str) -> Optional[ArticleRegistryRecord]:
        """Busca um registro de artigo pelo hash SHA-256 do arquivo PDF."""
        reg = self.load_registry()
        for rec in reg.values():
            if rec.pdf_sha256 == sha256_hash:
                return rec
        return None

    def get_by_filename(self, filename: str) -> Optional[ArticleRegistryRecord]:
        """Busca um registro de artigo pelo nome do arquivo fonte."""
        reg = self.load_registry()
        for rec in reg.values():
            if rec.source_filename == filename:
                return rec
        return None

    def upsert_record(self, record: ArticleRegistryRecord) -> None:
        """Insere ou atualiza um registro de artigo."""
        reg = self.load_registry()
        record.last_updated_at = datetime.now().isoformat()
        reg[record.article_id] = record
        self.save_registry(reg)

    def analyze_batch(self, pdf_paths: List[Path]) -> DuplicateSummary:
        """
        Analisa previamente um lote de arquivos PDF contra o registro consolidado
        e retorna um resumo completo categorizando cada item.
        """
        reg = self.load_registry()

        # Mapeamento rápido por SHA256 e por Filename
        sha_map: Dict[str, ArticleRegistryRecord] = {}
        fname_map: Dict[str, ArticleRegistryRecord] = {}

        for rec in reg.values():
            sha_map[rec.pdf_sha256] = rec
            fname_map[rec.source_filename] = rec

        summary = DuplicateSummary(
            total_found=len(pdf_paths),
            categorized_items={
                "new": [],
                "duplicate_content": [],
                "same_name_different_hash": [],
                "previously_interrupted": [],
                "previous_error": []
            }
        )

        for p in pdf_paths:
            if not p.exists():
                continue

            sha = calculate_file_sha256(p)
            fname = p.name
            fsize = p.stat().st_size

            # 1. Checa por hash de conteúdo
            if sha in sha_map:
                existing = sha_map[sha]
                if existing.processing_status == "completed":
                    summary.duplicate_content += 1
                    summary.categorized_items["duplicate_content"].append({
                        "path": str(p),
                        "filename": fname,
                        "sha256": sha,
                        "article_id": existing.article_id,
                        "existing_status": existing.processing_status
                    })
                elif existing.processing_status in ["interrupted", "pending"]:
                    summary.previously_interrupted += 1
                    summary.categorized_items["previously_interrupted"].append({
                        "path": str(p),
                        "filename": fname,
                        "sha256": sha,
                        "article_id": existing.article_id
                    })
                else:  # error
                    summary.previous_error += 1
                    summary.categorized_items["previous_error"].append({
                        "path": str(p),
                        "filename": fname,
                        "sha256": sha,
                        "article_id": existing.article_id,
                        "error_message": existing.error_message
                    })
            elif fname in fname_map:
                # Mesmo nome com hash diferente -> nova versão
                existing = fname_map[fname]
                summary.same_name_different_hash += 1
                summary.categorized_items["same_name_different_hash"].append({
                    "path": str(p),
                    "filename": fname,
                    "sha256": sha,
                    "existing_sha256": existing.pdf_sha256,
                    "article_id": existing.article_id,
                    "existing_version": existing.version_number
                })
            else:
                summary.new_files += 1
                summary.categorized_items["new"].append({
                    "path": str(p),
                    "filename": fname,
                    "sha256": sha,
                    "suggested_id": generate_article_id(sha, fname)
                })

        return summary
