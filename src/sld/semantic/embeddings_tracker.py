"""
Gerenciador de Embeddings por Artigo e Registro de Rastreamento em Markdown (embeddings_tracker.md).
Gera 1 arquivo .npy individual por artigo sob index/embeddings/ e atualiza o manifesto .md.
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from src.sld.utils.atomic import atomic_write_text, atomic_write_numpy, atomic_write_json
from src.sld.utils.files import ensure_directory


class PerArticleEmbeddingRecord(BaseModel):
    """Representa a entrada de rastreamento do arquivo de embedding (.npy) de um artigo individual."""
    article_id: str
    source_pdf: str
    paragraph_count: int
    embedding_dim: int
    model_name: str
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    npy_path: str
    status: str = "completed"  # "completed", "pending", "error"
    error_message: Optional[str] = None


class EmbeddingsTracker:
    """
    Gerencia o salvamento isolado de vetores por artigo (index/embeddings/<article_id>.npy)
    e mantém o arquivo de rastreamento auditável em Markdown (index/embeddings_tracker.md).
    """

    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir).expanduser().resolve()
        self.embeddings_dir = self.index_dir / "embeddings"
        ensure_directory(self.embeddings_dir)

        self.tracker_md_path = self.index_dir / "embeddings_tracker.md"
        self.tracker_json_path = self.embeddings_dir / "tracker_registry.json"

    def load_records(self) -> Dict[str, PerArticleEmbeddingRecord]:
        """Carrega os registros de embeddings por artigo a partir de JSON e Markdown."""
        if not self.tracker_json_path.exists():
            return {}
        try:
            with open(self.tracker_json_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            records = {}
            for k, v in raw_data.items():
                records[k] = PerArticleEmbeddingRecord(**v)
            return records
        except Exception:
            return {}

    def save_records(self, records: Dict[str, PerArticleEmbeddingRecord]) -> None:
        """Salva os registros no formato JSON de suporte e gera o arquivo Markdown legível."""
        # 1. Salva suporte JSON
        json_data = {k: v.model_dump() for k, v in records.items()}
        atomic_write_json(self.tracker_json_path, json_data)

        # 2. Gera o arquivo Markdown embeddings_tracker.md
        md_lines = [
            "# Registro de Embeddings Vetoriais por Artigo — SLD",
            "",
            f"**Última Atualização:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`  ",
            f"**Total de Artigos Processados:** `{len(records)}`",
            "",
            "| Status | ID do Artigo | Arquivo PDF | Parágrafos | Dimensão | Modelo | Data/Hora | Arquivo Embedding (.npy) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"
        ]

        for r in sorted(records.values(), key=lambda x: x.article_id):
            icon = "✓" if r.status == "completed" else "❌"
            npy_basename = Path(r.npy_path).name
            dt_str = r.generated_at[:16].replace("T", " ")
            line = f"| {icon} {r.status.upper()} | `{r.article_id}` | `{r.source_pdf}` | {r.paragraph_count} | {r.embedding_dim} | `{r.model_name}` | `{dt_str}` | `{npy_basename}` |"
            md_lines.append(line)

        md_lines.append("")
        atomic_write_text(self.tracker_md_path, "\n".join(md_lines))

    def has_article_embedding(self, article_id: str) -> bool:
        """Verifica se o artigo já possui arquivo .npy gerado e válido."""
        records = self.load_records()
        if article_id in records and records[article_id].status == "completed":
            npy_p = Path(records[article_id].npy_path)
            if not npy_p.is_absolute():
                npy_p = self.index_dir / records[article_id].npy_path
            return npy_p.exists()
        return False

    def save_article_embedding(
        self,
        article_id: str,
        source_pdf: str,
        embeddings: np.ndarray,
        model_name: str,
        paragraph_count: int
    ) -> Path:
        """
        Salva o arquivo .npy exclusivo do artigo sob index/embeddings/<article_id>.npy
        e atualiza o registro em embeddings_tracker.md.
        """
        ensure_directory(self.embeddings_dir)
        safe_filename = re.sub(r'[^\w\-_\.]', '_', article_id)
        npy_path = self.embeddings_dir / f"{safe_filename}.npy"

        atomic_write_numpy(npy_path, embeddings)

        dim = embeddings.shape[1] if embeddings.ndim == 2 else 0

        rec = PerArticleEmbeddingRecord(
            article_id=article_id,
            source_filename=source_pdf,
            source_pdf=source_pdf,
            paragraph_count=paragraph_count,
            embedding_dim=dim,
            model_name=model_name,
            generated_at=datetime.now().isoformat(),
            npy_path=str(npy_path.relative_to(self.index_dir) if npy_path.is_relative_to(self.index_dir) else npy_path),
            status="completed"
        )

        records = self.load_records()
        records[article_id] = rec
        self.save_records(records)
        return npy_path

    def load_article_embedding(self, article_id: str) -> Optional[np.ndarray]:
        """Carrega a matriz de embeddings .npy de um artigo individual."""
        records = self.load_records()
        if article_id not in records:
            return None

        rec = records[article_id]
        npy_p = Path(rec.npy_path)
        if not npy_p.is_absolute():
            npy_p = self.index_dir / rec.npy_path

        if not npy_p.exists():
            return None

        try:
            return np.load(npy_p)
        except Exception:
            return None

    def get_summary_counts(self) -> Tuple[int, int]:
        """Retorna (total_registrados, total_concluidos)."""
        recs = self.load_records()
        completed = sum(1 for r in recs.values() if r.status == "completed")
        return len(recs), completed
