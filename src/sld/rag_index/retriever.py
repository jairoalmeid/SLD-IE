"""
Recuperador semântico (Retriever) para consultas Top-k no Índice RAG FAISS.
"""

import json
import zipfile
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import faiss
import pyarrow.parquet as pq

from src.sld.semantic.embedding_service import EmbeddingService
from src.sld.rag_index.models import RAGQueryResult, RAGIndexManifest, IndexStats


class RAGIndexRetriever:
    """
    Carrega e executa consultas vetoriais sobre o índice FAISS e a tabela de metadados Parquet.
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
    ):
        self.index_path = Path(index_path) if index_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.manifest_path = Path(manifest_path) if manifest_path else None

        self.faiss_index: Optional[faiss.Index] = None
        self.metadata_df: Optional[pd.DataFrame] = None
        self.manifest: Optional[RAGIndexManifest] = None

    def is_loaded(self) -> bool:
        return self.faiss_index is not None and self.metadata_df is not None

    def load_from_dir(self, rag_dir: Path) -> None:
        """Carrega os artefatos a partir de um diretório que contenha o índice."""
        rag_dir = Path(rag_dir)
        idx_f = rag_dir / "corpus_refinado.faiss"
        meta_f = rag_dir / "metadata.parquet"
        mani_f = rag_dir / "manifest.json"

        if not idx_f.exists() or not meta_f.exists():
            raise FileNotFoundError(f"Arquivos corpus_refinado.faiss ou metadata.parquet não encontrados em {rag_dir}.")

        self.index_path = idx_f
        self.metadata_path = meta_f
        self.manifest_path = mani_f if mani_f.exists() else None

        self.faiss_index = faiss.read_index(str(idx_f))
        self.metadata_df = pd.read_parquet(meta_f)

        if mani_f.exists():
            with open(mani_f, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.manifest = RAGIndexManifest(**data)

    def load_from_zip(self, zip_bytes_or_path: Any, extract_to_dir: Optional[Path] = None) -> Path:
        """Descompacta um arquivo ZIP e carrega o índice resultante."""
        if extract_to_dir is None:
            extract_to_dir = Path(tempfile.mkdtemp(prefix="rag_index_extracted_"))
        else:
            extract_to_dir = Path(extract_to_dir)
            extract_to_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_bytes_or_path, "r") as zf:
            zf.extractall(extract_to_dir)

        # Procura os arquivos descompactados (pode estar na raiz ou numa subpasta)
        idx_matches = list(extract_to_dir.rglob("corpus_refinado.faiss"))
        meta_matches = list(extract_to_dir.rglob("metadata.parquet"))

        if not idx_matches or not meta_matches:
            raise FileNotFoundError("O arquivo ZIP não contém os arquivos corpus_refinado.faiss ou metadata.parquet.")

        self.load_from_dir(idx_matches[0].parent)
        return idx_matches[0].parent

    def query(
        self,
        query_text: str,
        embedding_service: EmbeddingService,
        top_k: int = 10,
        required_classes: Optional[List[str]] = None,
        min_score: float = -1.0,
    ) -> List[RAGQueryResult]:
        """
        Executa a recuperação vetorial Top-k no índice FAISS e mapeia os metadados correspondentes.
        """
        if not self.is_loaded():
            raise RuntimeError("O índice FAISS e os metadados precisam ser carregados antes da consulta.")

        # 1. Gera embedding normalizado L2 da query de texto
        q_vec = embedding_service.encode_queries([query_text], normalize=True).astype(np.float32)

        # 2. Busca vetorial no FAISS
        # Busca uma janela maior se houver filtros conceituais por classe
        search_k = min(self.faiss_index.ntotal, top_k * 5 if required_classes else top_k)
        if search_k <= 0:
            return []

        scores, indices = self.faiss_index.search(q_vec, search_k)

        scores_row = scores[0]
        indices_row = indices[0]

        results: List[RAGQueryResult] = []
        rank_counter = 1

        for s_val, faiss_id in zip(scores_row, indices_row):
            if faiss_id < 0 or faiss_id >= len(self.metadata_df):
                continue

            if min_score > 0 and float(s_val) < min_score:
                continue

            row = self.metadata_df.iloc[int(faiss_id)]

            # Filtros conceituais opcionais (ex: class_1=True)
            if required_classes:
                has_all_req = True
                for rc in required_classes:
                    col = f"class_{rc}" if not rc.startswith("class_") else rc
                    if col in row and not bool(row[col]):
                        has_all_req = False
                        break
                if not has_all_req:
                    continue

            # Monta lista de classes ativas para exibição
            active_classes = []
            if row.get("class_1", False):
                active_classes.append("1 — Definição")
            if row.get("class_2", False):
                active_classes.append("2 — Fator Determinante")
            if row.get("class_3", False):
                active_classes.append("3 — Tipo/Dimensão")
            if row.get("class_4", False):
                active_classes.append("4 — Relação Causal")
            if row.get("class_5", False):
                active_classes.append("5 — Propriedade")

            results.append(
                RAGQueryResult(
                    rank=rank_counter,
                    faiss_id=int(faiss_id),
                    paragraph_id=str(row["paragraph_id"]),
                    article_id=str(row["article_id"]),
                    score=float(s_val),
                    classes=active_classes,
                    section=str(row.get("section", "") or ""),
                    title=str(row.get("title", "") or ""),
                    text=str(row.get("text", "")),
                )
            )
            rank_counter += 1

            if len(results) >= top_k:
                break

        return results
