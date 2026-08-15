"""
Índice vetorial local baseado em matrizes NumPy e arquivos JSONL com suporte a gravações atômicas e append incremental.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from src.sld.models.search_result import Segment
from src.sld.utils.files import ensure_directory
from src.sld.utils.hashing import calculate_text_sha256
from src.sld.utils.atomic import atomic_write_json, atomic_write_numpy, atomic_write_text, validate_vector_index_files


class VectorIndex:
    """Gerencia o armazenamento, carregamento, atualização incremental e validação do índice vetorial local."""

    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir).expanduser().resolve()
        self.embeddings_path = self.index_dir / "embeddings.npy"
        self.segments_path = self.index_dir / "segments.jsonl"
        self.metadata_path = self.index_dir / "index_metadata.json"

        self.embeddings: Optional[np.ndarray] = None
        self.segments: List[Segment] = []
        self.metadata: Dict[str, Any] = {}

    def is_valid(self, expected_config: Dict[str, Any], current_segments: Optional[List[Segment]] = None) -> bool:
        """
        Verifica se o índice existente no disco é válido e compatível com a configuração atual.
        """
        is_valid, errors = validate_vector_index_files(
            self.embeddings_path,
            self.segments_path,
            self.metadata_path,
            expected_model=expected_config.get("embedding_model")
        )
        if not is_valid:
            return False

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if meta.get("embedding_model") != expected_config.get("embedding_model"):
                return False
            if meta.get("min_words") != expected_config.get("min_words", 8):
                return False
            if meta.get("max_characters") != expected_config.get("max_characters", 500):
                return False
            if meta.get("long_text_strategy") != expected_config.get("long_text_strategy", "chunk"):
                return False

            if current_segments is not None:
                corpus_hash = self._calculate_corpus_hash(current_segments)
                if meta.get("corpus_sha256") != corpus_hash:
                    return False

            return True
        except Exception:
            return False

    def build_and_save(
        self,
        embeddings: np.ndarray,
        segments: List[Segment],
        model_name: str,
        device: str,
        config: Dict[str, Any]
    ) -> None:
        """Salva atomicamente a matriz de embeddings, a lista de segmentos e os metadados no disco."""
        ensure_directory(self.index_dir)

        self.embeddings = embeddings
        self.segments = segments

        atomic_write_numpy(self.embeddings_path, embeddings)

        segments_lines = [json.dumps(seg.to_dict(), ensure_ascii=False) for seg in segments]
        atomic_write_text(self.segments_path, "\n".join(segments_lines) + "\n")

        unique_articles = len(set(s.article_id for s in segments))
        valid_segs = sum(1 for s in segments if getattr(s, "status", "valid") in ["valid_paragraph", "INGESTED", "valid"])
        corpus_hash = self._calculate_corpus_hash(segments)

        self.metadata = {
            "embedding_model": model_name,
            "vector_dim": embeddings.shape[1] if embeddings.ndim == 2 else 0,
            "total_segments": len(segments),
            "valid_segments": valid_segs,
            "total_articles": unique_articles,
            "device": device,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "min_words": config.get("min_words", 8),
            "min_characters": config.get("min_characters", 40),
            "max_characters": config.get("max_characters", 500),
            "long_text_strategy": config.get("long_text_strategy", "chunk"),
            "corpus_sha256": corpus_hash,
            "index_version": config.get("index_version", 1)
        }

        atomic_write_json(self.metadata_path, self.metadata)

    def append_vectors_and_segments(
        self,
        new_embeddings: np.ndarray,
        new_segments: List[Segment],
        model_name: str,
        device: str,
        config: Dict[str, Any]
    ) -> None:
        """
        Concatena novos embeddings e segmentos ao índice existente de forma incremental e atômica.
        Lança ValueError se o modelo for incompatível.
        """
        if not self.load():
            # Se o índice não existir, constrói do zero
            self.build_and_save(new_embeddings, new_segments, model_name, device, config)
            return

        # Checa incompatibilidade do modelo
        current_model = self.metadata.get("embedding_model")
        if current_model and current_model != model_name:
            raise ValueError(
                f"Modelo de embedding incompatível para atualização incremental: "
                f"existente '{current_model}', novo '{model_name}'. Reconstrução do índice é necessária."
            )

        if self.embeddings is None or self.embeddings.size == 0:
            combined_embeddings = new_embeddings
        else:
            combined_embeddings = np.vstack([self.embeddings, new_embeddings])

        combined_segments = self.segments + new_segments

        # Incrementa versão do índice
        config = dict(config)
        config["index_version"] = self.metadata.get("index_version", 1) + 1

        self.build_and_save(combined_embeddings, combined_segments, model_name, device, config)

    def load(self, mmap_mode: Optional[str] = "r") -> bool:
        """
        Carrega o índice vetorial existente a partir do disco.
        Por padrão utiliza mmap_mode="r" para mapeamento incremental em disco de grandes matrizes.
        """
        if not (self.embeddings_path.exists() and self.segments_path.exists() and self.metadata_path.exists()):
            return False

        try:
            self.embeddings = np.load(self.embeddings_path, mmap_mode=mmap_mode)

            self.segments = []
            with open(self.segments_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.segments.append(Segment.from_dict(data))

            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

            return True
        except Exception:
            return False

    def clear(self):
        """Apaga os arquivos de índice do disco."""
        for p in [self.embeddings_path, self.segments_path, self.metadata_path]:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        self.embeddings = None
        self.segments = []
        self.metadata = {}

    @staticmethod
    def _calculate_corpus_hash(segments: List[Segment]) -> str:
        """Calcula o hash compilado de todos os segmentos para validar o corpus."""
        hashes = []
        for s in segments:
            h = getattr(s, "text_sha256", None) or calculate_text_sha256(getattr(s, "text", ""))
            hashes.append(h)
        combined_hashes = "".join(sorted(hashes))
        return calculate_text_sha256(combined_hashes)
