"""
Serviço de geração de Sentence Embeddings com suporte a Ollama local (nomic-embed-text) e SentenceTransformers.
"""

import os
import sys
import platform
import requests
from typing import List, Union, Dict, Any, Optional
import numpy as np

# Evita race conditions em tokenizers Rust / PyTorch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import transformers
import sentence_transformers
from sentence_transformers import SentenceTransformer
from config.settings import DEFAULT_EMBEDDING_MODEL, OLLAMA_EMBED_URL, BATCH_SIZE
from src.sld.models.experiment import EnvironmentMetadata


class EmbeddingService:
    """Gerencia modelos de Sentence Embeddings locais via Ollama API ou SentenceTransformers."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, model_revision: str = "main"):
        self.model_name = model_name
        self.model_revision = model_revision
        self.is_ollama = self._check_is_ollama(model_name)
        self.device = "ollama (http://localhost:11434)" if self.is_ollama else self._detect_device()
        self._model: Union[SentenceTransformer, None] = None
        self._ollama_dim: Optional[int] = None

    @staticmethod
    def _check_is_ollama(name: str) -> bool:
        clean = name.lower().strip()
        return "nomic" in clean or "ollama" in clean or clean in ["nomic-embed-text", "nomic-embed-text:latest"]

    def _detect_device(self) -> str:
        """Detecta se CUDA, MPS (Apple Silicon) ou CPU está disponível de forma segura."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                test_tensor = torch.zeros(1, device="mps")
                del test_tensor
                return "mps"
            except Exception:
                return "cpu"
        return "cpu"

    @property
    def model(self) -> SentenceTransformer:
        """Carrega o modelo SentenceTransformer (se não for Ollama)."""
        if self._model is None and not self.is_ollama:
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    revision=self.model_revision if self.model_revision != "main" else None
                )
            except Exception:
                self.device = "cpu"
                self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def encode(
        self,
        texts: List[str],
        batch_size: int = BATCH_SIZE,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Gera a matriz de embeddings para uma lista de textos via Ollama ou SentenceTransformers.
        Retorna np.ndarray de formato (len(texts), vector_dim), dtype=float32.
        """
        if not texts:
            return np.empty((0, self.get_vector_dimension()), dtype=np.float32)

        if self.is_ollama:
            return self._encode_ollama(texts, batch_size=batch_size, normalize=normalize)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

        return embeddings.astype(np.float32)

    def get_embeddings(
        self,
        texts: List[str],
        batch_size: int = BATCH_SIZE,
        normalize: bool = True
    ) -> np.ndarray:
        """Alias para encode(). Gera a matriz de embeddings para uma lista de textos."""
        return self.encode(texts, batch_size=batch_size, normalize=normalize)

    def _encode_ollama(
        self,
        texts: List[str],
        batch_size: int = BATCH_SIZE,
        normalize: bool = True
    ) -> np.ndarray:
        """Processa inferência via Ollama local API /api/embed."""
        ollama_model_name = self.model_name.replace("ollama/", "")
        all_vecs: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {"model": ollama_model_name, "input": batch}
            try:
                resp = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                embeddings_batch = data.get("embeddings", [])
                all_vecs.extend(embeddings_batch)
            except Exception as e:
                # Fallback para endpoint legado /api/embeddings por item se batch falhar
                for text in batch:
                    legacy_payload = {"model": ollama_model_name, "prompt": text}
                    l_resp = requests.post("http://localhost:11434/api/embeddings", json=legacy_payload, timeout=60)
                    l_resp.raise_for_status()
                    all_vecs.append(l_resp.json().get("embedding", []))

        arr = np.array(all_vecs, dtype=np.float32)
        if arr.size > 0:
            self._ollama_dim = arr.shape[1]

        if normalize and arr.size > 0:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms

        return arr.astype(np.float32)

    def encode_queries(self, queries: List[str], normalize: bool = True) -> np.ndarray:
        """Gera a matriz de embeddings para um conjunto de sentenças de consulta/âncoras."""
        return self.encode(queries, normalize=normalize)

    def get_vector_dimension(self) -> int:
        """Retorna a dimensão vetorial do modelo."""
        if self.is_ollama:
            if self._ollama_dim is not None:
                return self._ollama_dim
            # Tenta detectar via exemplo
            sample_vec = self._encode_ollama(["test"], batch_size=1, normalize=False)
            return sample_vec.shape[1] if sample_vec.size > 0 else 768

        dim = self.model.get_sentence_embedding_dimension()
        return dim if dim is not None else 768

    def get_max_sequence_length(self) -> int:
        """Retorna o comprimento máximo de sequência aceito pelo modelo (tokens)."""
        if self.is_ollama:
            return 2048  # nomic-embed-text aceita até 2048 tokens
        try:
            return getattr(self.model, "max_seq_length", 128)
        except Exception:
            return 128

    def get_environment_metadata(self) -> EnvironmentMetadata:
        """Retorna o snapshot completo de metadados do ambiente de execução."""
        cuda_name = None
        if torch.cuda.is_available():
            cuda_name = torch.cuda.get_device_name(0)

        dev_str = f"Ollama Local ({self.model_name})" if self.is_ollama else self.device

        return EnvironmentMetadata(
            python_version=sys.version.split()[0],
            torch_version=torch.__version__,
            transformers_version=transformers.__version__,
            sentence_transformers_version=sentence_transformers.__version__,
            device=dev_str,
            cpu_info=f"{platform.processor()} ({platform.machine()})",
            cuda_device_name=cuda_name,
        )
