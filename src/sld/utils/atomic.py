"""
Utilitários de gravação atômica em disco (escrita em .tmp + substituição atômica)
e validação transacional do índice vetorial para prevenção de corrupção de dados.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from src.sld.utils.files import ensure_directory


def atomic_write_json(file_path: Path, data: Any, indent: int = 2) -> None:
    """Escreve um arquivo JSON de forma atômica utilizando um arquivo temporário intermediário."""
    file_path = Path(file_path).expanduser().resolve()
    ensure_directory(file_path.parent)
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    tmp_path.replace(file_path)


def atomic_write_text(file_path: Path, text: str, encoding: str = "utf-8") -> None:
    """Escreve um arquivo de texto de forma atômica utilizando um arquivo temporário intermediário."""
    file_path = Path(file_path).expanduser().resolve()
    ensure_directory(file_path.parent)
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")

    with open(tmp_path, "w", encoding=encoding) as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())

    tmp_path.replace(file_path)


def atomic_write_numpy(file_path: Path, arr: np.ndarray) -> None:
    """Salva uma matriz NumPy .npy de forma atômica utilizando um arquivo temporário intermediário."""
    file_path = Path(file_path).expanduser().resolve()
    ensure_directory(file_path.parent)
    tmp_path = file_path.with_suffix(".npy.tmp")

    with open(tmp_path, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())

    tmp_path.replace(file_path)


def validate_vector_index_files(
    embeddings_path: Path,
    segments_path: Path,
    metadata_path: Path,
    expected_model: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Realiza validação transacional dos arquivos do índice vetorial em disco.
    Retorna (is_valid, list_of_errors).
    """
    errors: List[str] = []

    if not embeddings_path.exists():
        errors.append(f"Arquivo de embeddings ausente: {embeddings_path.name}")
    if not segments_path.exists():
        errors.append(f"Arquivo de segmentos ausente: {segments_path.name}")
    if not metadata_path.exists():
        errors.append(f"Arquivo de metadados do índice ausente: {metadata_path.name}")

    if errors:
        return False, errors

    try:
        embeddings = np.load(embeddings_path)
    except Exception as e:
        errors.append(f"Falha ao carregar matriz de embeddings ({embeddings_path.name}): {e}")
        return False, errors

    segment_count = 0
    segment_ids = set()
    dup_ids = set()

    try:
        with open(segments_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str:
                    continue
                segment_count += 1
                try:
                    data = json.loads(line_str)
                    seg_id = data.get("paragraph_id") or data.get("segment_id") or f"idx_{line_idx}"
                    if seg_id in segment_ids:
                        dup_ids.add(seg_id)
                    segment_ids.add(seg_id)
                except Exception as e:
                    errors.append(f"Linha {line_idx} corrompida em {segments_path.name}: {e}")
    except Exception as e:
        errors.append(f"Falha ao ler {segments_path.name}: {e}")
        return False, errors

    if dup_ids:
        errors.append(f"Encontrados {len(dup_ids)} IDs de parágrafos/segmentos duplicados em {segments_path.name}")

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        errors.append(f"Falha ao ler metadados do índice ({metadata_path.name}): {e}")
        return False, errors

    n_vectors = embeddings.shape[0] if embeddings.ndim >= 1 else 0
    vector_dim = embeddings.shape[1] if embeddings.ndim == 2 else 0

    if n_vectors != segment_count:
        errors.append(
            f"Divergência entre matriz de embeddings ({n_vectors} vetores) e quantidade de segmentos ({segment_count} registros)"
        )

    meta_total = meta.get("total_segments", meta.get("valid_segments", n_vectors))
    if meta_total != n_vectors:
        errors.append(f"Metadados registram {meta_total} vetores, mas a matriz contém {n_vectors}")

    if expected_model and meta.get("embedding_model") != expected_model:
        errors.append(
            f"Modelo incompatível no índice: gravado '{meta.get('embedding_model')}', esperado '{expected_model}'"
        )

    return (len(errors) == 0), errors
