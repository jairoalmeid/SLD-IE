"""
Funções utilitárias para geração de hashes SHA-256 e identificadores estáveis.
"""

import hashlib
from pathlib import Path


def calculate_file_sha256(file_path: Path) -> str:
    """Calcula o hash SHA-256 de um arquivo local."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def calculate_text_sha256(text: str) -> str:
    """Calcula o hash SHA-256 de uma string de texto UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_article_id(file_sha256: str, filename: str = "") -> str:
    """
    Gere um identificador único e estável para o artigo.
    Exemplo: SLD-A1B2C3D4
    """
    prefix = file_sha256[:8].upper()
    return f"SLD-{prefix}"
