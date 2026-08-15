"""
Módulo de extração segura de arquivos ZIP contendo PDFs, com proteção anti-ZIP Slip e rastreabilidade de proveniência.
"""

import os
import zipfile
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple


class ZipSecurityError(Exception):
    """Exceção lançada quando uma tentativa de Path Traversal (ZIP Slip) é detectada."""
    pass


def extract_zip_safely(zip_path: Path, target_dir: Path) -> List[Dict[str, Any]]:
    """
    Descompacta arquivos PDF de um arquivo ZIP de maneira segura.
    - Previne vulnerabilidade ZIP Slip.
    - Ignora arquivos ocultos (__MACOSX, .DS_Store).
    - Retorna lista de dicionários com metadados de proveniência de cada PDF extraído.
    """
    zip_path = Path(zip_path).expanduser().resolve()
    target_dir = Path(target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    extracted_pdf_records: List[Dict[str, Any]] = []

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            # Ignora diretórios e arquivos ocultos / de sistema
            if member.is_dir() or "__MACOSX" in member.filename or os.path.basename(member.filename).startswith("."):
                continue

            # Apenas arquivos PDF
            if not member.filename.lower().endswith(".pdf"):
                continue

            # Proteção anti-ZIP Slip
            destination_path = (target_dir / member.filename).resolve()
            if not str(destination_path).startswith(str(target_dir)):
                raise ZipSecurityError(f"Tentativa de ZIP Slip detectada no arquivo: {member.filename}")

            # Garante diretórios pai
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            # Extrai o arquivo
            with zip_ref.open(member) as source, open(destination_path, "wb") as target:
                shutil.copyfileobj(source, target)

            extracted_pdf_records.append({
                "extracted_path": destination_path,
                "source_type": "ZIP",
                "zip_source": zip_path.name,
                "original_pdf_name": os.path.basename(member.filename),
            })

    return extracted_pdf_records


def find_all_inputs(input_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    Varre o diretório de entrada encontrando PDFs diretos e arquivos ZIP.
    Retorna (pdf_paths, zip_paths).
    """
    input_dir = Path(input_dir).expanduser().resolve()
    if not input_dir.exists():
        return [], []

    pdfs = [p.resolve() for p in input_dir.glob("*.pdf")] + [p.resolve() for p in input_dir.glob("*.PDF")]
    zips = [p.resolve() for p in input_dir.glob("*.zip")] + [p.resolve() for p in input_dir.glob("*.ZIP")]

    # Duplicatas ordenadas por caminho
    unique_pdfs = sorted(list(set(pdfs)))
    unique_zips = sorted(list(set(zips)))

    return unique_pdfs, unique_zips
