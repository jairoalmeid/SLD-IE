"""
Leitor e extrator de PDFs baseado em PyMuPDF (fitz).
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF


class PDFReaderError(Exception):
    """Exceção para erros no processamento de arquivos PDF."""
    pass


def read_pdf(pdf_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Lê um PDF local, extrai o texto página por página e retorna os metadados do documento.

    Retorna:
        Tuple contendo:
        1. Lista de dicionários por página: [{"page": 1, "text": "..."}, ...]
        2. Dicionário de metadados internos do PDF (PyMuPDF doc.metadata)

    Lança:
        PDFReaderError em caso de PDF inexistente, corrompido, protegido ou sem camada de texto.
    """
    if not pdf_path.exists():
        raise PDFReaderError(f"Arquivo PDF não encontrado: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise PDFReaderError(f"Falha ao abrir PDF corrompido ou inválido ({pdf_path.name}): {e}")

    if doc.is_encrypted:
        raise PDFReaderError(f"O PDF está protegido por senha e não pode ser lido: {pdf_path.name}")

    if doc.page_count == 0:
        raise PDFReaderError(f"O arquivo PDF está vazio (0 páginas): {pdf_path.name}")

    pages_data: List[Dict[str, Any]] = []
    total_chars = 0

    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        text = page.get_text("text") or ""
        total_chars += len(text.strip())
        pages_data.append({
            "page": page_idx + 1,
            "text": text,
        })

    pdf_metadata = doc.metadata or {}
    doc.close()

    if total_chars == 0:
        raise PDFReaderError(
            f"OCR não implementado nesta versão. O PDF '{pdf_path.name}' não possui camada de texto extraível (pode ser um documento digitalizado por imagem)."
        )

    return pages_data, pdf_metadata


def read_pdf_content(pdf_path: Path) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
    """
    Lê o PDF local e extrai os metadados do documento, o texto completo concatenado e as páginas.
    Retorna: (doc_metadata, raw_text, pages_data)
    """
    from datetime import datetime
    from src.sld.utils.files import compute_file_hash

    pages_data, pdf_meta = read_pdf(pdf_path)
    full_text = "\n\n".join(p["text"] for p in pages_data)
    document_hash = compute_file_hash(pdf_path)

    doc_metadata = {
        "title": pdf_meta.get("title", pdf_path.stem),
        "author": pdf_meta.get("author", ""),
        "source_file": pdf_path.name,
        "document_hash": document_hash,
        "extraction_date": datetime.now().isoformat()
    }
    return doc_metadata, full_text, pages_data
