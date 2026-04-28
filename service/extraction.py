import os
import tempfile

import fitz
import pymupdf4llm


def extract_pdf_text(file_bytes: bytes) -> tuple[str, int]:
    """Extracts markdown text and page count from PDF bytes without saving anything."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(file_bytes)
        temp_path = f.name
    try:
        doc = fitz.open(temp_path)
        pages = len(doc)
        doc.close()
        text = pymupdf4llm.to_markdown(temp_path)
    finally:
        os.unlink(temp_path)
    return text, pages
