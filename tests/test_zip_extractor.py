"""
Testes unitários para extração de arquivos ZIP e segurança anti-ZIP Slip.
"""

import os
import zipfile
import pytest
from pathlib import Path
from src.sld.ingestion.zip_extractor import extract_zip_safely, ZipSecurityError


def test_extract_zip_safely_valid_pdf(tmp_path: Path):
    """Testa descompactação de ZIP válido com PDFs."""
    zip_file = tmp_path / "valid_sample.zip"
    extract_target = tmp_path / "extracted"

    with zipfile.ZipFile(zip_file, "w") as z:
        z.writestr("test_article.pdf", b"%PDF-1.4 dummy pdf content")

    records = extract_zip_safely(zip_file, extract_target)

    assert len(records) == 1
    assert records[0]["source_type"] == "ZIP"
    assert records[0]["original_pdf_name"] == "test_article.pdf"
    assert records[0]["extracted_path"].exists()


def test_zip_slip_prevention(tmp_path: Path):
    """Testa se tentativas de Path Traversal (ZIP Slip) disparam exceção ZipSecurityError."""
    zip_file = tmp_path / "malicious.zip"
    extract_target = tmp_path / "target"

    with zipfile.ZipFile(zip_file, "w") as z:
        z.writestr("../evil.pdf", b"%PDF-1.4 malicious pdf content")

    with pytest.raises(ZipSecurityError):
        extract_zip_safely(zip_file, extract_target)
