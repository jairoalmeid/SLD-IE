"""
Módulo de Ingestão de PDFs e Conversão para Markdown.
"""

from .pdf_reader import read_pdf, PDFReaderError
from .metadata_extractor import extract_metadata
from .text_cleaner import normalize_characters_and_spaces, recompose_hyphenation, clean_pages
from .reference_remover import remove_references, ReferenceRemovalDecision
from .markdown_writer import generate_markdown_content, write_markdown_file

__all__ = [
    "read_pdf",
    "PDFReaderError",
    "extract_metadata",
    "normalize_characters_and_spaces",
    "recompose_hyphenation",
    "clean_pages",
    "remove_references",
    "ReferenceRemovalDecision",
    "generate_markdown_content",
    "write_markdown_file",
]
