"""
Modelos de dados para artigos e metadados de ingestão.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ArticleMetadata:
    """Metadados identificados do artigo científico."""
    sld_id: str
    title: str = "não identificado"
    authors: List[str] = field(default_factory=list)
    doi: Optional[str] = None
    year: Optional[int] = None
    journal: str = "não identificado"
    language: str = "não identificado"
    source_pdf: str = ""
    source_path: str = ""
    pdf_sha256: str = ""
    processed_at: str = ""
    extraction_engine: str = "PyMuPDF"
    extraction_engine_version: str = ""
    references_removed: bool = False
    reference_start_page: Optional[int] = None
    metadata_warnings: List[str] = field(default_factory=list)

    def to_yaml_dict(self) -> Dict[str, Any]:
        """Converte metadados para dicionário estruturado para YAML."""
        return {
            "sld_id": self.sld_id,
            "title": self.title,
            "authors": self.authors if self.authors else ["não identificado"],
            "doi": self.doi,
            "year": self.year,
            "journal": self.journal,
            "language": self.language,
            "source_pdf": self.source_pdf,
            "source_path": self.source_path,
            "pdf_sha256": self.pdf_sha256,
            "processed_at": self.processed_at,
            "extraction_engine": self.extraction_engine,
            "extraction_engine_version": self.extraction_engine_version,
            "references_removed": self.references_removed,
            "reference_start_page": self.reference_start_page,
            "metadata_warnings": self.metadata_warnings,
        }


@dataclass
class ProcessedArticle:
    """Resultado do processamento completo de um artigo em PDF."""
    metadata: ArticleMetadata
    markdown_content: str
    markdown_path: str
    page_count: int
    char_count: int
    status: str  # "success", "warning", "error"
    error_message: Optional[str] = None
    processing_duration: float = 0.0
