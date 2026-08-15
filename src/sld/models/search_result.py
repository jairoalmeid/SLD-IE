"""
Modelos de dados para segmentos de texto, chunks e resultados enriquecidos da busca semântica.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Segment:
    """Representa um parágrafo/segmento de texto extraído com rastreabilidade persistente."""
    segment_id: str  # e.g., SLD-A1B2_P0001
    article_id: str
    paragraph_id: str  # e.g., P0001
    source_pdf: str
    markdown_path: str
    title: str
    section: str
    subsection: str
    page_start: int
    page_end: int
    text: str
    text_sha256: str  # paragraph_hash
    word_count: int = 0
    char_count: int = 0
    status: str = "valid_paragraph"  # "valid_paragraph", "excluded_short_fragment"
    exclusion_reason: Optional[str] = None
    is_chunk: bool = False
    chunk_id: Optional[str] = None  # e.g., P0001_C01
    chunk_index: int = 0
    total_chunks: int = 1
    segment_index_in_doc: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "article_id": self.article_id,
            "paragraph_id": self.paragraph_id,
            "source_pdf": self.source_pdf,
            "markdown_path": self.markdown_path,
            "title": self.title,
            "section": self.section,
            "subsection": self.subsection,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "status": self.status,
            "exclusion_reason": self.exclusion_reason or "",
            "is_chunk": self.is_chunk,
            "chunk_id": self.chunk_id or "",
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "segment_index_in_doc": self.segment_index_in_doc,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Segment":
        return cls(
            segment_id=data["segment_id"],
            article_id=data["article_id"],
            paragraph_id=data.get("paragraph_id", "P0000"),
            source_pdf=data["source_pdf"],
            markdown_path=data["markdown_path"],
            title=data.get("title", "não identificado"),
            section=data.get("section", "Geral"),
            subsection=data.get("subsection", ""),
            page_start=data.get("page_start", 1),
            page_end=data.get("page_end", 1),
            text=data["text"],
            text_sha256=data.get("text_sha256", ""),
            word_count=data.get("word_count", len(data["text"].split())),
            char_count=data.get("char_count", len(data["text"])),
            status=data.get("status", "valid_paragraph"),
            exclusion_reason=data.get("exclusion_reason"),
            is_chunk=data.get("is_chunk", False),
            chunk_id=data.get("chunk_id"),
            chunk_index=data.get("chunk_index", 0),
            total_chunks=data.get("total_chunks", 1),
            segment_index_in_doc=data.get("segment_index_in_doc", 0),
        )


@dataclass
class SearchResult:
    """Resultado individual ranqueado de uma recuperação semântica com decomposição por âncora."""
    rank: int
    aggregate_score: float
    article_id: str
    paragraph_id: str
    paragraph_hash: str
    title: str
    authors: List[str]
    source_pdf: str
    section: str
    subsection: str
    page_range: str
    segment_id: str
    chunk_id: Optional[str]
    text: str
    anchor_scores: Dict[str, float] = field(default_factory=dict)  # {"Q1": 0.82, "Q2": 0.45, ...}
    best_anchor_id: str = "Q1"
    best_anchor_text: str = ""
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    markdown_path: str = ""
    run_id: str = ""
    threshold_used: float = 0.50
    selected: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Formato tabular detalhado para exportação CSV."""
        row = {
            "Rank": self.rank,
            "Aggregate Score": round(self.aggregate_score, 4),
            "ID Artigo": self.article_id,
            "ID Parágrafo": self.paragraph_id,
            "ID Chunk": self.chunk_id or "N/A",
            "Hash Parágrafo": self.paragraph_hash[:12],
            "Título Artigo": self.title,
            "Autores": ", ".join(self.authors) if self.authors else "não identificado",
            "PDF Origem": self.source_pdf,
            "Seção": self.section,
            "Página(s)": self.page_range,
            "Melhor Âncoras": self.best_anchor_id,
            "Texto Parágrafo": self.text,
            "Contexto Anterior": self.context_before or "",
            "Contexto Posterior": self.context_after or "",
            "Caminho Markdown": self.markdown_path,
            "Run ID": self.run_id,
            "Threshold": self.threshold_used,
            "Selecionado": self.selected,
        }

        # Adiciona colunas para cada score de âncora individual
        for anchor_id, score in self.anchor_scores.items():
            row[f"Similarity_{anchor_id}"] = round(score, 4)

        return row
