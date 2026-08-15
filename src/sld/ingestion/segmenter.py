"""
Módulo de segmentação de texto Markdown em objetos ParagraphRecord com suporte a fragmentação por sentenças (chunks menores).
"""

import re
from pathlib import Path
from typing import List, Optional
from src.sld.models.classification import ParagraphRecord
from src.sld.utils.hashing import calculate_text_sha256


def split_text_into_chunks(text: str, max_characters: int = 500) -> List[str]:
    """
    Divide um bloco longo de texto por sentenças de modo que cada segmento 
    tenha no máximo `max_characters` caracteres.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []
    current_len = 0

    for s in sentences:
        s_len = len(s)
        if current_len + s_len > max_characters and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(s)
        current_len += s_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def segment_markdown_paragraphs(
    markdown_content: str,
    article_id: str,
    doc_id: Optional[str] = None,
    max_characters: int = 500
) -> List[ParagraphRecord]:
    """
    Segmente o conteúdo Markdown em objetos ParagraphRecord menores e concisos.
    - Atribui IDs no formato DOC000001_P000001
    - Se o parágrafo for maior que max_characters, realiza fragmentação por sentenças.
    - Calcula hash SHA-256 do texto de cada fragmento.
    """
    d_id = doc_id or article_id
    lines = markdown_content.split("\n")
    paragraph_blocks: List[str] = []
    current_block: List[str] = []

    for line in lines:
        stripped = line.strip()
        # Ignora cabeçalhos YAML frontmatter e marcadores
        if stripped.startswith("---") or stripped.startswith("<!--") or stripped.startswith("#"):
            if current_block:
                paragraph_blocks.append(" ".join(current_block))
                current_block = []
            continue

        if not stripped:
            if current_block:
                paragraph_blocks.append(" ".join(current_block))
                current_block = []
            continue

        current_block.append(stripped)

    if current_block:
        paragraph_blocks.append(" ".join(current_block))

    records: List[ParagraphRecord] = []
    p_counter = 0

    for block in paragraph_blocks:
        clean_text = block.strip()
        if len(clean_text.split()) < 5:  # descarta fragmentos extremamente curtos
            continue

        p_counter += 1
        base_para_id = f"{d_id}_P{p_counter:06d}"

        # Se o parágrafo exceder o tamanho máximo, divide em sentenças menores
        if len(clean_text) > max_characters:
            sub_chunks = split_text_into_chunks(clean_text, max_characters=max_characters)
            for sub_idx, sub_text in enumerate(sub_chunks, start=1):
                chunk_id = f"{base_para_id}_S{sub_idx:02d}"
                text_hash = calculate_text_sha256(sub_text)
                records.append(
                    ParagraphRecord(
                        paragraph_id=chunk_id,
                        article_id=article_id,
                        paragraph_index=len(records) + 1,
                        text=sub_text,
                        text_sha256=text_hash,
                        status="RAW"
                    )
                )
        else:
            text_hash = calculate_text_sha256(clean_text)
            records.append(
                ParagraphRecord(
                    paragraph_id=base_para_id,
                    article_id=article_id,
                    paragraph_index=len(records) + 1,
                    text=clean_text,
                    text_sha256=text_hash,
                    status="RAW"
                )
            )

    return records
