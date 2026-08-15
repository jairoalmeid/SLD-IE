"""
Segmentador científico de artigos em Markdown orientados por parágrafos com proveniência persistente e chunking.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import yaml
from src.sld.models.search_result import Segment
from src.sld.utils.hashing import calculate_text_sha256


def parse_markdown_with_yaml(markdown_content: str) -> Tuple[Dict[str, Any], str]:
    """Extrai o YAML Frontmatter e separa o corpo principal do texto em Markdown."""
    frontmatter = {}
    body = markdown_content

    if markdown_content.startswith("---"):
        parts = markdown_content.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
            except Exception:
                frontmatter = {}
            body = parts[2]

    return frontmatter, body


def segment_markdown(
    markdown_content: str,
    markdown_path: Path,
    min_words: int = 10,
    min_characters: int = 50,
    max_characters: int = 1500,
    long_text_strategy: str = "chunk",  # "chunk", "truncate", "skip"
    overlap: int = 100
) -> List[Segment]:
    """
    Segmente o corpo em parágrafos com rastreabilidade estruturada (`article_id/section_id/paragraph_id/paragraph_hash`).
    Parágrafos curtos são classificados sem exclusão silenciosa.
    """
    metadata_dict, body = parse_markdown_with_yaml(markdown_content)

    article_id = metadata_dict.get("sld_id", f"SLD-{markdown_path.stem[:8].upper()}")
    source_pdf = metadata_dict.get("source_pdf", markdown_path.name)
    title = metadata_dict.get("title", markdown_path.stem)

    segments: List[Segment] = []
    current_page = 1
    current_section = "Introdução / Geral"
    current_subsection = ""

    page_marker_pattern = re.compile(r'<!--\s*page:\s*(\d+)\s*-->')
    header_pattern = re.compile(r'^(#{1,6})\s+(.*)$')

    lines = body.split("\n")
    paragraph_buffer: List[str] = []
    paragraph_start_page = current_page
    paragraph_counter = 0

    def process_paragraph_block(text_block: str, p_start: int, p_end: int, sec_name: str, subsec_name: str):
        nonlocal paragraph_counter
        text_clean = text_block.strip()
        if not text_clean:
            return

        paragraph_counter += 1
        para_id = f"P{paragraph_counter:04d}"
        para_hash = calculate_text_sha256(text_clean)
        words = len(text_clean.split())
        chars = len(text_clean)

        # 1. Validação de fragmentos curtos (Classificação Não-Destrutiva)
        if words < min_words or chars < min_characters:
            seg_id = f"{article_id}_{para_id}"
            segments.append(
                Segment(
                    segment_id=seg_id,
                    article_id=article_id,
                    paragraph_id=para_id,
                    source_pdf=source_pdf,
                    markdown_path=str(markdown_path.resolve()),
                    title=title,
                    section=sec_name,
                    subsection=subsec_name,
                    page_start=p_start,
                    page_end=p_end,
                    text=text_clean,
                    text_sha256=para_hash,
                    word_count=words,
                    char_count=chars,
                    status="excluded_short_fragment",
                    exclusion_reason=f"Fragmento curto ({chars} chars, {words} palavras < min {min_characters}/{min_words})",
                    segment_index_in_doc=len(segments),
                )
            )
            return

        # 2. Tratamento de parágrafos longos
        if chars > max_characters:
            if long_text_strategy == "chunk":
                chunks = _split_paragraph_into_chunks(text_clean, max_characters, overlap)
                total_chunks = len(chunks)
                for chunk_idx, chunk_text in enumerate(chunks, start=1):
                    chunk_id = f"{para_id}_C{chunk_idx:02d}"
                    seg_id = f"{article_id}_{chunk_id}"
                    c_hash = calculate_text_sha256(chunk_text)
                    segments.append(
                        Segment(
                            segment_id=seg_id,
                            article_id=article_id,
                            paragraph_id=para_id,
                            source_pdf=source_pdf,
                            markdown_path=str(markdown_path.resolve()),
                            title=title,
                            section=sec_name,
                            subsection=subsec_name,
                            page_start=p_start,
                            page_end=p_end,
                            text=chunk_text,
                            text_sha256=c_hash,
                            word_count=len(chunk_text.split()),
                            char_count=len(chunk_text),
                            status="valid_paragraph",
                            is_chunk=True,
                            chunk_id=chunk_id,
                            chunk_index=chunk_idx,
                            total_chunks=total_chunks,
                            segment_index_in_doc=len(segments),
                        )
                    )
            elif long_text_strategy == "truncate":
                truncated_text = text_clean[:max_characters]
                seg_id = f"{article_id}_{para_id}"
                segments.append(
                    Segment(
                        segment_id=seg_id,
                        article_id=article_id,
                        paragraph_id=para_id,
                        source_pdf=source_pdf,
                        markdown_path=str(markdown_path.resolve()),
                        title=title,
                        section=sec_name,
                        subsection=subsec_name,
                        page_start=p_start,
                        page_end=p_end,
                        text=truncated_text,
                        text_sha256=calculate_text_sha256(truncated_text),
                        word_count=len(truncated_text.split()),
                        char_count=len(truncated_text),
                        status="valid_paragraph",
                        segment_index_in_doc=len(segments),
                    )
                )
            else:  # "skip"
                seg_id = f"{article_id}_{para_id}"
                segments.append(
                    Segment(
                        segment_id=seg_id,
                        article_id=article_id,
                        paragraph_id=para_id,
                        source_pdf=source_pdf,
                        markdown_path=str(markdown_path.resolve()),
                        title=title,
                        section=sec_name,
                        subsection=subsec_name,
                        page_start=p_start,
                        page_end=p_end,
                        text=text_clean,
                        text_sha256=para_hash,
                        word_count=words,
                        char_count=chars,
                        status="excluded_too_long",
                        exclusion_reason=f"Parágrafo excede limite máximo ({chars} > {max_characters} chars)",
                        segment_index_in_doc=len(segments),
                    )
                )
        else:
            # Parágrafo válido de tamanho normal
            seg_id = f"{article_id}_{para_id}"
            segments.append(
                Segment(
                    segment_id=seg_id,
                    article_id=article_id,
                    paragraph_id=para_id,
                    source_pdf=source_pdf,
                    markdown_path=str(markdown_path.resolve()),
                    title=title,
                    section=sec_name,
                    subsection=subsec_name,
                    page_start=p_start,
                    page_end=p_end,
                    text=text_clean,
                    text_sha256=para_hash,
                    word_count=words,
                    char_count=chars,
                    status="valid_paragraph",
                    segment_index_in_doc=len(segments),
                )
            )

    for line in lines:
        line_str = line.strip()

        page_match = page_marker_pattern.search(line_str)
        if page_match:
            if paragraph_buffer:
                process_paragraph_block(" ".join(paragraph_buffer), paragraph_start_page, current_page, current_section, current_subsection)
                paragraph_buffer = []
            current_page = int(page_match.group(1))
            paragraph_start_page = current_page
            continue

        header_match = header_pattern.match(line_str)
        if header_match:
            if paragraph_buffer:
                process_paragraph_block(" ".join(paragraph_buffer), paragraph_start_page, current_page, current_section, current_subsection)
                paragraph_buffer = []
            level = len(header_match.group(1))
            h_text = header_match.group(2).strip()
            if level <= 2:
                current_section = h_text
                current_subsection = ""
            else:
                current_subsection = h_text
            paragraph_start_page = current_page
            continue

        if not line_str:
            if paragraph_buffer:
                process_paragraph_block(" ".join(paragraph_buffer), paragraph_start_page, current_page, current_section, current_subsection)
                paragraph_buffer = []
                paragraph_start_page = current_page
            continue

        if not paragraph_buffer:
            paragraph_start_page = current_page
        paragraph_buffer.append(line_str)

    if paragraph_buffer:
        process_paragraph_block(" ".join(paragraph_buffer), paragraph_start_page, current_page, current_section, current_subsection)

    return segments


def _split_paragraph_into_chunks(text: str, max_chars: int, overlap: int) -> List[str]:
    """Divide um parágrafo por sentenças respeitando o limite máximo e sobreposição."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current_chunk = overlap_sentences
            current_len = overlap_len

        current_chunk.append(sentence)
        current_len += len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
