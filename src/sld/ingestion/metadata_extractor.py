"""
Extrator de metadados bibliográficos de artigos em PDF.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz
from src.sld.models.article import ArticleMetadata
from src.sld.utils.hashing import generate_article_id, calculate_file_sha256


# Regex para identificação de DOI
DOI_REGEX = re.compile(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+')
# Regex para identificação de Ano (19xx a 20xx)
YEAR_REGEX = re.compile(r'\b(19\d{2}|20[0-2]\d)\b')


def extract_metadata(
    pdf_path: Path,
    pdf_sha256: str,
    pages_data: List[Dict[str, Any]],
    raw_pdf_metadata: Dict[str, Any]
) -> ArticleMetadata:
    """
    Tenta identificar metadados do artigo científico (Título, Autores, DOI, Ano, Periódico, Idioma).
    Se uma informação não for identificada com segurança, registra 'não identificado' ou null.
    """
    warnings: List[str] = []
    sld_id = generate_article_id(pdf_sha256, pdf_path.name)

    # 1. DOI
    doi = _extract_doi(pages_data, raw_pdf_metadata)
    if not doi:
        warnings.append("DOI não identificado no texto ou metadados.")

    # 2. Ano
    year = _extract_year(pages_data, raw_pdf_metadata)
    if not year:
        warnings.append("Ano de publicação não identificado com certeza.")

    # 3. Título
    title = _extract_title(pages_data, raw_pdf_metadata, pdf_path.stem)

    # 4. Autores
    authors = _extract_authors(pages_data, raw_pdf_metadata)
    if not authors:
        warnings.append("Autores não identificados explicitamente nos metadados.")

    # 5. Periódico / Journal
    journal = raw_pdf_metadata.get("subject") or raw_pdf_metadata.get("keywords") or "não identificado"
    if journal == "não identificado":
        journal = _infer_journal(pages_data)

    # 6. Idioma
    language = _infer_language(pages_data)

    processed_at = datetime.now().isoformat()

    return ArticleMetadata(
        sld_id=sld_id,
        title=title,
        authors=authors,
        doi=doi,
        year=year,
        journal=journal,
        language=language,
        source_pdf=pdf_path.name,
        source_path=str(pdf_path.resolve()),
        pdf_sha256=pdf_sha256,
        processed_at=processed_at,
        extraction_engine="PyMuPDF",
        extraction_engine_version=fitz.__version__,
        references_removed=False,
        reference_start_page=None,
        metadata_warnings=warnings,
    )


def _extract_doi(pages_data: List[Dict[str, Any]], raw_metadata: Dict[str, Any]) -> Optional[str]:
    """Busca padrão de DOI nos metadados brutos e nas primeiras páginas do PDF."""
    for key, val in raw_metadata.items():
        if isinstance(val, str):
            match = DOI_REGEX.search(val)
            if match:
                return match.group(0).rstrip(".,;")

    # Procura nas 3 primeiras páginas
    for page_info in pages_data[:3]:
        match = DOI_REGEX.search(page_info.get("text", ""))
        if match:
            return match.group(0).rstrip(".,;")

    return None


def _extract_year(pages_data: List[Dict[str, Any]], raw_metadata: Dict[str, Any]) -> Optional[int]:
    """Busca ano de publicação nos metadados brutos e no texto de abertura do PDF."""
    # Metadados PDF Date ex: 'D:20230512...'
    creation_date = raw_metadata.get("creationDate", "")
    if isinstance(creation_date, str) and len(creation_date) >= 6:
        match = re.search(r'20\d{2}|19\d{2}', creation_date)
        if match:
            return int(match.group(0))

    # Texto da 1ª página
    if pages_data:
        text_p1 = pages_data[0].get("text", "")
        # Procura datas perto de palavras como "Received", "Accepted", "Published", "©", "Copyright"
        match_pub = re.search(r'(?:Published|Accepted|Copyright|©|\b19\d{2}\b|\b20\d{2}\b)[^\n]*?(19\d{2}|20[0-2]\d)', text_p1, re.IGNORECASE)
        if match_pub:
            return int(match_pub.group(1))

        years = YEAR_REGEX.findall(text_p1[:1500])
        if years:
            # Retorna o primeiro ano razoável
            current_year = datetime.now().year
            valid_years = [int(y) for y in years if 1900 <= int(y) <= current_year]
            if valid_years:
                return valid_years[0]

    return None


def _extract_title(pages_data: List[Dict[str, Any]], raw_metadata: Dict[str, Any], fallback_stem: str) -> str:
    """Extrai título dos metadados ou do topo da primeira página."""
    meta_title = raw_metadata.get("title", "").strip()
    if meta_title and len(meta_title) > 5 and not meta_title.lower().endswith(".pdf"):
        return meta_title

    if pages_data:
        text_p1 = pages_data[0].get("text", "").strip()
        lines = [line.strip() for line in text_p1.split("\n") if line.strip()]
        # Procura pelas primeiras linhas com bom tamanho que não pareçam cabeçalho de revista
        candidate_lines = []
        for line in lines[:8]:
            if len(line) > 10 and not re.match(r'^(ISSN|DOI|http|Volume|Vol\.|Issue|Page|Página)', line, re.IGNORECASE):
                candidate_lines.append(line)
                if len(" ".join(candidate_lines)) > 30:
                    break
        if candidate_lines:
            return " ".join(candidate_lines)

    return fallback_stem.replace("_", " ").title()


def _extract_authors(pages_data: List[Dict[str, Any]], raw_metadata: Dict[str, Any]) -> List[str]:
    """Extrai lista de autores dos metadados do PDF ou de heurísticas na 1ª página."""
    author_str = raw_metadata.get("author", "").strip()
    if author_str and len(author_str) > 2:
        # Separa por vírgula, ponto e vírgula ou 'and'
        authors = re.split(r'[,;]|\band\b', author_str)
        cleaned_authors = [a.strip() for a in authors if len(a.strip()) > 2]
        if cleaned_authors:
            return cleaned_authors

    return []


def _infer_journal(pages_data: List[Dict[str, Any]]) -> str:
    """Tenta identificar nome de revista científica nas primeiras linhas da página 1."""
    if not pages_data:
        return "não identificado"
    text_p1 = pages_data[0].get("text", "")
    lines = [l.strip() for l in text_p1.split("\n") if l.strip()]
    for line in lines[:5]:
        if any(kw in line.lower() for kw in ["journal of", "proceedings of", "transactions on", "annals of", "review of", "ieee", "nature", "science"]):
            return line
    return "não identificado"


def _infer_language(pages_data: List[Dict[str, Any]]) -> str:
    """Identifica idioma predominante com base em palavras funcionais comuns."""
    if not pages_data:
        return "não identificado"
    sample_text = " ".join([p.get("text", "") for p in pages_data[:2]]).lower()

    pt_words = len(re.findall(r'\b(de|do|da|em|para|com|como|por|referências|resumo|artigo)\b', sample_text))
    en_words = len(re.findall(r'\b(the|of|and|in|to|for|with|as|by|references|abstract|paper)\b', sample_text))
    es_words = len(re.findall(r'\b(el|la|los|las|de|en|por|con|como|resumen|referencias)\b', sample_text))

    counts = {"pt": pt_words, "en": en_words, "es": es_words}
    best_lang = max(counts, key=counts.get)
    if counts[best_lang] > 5:
        return best_lang
    return "não identificado"
