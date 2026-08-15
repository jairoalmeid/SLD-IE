"""
Mecanismo conservador de remoção da seção de referências bibliográficas.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Union


REF_HEADER_TITLES = [
    "references",
    "referências",
    "referencias",
    "referências bibliográficas",
    "referencias bibliograficas",
    "bibliography",
    "literatura citada",
    "works cited",
]


@dataclass
class ReferenceRemovalDecision:
    """Registra a decisão tomada sobre a remoção de referências para fins de auditoria."""
    references_removed: bool
    detected_title: Optional[str]
    start_page: Optional[int]
    confidence: float
    method: str
    warnings: List[str]


def remove_references(
    pages_data: Union[List[Dict[str, Any]], str],
    confidence_threshold: float = 0.60
) -> Tuple[Union[List[Dict[str, Any]], str], ReferenceRemovalDecision]:
    """
    Identifica e remove conservadoramente a seção de referências bibliográficas.

    Aceita tanto uma lista de dicionários por página [{"page": 1, "text": "..."}, ...] quanto uma string direta.
    """
    is_string_input = isinstance(pages_data, str)

    if is_string_input:
        normalized_pages = [{"page": 1, "text": pages_data}]
    else:
        normalized_pages = []
        for idx, item in enumerate(pages_data, start=1):
            if isinstance(item, dict):
                p_num = item.get("page", idx)
                p_txt = item.get("text", "")
                normalized_pages.append({"page": p_num, "text": p_txt})
            else:
                normalized_pages.append({"page": idx, "text": str(item)})

    total_pages = len(normalized_pages)
    if total_pages == 0:
        decision = ReferenceRemovalDecision(
            references_removed=False,
            detected_title=None,
            start_page=None,
            confidence=0.0,
            method="no_pages",
            warnings=["Documento sem páginas para análise."]
        )
        return (pages_data if not is_string_input else ""), decision

    min_page_start = max(1 if total_pages <= 2 else 2, int(total_pages * 0.4))

    best_candidate: Optional[Dict[str, Any]] = None
    highest_score = 0.0

    for page_idx, page_info in enumerate(normalized_pages):
        page_num = page_info["page"]
        if page_num < min_page_start and not is_string_input:
            continue

        lines = page_info["text"].split("\n")
        for line_idx, line in enumerate(lines):
            stripped_line = line.strip().lower()
            clean_heading = re.sub(r'^[#*\s\-\d\.]+', '', stripped_line).strip()

            if clean_heading in REF_HEADER_TITLES or any(clean_heading == title for title in REF_HEADER_TITLES):
                confidence, warnings = _evaluate_reference_section_evidence(
                    normalized_pages, page_idx, line_idx
                )

                if confidence > highest_score:
                    highest_score = confidence
                    best_candidate = {
                        "detected_title": line.strip(),
                        "start_page": page_num,
                        "page_idx": page_idx,
                        "line_idx": line_idx,
                        "confidence": confidence,
                        "warnings": warnings,
                    }

    if best_candidate and best_candidate["confidence"] >= confidence_threshold:
        removed_pages = _truncate_pages_at(
            normalized_pages,
            best_candidate["page_idx"],
            best_candidate["line_idx"]
        )
        decision = ReferenceRemovalDecision(
            references_removed=True,
            detected_title=best_candidate["detected_title"],
            start_page=best_candidate["start_page"],
            confidence=round(best_candidate["confidence"], 2),
            method="multi_factor_heuristic",
            warnings=best_candidate["warnings"]
        )

        if is_string_input:
            clean_str = "\n\n".join(p["text"] for p in removed_pages)
            return clean_str, decision
        return removed_pages, decision

    decision_warnings = ["Seção de referências não detectada com confiança suficiente. Conteúdo integral preservado."]
    if best_candidate:
        decision_warnings.append(
            f"Candidato '{best_candidate['detected_title']}' na página {best_candidate['start_page']} "
            f"obteve confiança insuficiente ({best_candidate['confidence']:.2f} < {confidence_threshold:.2f})."
        )

    decision = ReferenceRemovalDecision(
        references_removed=False,
        detected_title=best_candidate["detected_title"] if best_candidate else None,
        start_page=best_candidate["start_page"] if best_candidate else None,
        confidence=round(highest_score, 2),
        method="conservative_preservation",
        warnings=decision_warnings
    )

    if is_string_input:
        return pages_data, decision
    return normalized_pages, decision


def _evaluate_reference_section_evidence(
    pages_data: List[Dict[str, Any]],
    page_idx: int,
    line_idx: int
) -> Tuple[float, List[str]]:
    score = 0.3
    warnings = []

    total_pages = len(pages_data)
    current_page = pages_data[page_idx]["page"]

    relative_pos = current_page / total_pages if total_pages > 0 else 1.0
    if relative_pos >= 0.7:
        score += 0.25
    elif relative_pos >= 0.5:
        score += 0.15

    remaining_text = _extract_subsequent_text(pages_data, page_idx, line_idx, max_lines=40)

    if not remaining_text.strip():
        warnings.append("Pouco ou nenhum texto subsequente após o título de referências.")
        return score, warnings

    years_found = len(re.findall(r'\b(19\d{2}|20[0-2]\d)\b', remaining_text))
    if years_found >= 5:
        score += 0.25
    elif years_found >= 2:
        score += 0.15

    num_citations = len(re.findall(r'\[\d+\]', remaining_text))
    dois_urls = len(re.findall(r'(10\.\d{4,9}/|https?://|doi:)', remaining_text, re.IGNORECASE))
    if num_citations >= 3 or dois_urls >= 2:
        score += 0.20

    current_line = pages_data[page_idx]["text"].split("\n")[line_idx].strip()
    if len(current_line) > 50:
        score -= 0.30
        warnings.append("Título de referências está inserido em uma frase longa do corpo do texto.")

    return min(1.0, max(0.0, score)), warnings


def _extract_subsequent_text(pages_data: List[Dict[str, Any]], start_page_idx: int, start_line_idx: int, max_lines: int = 40) -> str:
    lines_collected = []
    for p_idx in range(start_page_idx, len(pages_data)):
        lines = pages_data[p_idx]["text"].split("\n")
        l_start = start_line_idx + 1 if p_idx == start_page_idx else 0
        for l in lines[l_start:]:
            if l.strip():
                lines_collected.append(l.strip())
                if len(lines_collected) >= max_lines:
                    break
        if len(lines_collected) >= max_lines:
            break
    return " ".join(lines_collected)


def _truncate_pages_at(pages_data: List[Dict[str, Any]], cut_page_idx: int, cut_line_idx: int) -> List[Dict[str, Any]]:
    truncated: List[Dict[str, Any]] = []

    for p_idx, page_info in enumerate(pages_data):
        if p_idx < cut_page_idx:
            truncated.append(page_info)
        elif p_idx == cut_page_idx:
            lines = page_info["text"].split("\n")
            kept_lines = lines[:cut_line_idx]
            truncated_text = "\n".join(kept_lines).strip()
            if truncated_text:
                truncated.append({
                    "page": page_info["page"],
                    "text": truncated_text
                })
            break

    return truncated
