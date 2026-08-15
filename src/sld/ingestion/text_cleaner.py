"""
Limpeza de texto, recomposição de hifenização e remoção de cabeçalhos/rodapés repetitivos.
"""

import re
import unicodedata
from typing import List, Dict, Any


def normalize_characters_and_spaces(text: str) -> str:
    """
    Normaliza caracteres unicode mantendo diacríticos/acentos e substitui
    múltiplos espaços horizontais por um único espaço. Preserva quebras de linha significativas.
    """
    if not text:
        return ""

    # Normalização Unicode NFC (combina acentos)
    text = unicodedata.normalize("NFC", text)

    # Substitui caracteres de controle não imprimíveis (exceto \n e \t)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Substitui múltiplos espaços na mesma linha por 1 espaço
    lines = text.split("\n")
    cleaned_lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]

    return "\n".join(cleaned_lines)


def recompose_hyphenation(text: str) -> str:
    """
    Recompõe palavras divididas por hifenização na quebra de linha.
    Exemplo: "para- \n digma" -> "paradigma"
    """
    if not text:
        return ""

    # Padrão: palavra com hífen no final da linha seguida de quebra de linha e continuação em minúscula
    pattern = re.compile(r'(\b[A-Za-zÀ-ÿ]{2,})-\s*\n\s*([a-zà-ÿ]{2,}\b)')
    recomposed = pattern.sub(r'\1\2', text)

    return recomposed


def clean_pages(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aplica limpeza completa em uma lista de páginas:
    1. Normalização de caracteres e espaços
    2. Recomposição de hifenização
    3. Identificação e redução de cabeçalhos e rodapés repetidos em múltiplas páginas
    """
    if not pages_data:
        return []

    # 1. Limpeza por página
    cleaned_pages: List[Dict[str, Any]] = []
    for page_info in pages_data:
        raw_text = page_info.get("text", "")
        normalized = normalize_characters_and_spaces(raw_text)
        recomposed = recompose_hyphenation(normalized)
        cleaned_pages.append({
            "page": page_info["page"],
            "text": recomposed
        })

    # 2. Identificação de cabeçalhos/rodapés repetitivos se houver mais de 2 páginas
    if len(cleaned_pages) > 2:
        headers, footers = _detect_repeated_headers_footers(cleaned_pages)
        for page in cleaned_pages:
            lines = page["text"].split("\n")
            filtered_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped in headers or stripped in footers:
                    continue
                filtered_lines.append(line)
            page["text"] = "\n".join(filtered_lines)

    return cleaned_pages


def _detect_repeated_headers_footers(pages: List[Dict[str, Any]]) -> Tuple[set, set]:
    """Detecta frases de topo (primeira linha) ou rodapé (última linha) que se repetem em > 40% das páginas."""
    first_lines: Dict[str, int] = {}
    last_lines: Dict[str, int] = {}
    total_pages = len(pages)

    for page in pages:
        lines = [l.strip() for l in page["text"].split("\n") if l.strip()]
        if not lines:
            continue

        top_line = lines[0]
        bottom_line = lines[-1]

        # Ignora números simples de página sozinhos
        if not top_line.isdigit() and len(top_line) > 5:
            first_lines[top_line] = first_lines.get(top_line, 0) + 1

        if not bottom_line.isdigit() and len(bottom_line) > 5:
            last_lines[bottom_line] = last_lines.get(bottom_line, 0) + 1

    threshold = max(2, int(total_pages * 0.4))
    repeated_headers = {line for line, count in first_lines.items() if count >= threshold}
    repeated_footers = {line for line, count in last_lines.items() if count >= threshold}

    return repeated_headers, repeated_footers
