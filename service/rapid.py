import re
from pathlib import Path

from service.transformation import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    OUTPUT_DIR,
    analyze_text,
)

MAX_RAPID_CHARS = 5_000

RAPID_QUESTIONS = [
    {"field": "tem_conceito",     "label": "Article presents a vulnerability concept",               "type": "Sim/Não",         "options": []},
    {"field": "tipo_definicao",   "label": "Definition type",                                       "type": "Múltipla escolha","options": ["Explicit", "Implicit"]},
    {"field": "disciplina",       "label": "Field or discipline (semicolons if multiple)",           "type": "Texto livre",     "options": []},
    {"field": "definicao",        "label": "Based on this article, what is vulnerability? Synthesize the authors' full understanding: include explicit definitions, implicit conceptualizations, components, and how they frame the concept", "type": "Texto livre","options": []},
    {"field": "tipos",            "label": "Specific vulnerability types mentioned",                 "type": "Lista",           "options": []},
    {"field": "dimensoes",        "label": "Analytical dimensions of vulnerability",                 "type": "Lista",           "options": []},
    {"field": "escala",           "label": "Spatial or social scale of analysis",                   "type": "Lista",           "options": []},
    {"field": "recorte_temporal", "label": "Temporal scope or period of the vulnerability assessment","type": "Texto livre",    "options": []},
    {"field": "hazard",           "label": "Hazard types associated with vulnerability",             "type": "Lista",           "options": []},
    {"field": "framework",        "label": "Theoretical framework or model referenced",              "type": "Texto livre",     "options": []},
    {"field": "componentes",      "label": "Key components, variables or dimensions identified",     "type": "Lista",           "options": []},
    {"field": "populacao",        "label": "Target population or subject of vulnerability",          "type": "Lista",           "options": []},
    {"field": "notas",            "label": "From a taxonomic perspective, what vulnerabilities appear in this study? List and classify every vulnerability type identified, noting how they relate or differ",  "type": "Texto livre",     "options": []},
]

_FIG_CAPTION = re.compile(
    r"^\s*(fig\.|figure|table|tabela|figura|quadro|graph|chart|appendix|apêndice)\s*\d",
    re.IGNORECASE,
)


def _strip_noise(text: str) -> str:
    """Remove table rows, figure/table captions, and equation-heavy lines."""
    cleaned = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            cleaned.append("")
            continue
        if _FIG_CAPTION.match(s):
            continue
        alpha = sum(c.isalpha() for c in s)
        if len(s) > 5 and alpha / len(s) < 0.25:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def analyze_rapid(
    text: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 300,
) -> tuple[dict, dict]:
    text = _strip_noise(text)
    return analyze_text(text, RAPID_QUESTIONS, model, ollama_url, timeout, max_chars=MAX_RAPID_CHARS)


# ── Output helpers ────────────────────────────────────────────────────────────

def _s(value) -> str:
    return str(value) if value is not None else "—"


def _lst(value) -> str:
    if isinstance(value, list) and value:
        return "; ".join(str(v) for v in value)
    return "—"


def save_rapid_result(
    pdf_filename: str,
    answers: dict,
    collection_name: str,
    article_index: int,
    total_articles: int,
) -> Path:
    output_dir = OUTPUT_DIR / collection_name
    output_dir.mkdir(parents=True, exist_ok=True)

    titulo     = answers.get("titulo") or Path(pdf_filename).stem
    autores    = _s(answers.get("autores"))
    ano        = answers.get("ano")
    periodico  = _s(answers.get("periodico"))
    doi        = answers.get("doi")
    tem        = answers.get("tem_conceito")
    tipo_def   = _s(answers.get("tipo_definicao"))
    disciplina = _s(answers.get("disciplina"))

    conceito_display = (
        f"Yes – {tipo_def}" if tem is True and tipo_def != "—"
        else "Yes" if tem is True
        else "No" if tem is False
        else "—"
    )
    doi_display = f"[{doi}](https://doi.org/{doi})" if doi else "—"

    # YAML frontmatter (compatible with Results tab)
    doi_yaml = f'"{doi}"' if doi else "null"
    yaml_lines = [
        "---",
        f'titulo: "{titulo}"',
        f'autores: "{autores}"',
        f"ano: {ano if isinstance(ano, int) else 'null'}",
        f'periodico: "{periodico}"',
        f"doi: {doi_yaml}",
        f"tem_conceito: {'true' if tem is True else 'false' if tem is False else 'null'}",
        f'tipo_definicao: "{tipo_def}"',
        f'disciplina: "{disciplina}"',
        f'arquivo_origem: "{pdf_filename}"',
        "---",
    ]

    body = [
        f"# {titulo}",
        "",
        "## Metadados",
        "",
        "| Campo | Valor |",
        "|-------|-------|",
        f"| **Autores** | {autores} |",
        f"| **Ano** | {ano if ano else '—'} |",
        f"| **Periódico** | {periodico} |",
        f"| **DOI** | {doi_display} |",
        f"| **Conceito de vulnerabilidade** | {conceito_display} |",
        f"| **Tipo de definição** | {tipo_def} |",
        f"| **Disciplina/Campo** | {disciplina} |",
        "",
        "---",
        "",
        "## Definição de Vulnerabilidade",
        "",
        _s(answers.get("definicao")),
        "",
        "---",
        "",
        "## Categorização",
        "",
        "### Tipos de Vulnerabilidade",
        _lst(answers.get("tipos")),
        "",
        "### Dimensão da Vulnerabilidade",
        _lst(answers.get("dimensoes")),
        "",
        "### Escala de Análise",
        _lst(answers.get("escala")),
        "",
        "### Recorte Temporal",
        _s(answers.get("recorte_temporal")),
        "",
        "### Tipo de Hazard",
        _lst(answers.get("hazard")),
        "",
        "---",
        "",
        "## Marco Teórico e Componentes",
        "",
        "### Framework Teórico",
        _s(answers.get("framework")),
        "",
        "### Componentes-Chave",
        _lst(answers.get("componentes")),
        "",
        "### Foco Populacional",
        _lst(answers.get("populacao")),
        "",
        "---",
        "",
        "## Notas Taxonômicas",
        "",
        _s(answers.get("notas")),
        "",
        "---",
        f"*Artigo {article_index} de {total_articles}*",
    ]

    stem = Path(pdf_filename).stem
    output_path = output_dir / f"{stem}.md"
    output_path.write_text(
        "\n".join(yaml_lines) + "\n\n" + "\n".join(body),
        encoding="utf-8",
    )
    return output_path
