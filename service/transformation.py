import json
import re
from pathlib import Path

import requests

OUTPUT_DIR = Path("data/collections")
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
# Modelos recomendados por qualidade: llama3.1:8b > llama3.2:3b > mistral > gemma2:2b
MAX_CONTENT_CHARS = 16_000  # local models lose coherence with very long contexts


def list_ollama_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def _system_prompt() -> str:
    return (
        "You are a precise information extractor for scientific articles. "
        "Always respond with a valid JSON object only — no text before, no text after, no markdown. "
        "All string values in the JSON must be written in English, regardless of the article's language."
    )


def _user_prompt(questions: list[dict], content: str) -> str:
    field_lines = [
        '"titulo": título completo do artigo (string)',
        '"autores": autores separados por ponto e vírgula (string)',
        '"ano": ano de publicação (número inteiro)',
        '"periodico": nome do periódico ou conferência (string)',
        '"doi": DOI se disponível (string ou null)',
    ]
    for q in questions:
        match q["type"]:
            case "Sim/Não":
                fmt = "true ou false"
            case "Lista":
                fmt = 'array de strings, ex: ["a", "b"]'
            case "Múltipla escolha":
                opts = ", ".join(f'"{o}"' for o in q.get("options", []))
                fmt = f"exatamente uma das opções: {opts}"
            case "Numérico":
                fmt = "número"
            case _:
                fmt = "string"
        field_lines.append(f'"{q["field"]}": {q["label"]} ({fmt})')

    truncated = content[:MAX_CONTENT_CHARS]

    return (
        "Extraia as informações abaixo do artigo científico e retorne um único objeto JSON.\n"
        "Use null para campos não encontrados no texto.\n\n"
        "CAMPOS ESPERADOS:\n"
        + "\n".join(f"- {l}" for l in field_lines)
        + "\n\nARTIGO:\n"
        + truncated
    )


def _parse_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def analyze_text(text: str, questions: list[dict], model: str = DEFAULT_MODEL) -> dict:
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(questions, text)},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_ctx": 8192},
        },
        timeout=300,
    )
    response.raise_for_status()
    return _parse_json(response.json()["message"]["content"])


def _yaml_line(field: str, value, qtype: str) -> str:
    if value is None:
        return f"{field}: null"
    if qtype == "Sim/Não":
        return f"{field}: {'true' if value else 'false'}"
    if qtype == "Lista":
        if not isinstance(value, list) or not value:
            return f"{field}: []"
        items = "\n".join(f"  - {v}" for v in value)
        return f"{field}:\n{items}"
    if qtype == "Numérico":
        return f"{field}: {value}"
    safe = str(value).replace('"', '\\"')
    return f'{field}: "{safe}"'


def save_result(pdf_filename: str, questions: list[dict], answers: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ano = answers.get("ano")
    doi = answers.get("doi")
    doi_yaml = f'"{doi}"' if doi else "null"

    yaml_lines = [
        "---",
        f'titulo: "{answers.get("titulo", "")}"',
        f'autores: "{answers.get("autores", "")}"',
        f"ano: {ano if isinstance(ano, int) else 'null'}",
        f'periodico: "{answers.get("periodico", "")}"',
        f"doi: {doi_yaml}",
    ]
    for q in questions:
        yaml_lines.append(_yaml_line(q["field"], answers.get(q["field"]), q["type"]))
    yaml_lines += [f'arquivo_origem: "{pdf_filename}"', "---"]

    body = [
        f"# {answers.get('titulo', Path(pdf_filename).stem)}",
        "",
        f"**Autores:** {answers.get('autores', '—')}  ",
        f"**Ano:** {ano if ano else '—'}  ",
        f"**Periódico:** {answers.get('periodico', '—')}  ",
    ]
    if doi:
        body.append(f"**DOI:** {doi}  ")
    body += ["", "---", "", "## Parâmetros identificados", ""]

    for q in questions:
        value = answers.get(q["field"])
        if isinstance(value, list):
            display = ", ".join(str(v) for v in value) if value else "—"
        elif value is None:
            display = "—"
        elif q["type"] == "Sim/Não":
            display = "Sim" if value else "Não"
        else:
            display = str(value)
        body.append(f"**{q['label']}:** {display}  ")

    stem = Path(pdf_filename).stem
    output_path = OUTPUT_DIR / f"{stem}.md"
    output_path.write_text(
        "\n".join(yaml_lines) + "\n\n" + "\n".join(body),
        encoding="utf-8",
    )
    return output_path
