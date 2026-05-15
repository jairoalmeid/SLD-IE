import json
import re
from pathlib import Path

import requests
from json_repair import repair_json

OUTPUT_DIR = Path("data/collections")
DEFAULT_OLLAMA_URL = "http://192.168.0.13:11434"
DEFAULT_MODEL = "gemma4:e4b"
MAX_CONTENT_CHARS = 6_000

_REF_HEADING = re.compile(
    r"(?m)^[ \t]*(?:#{1,4}[ \t]*)?(?:\d+[\.\)][ \t]*)?"
    r"(?:References|Bibliography|Referências|Bibliografía|Literatura(?:[ \t]+Citada)?|Works[ \t]+Cited)"
    r"[ \t]*$",
    re.IGNORECASE,
)


def list_ollama_models(url: str = DEFAULT_OLLAMA_URL) -> list[str]:
    try:
        r = requests.get(f"{url}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def test_model(url: str, model: str) -> str:
    r = requests.post(
        f"{url}/api/generate",
        json={"model": model, "prompt": "Reply with: ok", "stream": False, "options": {"num_predict": 4}},
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def _clean_text(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_references(text: str) -> str:
    threshold = int(len(text) * 0.6)
    m = _REF_HEADING.search(text, threshold)
    if m:
        return text[:m.start()].rstrip()
    return text


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    cut = max(window.rfind(". "), window.rfind(".\n"))
    cut = cut + 1 if cut != -1 else max_chars
    return text[:cut].rstrip() + "\n[truncated]"


def _type_placeholder(q: dict) -> str:
    match q["type"]:
        case "Sim/Não":      return "true"
        case "Lista":        return '["..."]'
        case "Numérico":     return "0"
        case "Múltipla escolha":
            opts = q.get("options", [])
            return f'"{opts[0]}"' if opts else '"..."'
        case _:              return '"..."'


# ── Prompt 1: compact schema, used with format:json ──────────────────────────

def _field_line(q: dict) -> str:
    match q["type"]:
        case "Sim/Não":
            return f'  "{q["field"]}": {q["label"]} (true/false)'
        case "Lista":
            return f'  "{q["field"]}": {q["label"]} (array of strings)'
        case "Múltipla escolha":
            opts = " | ".join(f'"{o}"' for o in q.get("options", []))
            return f'  "{q["field"]}": {q["label"]} ({opts})'
        case "Numérico":
            return f'  "{q["field"]}": {q["label"]} (number)'
        case _:
            return f'  "{q["field"]}": {q["label"]}'


def _build_schema_prompt(questions: list[dict], content: str) -> str:
    lines = [
        '  "titulo": full title',
        '  "autores": authors semicolon-separated',
        '  "ano": year as integer',
        '  "periodico": journal or conference',
        '  "doi": DOI string or null',
    ] + [_field_line(q) for q in questions]
    return (
        "Return only a valid JSON object. Use null for any missing field.\n\n"
        "Fields:\n" + "\n".join(lines) + "\n\nARTICLE:\n" + content
    )


# ── Prompt 2: JSON template, used without format:json (retry) ────────────────

def _build_template_prompt(questions: list[dict], content: str) -> str:
    fixed = {
        "titulo": "",
        "autores": "",
        "ano": 0,
        "periodico": "",
        "doi": None,
    }
    custom = {q["field"]: json.loads(_type_placeholder(q))
              if q["type"] in ("Sim/Não", "Numérico")
              else ([] if q["type"] == "Lista" else "")
              for q in questions}
    template = json.dumps({**fixed, **custom}, ensure_ascii=False, indent=2)
    return (
        "Fill in the JSON template below using ONLY information from the article. "
        "Output ONLY the completed JSON object, nothing else. "
        "Use null for any field not found.\n\n"
        + template
        + "\n\nARTICLE:\n" + content
    )


# ── JSON parsing ─────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    text = text.strip()
    if not text:
        return {}
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = repair_json(text)
        try:
            return json.loads(repaired) if repaired else {}
        except json.JSONDecodeError:
            return {}


# ── Normalization ─────────────────────────────────────────────────────────────

_BOOL_TRUE  = {"sim", "yes", "true", "1", "s", "y"}
_BOOL_FALSE = {"não", "nao", "no", "false", "0", "n"}
_NULL_VALS  = {"-", "—", "", "n/a", "na", "null", "none"}


def _normalize(answers: dict, questions: list[dict]) -> dict:
    for q in questions:
        field = q["field"]
        if field not in answers:
            continue
        val = answers[field]
        if q["type"] == "Sim/Não":
            if isinstance(val, bool):
                continue
            s = str(val).strip().lower()
            if s in _BOOL_TRUE:
                answers[field] = True
            elif s in _BOOL_FALSE:
                answers[field] = False
            else:
                answers[field] = None
        elif val is not None and isinstance(val, str) and val.strip().lower() in _NULL_VALS:
            answers[field] = None
    return answers


def _is_empty(answers: dict) -> bool:
    """True if the model returned nothing useful (all key fields missing or blank)."""
    for key in ("titulo", "autores", "periodico"):
        if answers.get(key) and str(answers[key]).strip():
            return False
    return True


# ── Ollama call ───────────────────────────────────────────────────────────────

def _ollama_chat(prompt: str, model: str, ollama_url: str, timeout: int, json_format: bool) -> tuple[str, dict]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 768},
    }
    if json_format:
        payload["format"] = "json"
    r = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"], data


def analyze_text(
    text: str,
    questions: list[dict],
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 300,
    max_chars: int = MAX_CONTENT_CHARS,
) -> tuple[dict, dict]:
    text = _clean_text(text)
    text = _strip_references(text)
    text = _truncate_at_sentence(text, max_chars)

    # Attempt 1: compact schema + format:json
    content, data = _ollama_chat(
        _build_schema_prompt(questions, text), model, ollama_url, timeout, json_format=True
    )
    answers = _parse_json(content)

    # Attempt 2: JSON template without format:json
    if _is_empty(answers):
        content, data = _ollama_chat(
            _build_template_prompt(questions, text), model, ollama_url, timeout, json_format=False
        )
        answers = _parse_json(content)

    if _is_empty(answers):
        raise ValueError("O modelo não retornou dados válidos. Tente um modelo maior ou verifique o PDF.")

    tokens = {
        "prompt": data.get("prompt_eval_count", 0),
        "response": data.get("eval_count", 0),
    }
    return _normalize(answers, questions), tokens


# ── Output ────────────────────────────────────────────────────────────────────

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


def save_result(pdf_filename: str, questions: list[dict], answers: dict, collection_name: str) -> Path:
    output_dir = OUTPUT_DIR / collection_name
    output_dir.mkdir(parents=True, exist_ok=True)

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
        f"**Authors:** {answers.get('autores', '—')}  ",
        f"**Year:** {ano if ano else '—'}  ",
        f"**Journal:** {answers.get('periodico', '—')}  ",
    ]
    if doi:
        body.append(f"**DOI:** {doi}  ")
    body += ["", "---", "", "## Identified Parameters", ""]

    for q in questions:
        value = answers.get(q["field"])
        if isinstance(value, list):
            display = ", ".join(str(v) for v in value) if value else "—"
        elif value is None:
            display = "—"
        elif q["type"] == "Sim/Não":
            display = "Yes" if value else "No"
        else:
            display = str(value)
        body.append(f"**{q['label']}:** {display}  ")

    stem = Path(pdf_filename).stem
    output_path = output_dir / f"{stem}.md"
    output_path.write_text(
        "\n".join(yaml_lines) + "\n\n" + "\n".join(body),
        encoding="utf-8",
    )
    return output_path
