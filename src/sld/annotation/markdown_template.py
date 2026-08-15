"""
Gerador e Parser de Arquivos Markdown de Anotação Supervisionada (.md).
Garante rastreabilidade por paragraph_id, verificação de text_hash e extração robusta por número da classe (0-5).
"""

import re
import yaml
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from src.sld.models.classification import ParagraphRecord, AnnotationRecord
from src.sld.models.concept_label import CONCEPT_LABEL_NAMES, validate_and_sanitize_labels
from src.sld.utils.hashing import calculate_text_sha256


SLD_ANNOTATION_FORMAT_VERSION = 1


def generate_annotation_markdown(
    paragraphs: List[ParagraphRecord],
    dataset_id: str = "ANNOTATION_SET_001",
    run_id: str = "",
    concept: str = "conceito investigado",
    annotator_name: str = "",
    hide_scores: bool = True,
    blind_mode: bool = False
) -> str:
    """
    Gera o conteúdo Markdown estruturado para anotação supervisionada externa.
    """
    frontmatter_dict = {
        "sld_annotation_format": SLD_ANNOTATION_FORMAT_VERSION,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "concept": concept,
        "created_at": datetime.now().isoformat(),
        "annotator": annotator_name or ""
    }

    yaml_str = yaml.dump(frontmatter_dict, default_flow_style=False, allow_unicode=True)

    lines = [
        "---",
        yaml_str.strip(),
        "---",
        "",
        "# SLD — Conjunto para Anotação Supervisionada",
        "",
        "## INSTRUÇÕES",
        "",
        "1. Leia cada trecho atentamente.",
        "2. Marque uma ou mais classes entre 1 e 5 quando aplicável.",
        "3. Use a classe 0 somente quando nenhuma das classes 1–5 for pertinente.",
        "4. Um trecho pode receber múltiplas classes (multilabel).",
        "5. Não altere document_id, paragraph_id ou text_hash.",
        "6. Não é obrigatório preencher observações.",
        "7. Salve o arquivo em formato Markdown (.md).",
        "",
        "## LEGENDA DAS CLASSES",
        "",
        "- 0 — Não relevante",
        "- 1 — Definição ou conceituação",
        "- 2 — Fator determinante",
        "- 3 — Tipo ou dimensão",
        "- 4 — Relação causal",
        "- 5 — Característica ou propriedade",
        "",
        "---",
        ""
    ]

    for idx, p in enumerate(paragraphs, start=1):
        lines.append(f"## Trecho {idx}")
        lines.append("")
        if not blind_mode:
            lines.append(f"document_id: {p.article_id}  ")
        lines.append(f"paragraph_id: {p.paragraph_id}  ")
        if not hide_scores and p.semantic_score is not None:
            lines.append(f"similarity_score: {p.semantic_score:.4f}  ")
        hash_val = p.text_sha256 if p.text_sha256 else calculate_text_sha256(p.text)
        lines.append(f"text_hash: {hash_val}  ")
        lines.append("")
        lines.append("### Texto")
        lines.append("")
        lines.append(p.text.strip())
        lines.append("")
        lines.append("### Classificação")
        lines.append("")
        lines.append("- [ ] 0 — Não relevante")
        lines.append("- [ ] 1 — Definição ou conceituação")
        lines.append("- [ ] 2 — Fator determinante")
        lines.append("- [ ] 3 — Tipo ou dimensão")
        lines.append("- [ ] 4 — Relação causal")
        lines.append("- [ ] 5 — Característica ou propriedade")
        lines.append("")
        lines.append("### Observação opcional")
        lines.append("")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def parse_annotation_markdown(markdown_content: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    """
    Realiza o parse tolerante de um arquivo Markdown de anotação supervisionada.

    Retorna:
        Tuple contendo:
        1. Dicionário de metadados extraídos do Frontmatter.
        2. Lista de dicionários por trecho extraído com labels e validações.
        3. Lista de mensagens de erros/alertas encontrados durante o parse.
    """
    metadata: Dict[str, Any] = {}
    items: List[Dict[str, Any]] = []
    warnings: List[str] = []

    # Extrai YAML Frontmatter
    frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', markdown_content, re.DOTALL)
    if frontmatter_match:
        yaml_content = frontmatter_match.group(1)
        try:
            metadata = yaml.safe_load(yaml_content) or {}
        except Exception as e:
            warnings.append(f"Falha ao ler YAML Frontmatter: {e}")
        body_content = markdown_content[frontmatter_match.end():]
    else:
        warnings.append("YAML Frontmatter não encontrado no início do arquivo.")
        body_content = markdown_content

    fmt_ver = metadata.get("sld_annotation_format", 1)
    if fmt_ver != SLD_ANNOTATION_FORMAT_VERSION:
        warnings.append(f"Versão do formato ({fmt_ver}) difere da versão esperada ({SLD_ANNOTATION_FORMAT_VERSION}).")

    # Divide pelos trechos (## Trecho N)
    sections = re.split(r'\n##\s+Trecho\s+\d+', body_content)

    for sec_idx, sec_text in enumerate(sections):
        if not sec_text.strip():
            continue

        p_id_match = re.search(r'paragraph_id:\s*([^\s\n]+)', sec_text)
        if not p_id_match:
            continue

        paragraph_id = p_id_match.group(1).strip()

        doc_id_match = re.search(r'document_id:\s*([^\s\n]+)', sec_text)
        document_id = doc_id_match.group(1).strip() if doc_id_match else ""

        hash_match = re.search(r'text_hash:\s*([^\s\n]+)', sec_text)
        text_hash = hash_match.group(1).strip() if hash_match else ""

        score_match = re.search(r'similarity_score:\s*([\d\.]+)', sec_text)
        sim_score = float(score_match.group(1)) if score_match else None

        # Extrai o Texto do parágrafo
        text_block = ""
        text_match = re.search(r'###\s+Texto\s*\n(.*?)(?=\n###|\Z)', sec_text, re.DOTALL)
        if text_match:
            text_block = text_match.group(1).strip()

        # Extrai a Observação
        note_text = ""
        note_match = re.search(r'###\s+Observação opcional\s*\n(.*?)(?=\n---|\Z)', sec_text, re.DOTALL)
        if note_match:
            note_text = note_match.group(1).strip()

        # Extrai as marcas dos Checkboxes priorizando os NÚMEROS (0, 1, 2, 3, 4, 5)
        checked_classes = []
        # Padrão: - [x] N ou - [X] N ou - [v] N ou - [*] N
        checkbox_matches = re.findall(r'-\s*\[([xXvV\*])\]\s*([0-5])', sec_text)
        for mark, class_num_str in checkbox_matches:
            checked_classes.append(int(class_num_str))

        checked_classes = sorted(list(set(checked_classes)))

        # Validação das regras da Classe 0 e Unannotated
        status = "valid"
        error_msg = None

        if not checked_classes:
            status = "unannotated"
        elif 0 in checked_classes and len(checked_classes) > 1:
            status = "invalid"
            error_msg = "Erro de classificação: 'Não relevante' (Classe 0) não pode coexistir com outras classes."
            warnings.append(f"Parágrafo `{paragraph_id}`: {error_msg}")

        items.append({
            "paragraph_id": paragraph_id,
            "document_id": document_id,
            "text_hash": text_hash,
            "semantic_score": sim_score,
            "text": text_block,
            "checked_classes": checked_classes,
            "status": status,
            "error": error_msg,
            "note": note_text
        })

    return metadata, items, warnings
