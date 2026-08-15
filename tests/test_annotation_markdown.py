"""
Testes unitários automatizados para exportação e importação de anotações em Markdown (.md),
regras da Classe 0, verificação de hash e concordância entre anotadores (Cohen's Kappa).
"""

import pytest
from pathlib import Path
from src.sld.models.classification import ParagraphRecord, AnnotationRecord
from src.sld.annotation.markdown_template import (
    generate_annotation_markdown,
    parse_annotation_markdown,
    SLD_ANNOTATION_FORMAT_VERSION
)
from src.sld.annotation.annotation_service import AnnotationService
from src.sld.evaluation.inter_annotator_agreement import compute_cohen_kappa, compute_inter_annotator_agreement


def test_generate_annotation_markdown():
    """Testa se o gerador de Markdown cria a estrutura YAML, legenda e trechos esperados."""
    paras = [
        ParagraphRecord(
            paragraph_id="DOC001_P000001",
            article_id="DOC001",
            text="Este é um parágrafo conceitual de teste sobre inteligência artificial.",
            text_sha256="abc123hash",
            semantic_score=0.85
        )
    ]

    md_text = generate_annotation_markdown(
        paragraphs=paras,
        dataset_id="SET_TEST_01",
        run_id="run_2026_08_11",
        concept="inteligência artificial",
        annotator_name="",
        hide_scores=False,
        blind_mode=False
    )

    assert "sld_annotation_format: 1" in md_text
    assert "dataset_id: SET_TEST_01" in md_text
    assert "paragraph_id: DOC001_P000001" in md_text
    assert "text_hash: abc123hash" in md_text
    assert "- [ ] 0 — Não relevante" in md_text
    assert "- [ ] 1 — Definição ou conceituação" in md_text
    assert "Dr. Jairo" not in md_text  # Sem nomes pessoais hardcoded


def test_parse_annotation_markdown_checkboxes():
    """Testa parse de caixas [x], [X], [v], [*] e extração pelos NÚMEROS das classes."""
    sample_md = """---
sld_annotation_format: 1
dataset_id: SET_TEST_02
run_id: run_test
concept: resiliência
annotator: ANN_USER
---

## Trecho 1

document_id: DOC001  
paragraph_id: DOC001_P000001  
similarity_score: 0.7500  
text_hash: hash123  

### Texto

Texto sobre fatores determinantes e tipos de resiliência.

### Classificação

- [ ] 0 — Não relevante
- [x] 1 — Definição ou conceituação
- [X] 2 — Fator determinante
- [ ] 3 — Tipo ou dimensão
- [ ] 4 — Relação causal
- [ ] 5 — Característica ou propriedade

### Observação opcional

Nota de teste.
"""

    metadata, items, warnings = parse_annotation_markdown(sample_md)

    assert metadata["dataset_id"] == "SET_TEST_02"
    assert len(items) == 1
    assert items[0]["paragraph_id"] == "DOC001_P000001"
    assert items[0]["checked_classes"] == [1, 2]
    assert items[0]["status"] == "valid"
    assert items[0]["note"] == "Nota de teste."


def test_parse_class_0_exclusivity_rule():
    """Testa a Regra da Classe 0: marcar 0 junto com 1..5 deve produzir erro e status invalid."""
    invalid_md = """---
sld_annotation_format: 1
dataset_id: SET_INVALID
---

## Trecho 1

paragraph_id: DOC001_P000002  

### Texto

Texto do parágrafo.

### Classificação

- [x] 0 — Não relevante
- [x] 2 — Fator determinante
- [ ] 3 — Tipo ou dimensão
- [ ] 4 — Relação causal
- [ ] 5 — Característica ou propriedade

---
"""

    metadata, items, warnings = parse_annotation_markdown(invalid_md)

    assert len(items) == 1
    assert items[0]["status"] == "invalid"
    assert "não pode coexistir" in items[0]["error"]


def test_parse_unchecked_item_is_unannotated():
    """Testa se um trecho sem caixas marcadas é classificado como unannotated e NÃO como Classe 0."""
    unannotated_md = """---
sld_annotation_format: 1
---

## Trecho 1

paragraph_id: DOC001_P000003  

### Texto

Texto sem caixas marcadas.

### Classificação

- [ ] 0 — Não relevante
- [ ] 1 — Definição ou conceituação
- [ ] 2 — Fator determinante
- [ ] 3 — Tipo ou dimensão
- [ ] 4 — Relação causal
- [ ] 5 — Característica ou propriedade

---
"""

    metadata, items, warnings = parse_annotation_markdown(unannotated_md)

    assert len(items) == 1
    assert items[0]["status"] == "unannotated"
    assert items[0]["checked_classes"] == []


def test_inter_annotator_cohen_kappa():
    """Testa cálculo de Cohen's Kappa para dois anotadores em dados conhecidos."""
    y1 = [1, 1, 0, 0, 1]
    y2 = [1, 1, 0, 0, 0]

    stats = compute_cohen_kappa(y1, y2)

    assert stats["n_samples"] == 5
    assert stats["p_o"] == 0.8  # 4/5 concordâncias
    assert "kappa" in stats


def test_annotation_service_multi_annotators(tmp_path: Path):
    """Testa registro de dois anotadores para o mesmo parágrafo sem sobrescrita silenciosa."""
    service = AnnotationService(tmp_path)

    a1 = AnnotationRecord(
        annotation_id="ANN1_P1",
        paragraph_id="P1",
        annotator_id="ANN_001",
        label_1=True,
        annotation_status="valid"
    )
    a2 = AnnotationRecord(
        annotation_id="ANN2_P1",
        paragraph_id="P1",
        annotator_id="ANN_002",
        label_2=True,
        annotation_status="valid"
    )

    service.save_annotations([a1, a2])
    loaded = service.load_annotations()

    assert len(loaded) == 2
    annotator_ids = {r.annotator_id for r in loaded}
    assert "ANN_001" in annotator_ids
    assert "ANN_002" in annotator_ids

    agreement_report = compute_inter_annotator_agreement(loaded)
    assert agreement_report["has_paired_annotations"] is True
    assert agreement_report["n_paired_paragraphs"] == 1
