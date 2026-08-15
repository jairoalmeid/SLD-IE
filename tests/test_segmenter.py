"""
Testes unitários para o módulo de segmentação de Markdown com proveniência persistente.
"""

from pathlib import Path
from src.sld.semantic.segmenter import segment_markdown, parse_markdown_with_yaml


def test_parse_markdown_with_yaml():
    """Testa separação de YAML Frontmatter e corpo principal do Markdown."""
    raw_md = """---
sld_id: "SLD-TEST1234"
title: "Artigo Sintético de Teste"
authors:
  - "Autor A"
---

# Introdução
Este é o primeiro parágrafo de introdução do artigo científico.
"""

    frontmatter, body = parse_markdown_with_yaml(raw_md)

    assert frontmatter["sld_id"] == "SLD-TEST1234"
    assert frontmatter["title"] == "Artigo Sintético de Teste"
    assert "Introdução" in body


def test_segment_markdown_paragraphs_and_pages():
    """Testa segmentação de parágrafos, identificadores persistentes P0001 e hashes."""
    raw_md = """---
sld_id: "SLD-ABC12345"
title: "Pesquisa em Inteligência Artificial"
source_pdf: "artigo_teste.pdf"
---

<!-- page: 1 -->
# Introdução
Este é o parágrafo inicial do artigo sobre inteligência artificial e aprendizado de máquina aplicada à biologia.

<!-- page: 2 -->
# Métodos
Utilizamos modelos de linguagem e embeddings multilíngues para busca semântica em bases de conhecimento científico.
"""
    md_path = Path("/tmp/artigo_teste.md")
    segments = segment_markdown(raw_md, md_path, min_words=5, min_characters=30, max_characters=500)

    assert len(segments) == 2
    assert segments[0].article_id == "SLD-ABC12345"
    assert segments[0].paragraph_id == "P0001"
    assert segments[0].page_start == 1
    assert segments[0].section == "Introdução"
    assert segments[0].status == "valid_paragraph"
    assert len(segments[0].text_sha256) == 64

    assert segments[1].paragraph_id == "P0002"
    assert segments[1].page_start == 2
    assert segments[1].section == "Métodos"
