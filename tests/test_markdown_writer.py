"""
Testes unitários para o módulo de geração de arquivos Markdown e YAML Frontmatter.
"""

from pathlib import Path
import yaml
from src.sld.models.article import ArticleMetadata
from src.sld.ingestion.markdown_writer import generate_markdown_content, write_markdown_file


def test_generate_markdown_content():
    """Testa se o YAML gerado é válido e se os marcadores de página são inseridos."""
    meta = ArticleMetadata(
        sld_id="SLD-12345678",
        title="Artigo de Exemplo",
        authors=["Silva, J.", "Souza, M."],
        doi="10.1234/example.2023",
        year=2023,
        source_pdf="exemplo.pdf",
        pdf_sha256="abcdef1234567890",
        processed_at="2026-08-10T20:00:00"
    )

    pages_data = [
        {"page": 1, "text": "Texto da primeira página."},
        {"page": 2, "text": "Texto da segunda página."}
    ]

    md_str = generate_markdown_content(meta, pages_data)

    assert md_str.startswith("---")
    assert "sld_id: SLD-12345678" in md_str
    assert "<!-- page: 1 -->" in md_str
    assert "<!-- page: 2 -->" in md_str
    assert "Texto da primeira página." in md_str


def test_write_markdown_file_skip_policy(tmp_path):
    """Testa escrita do arquivo .md e política de sobresscrever/ignorar."""
    meta = ArticleMetadata(
        sld_id="SLD-99999999",
        title="Teste de Arquivo",
        source_pdf="teste_arquivo.pdf"
    )
    pages_data = [{"page": 1, "text": "Conteúdo de teste."}]

    out_file, written = write_markdown_file(meta, pages_data, tmp_path, overwrite_policy="skip")
    assert written is True
    assert out_file.exists()

    # Segunda tentativa com 'skip' não deve sobrescrever
    out_file_2, written_2 = write_markdown_file(meta, pages_data, tmp_path, overwrite_policy="skip")
    assert written_2 is False
    assert out_file_2 == out_file
