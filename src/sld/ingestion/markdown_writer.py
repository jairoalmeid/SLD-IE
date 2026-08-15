"""
Escritor de arquivos Markdown com cabeçalho YAML e marcadores de página.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import yaml
from src.sld.models.article import ArticleMetadata
from src.sld.utils.files import sanitize_filename, ensure_directory


def generate_markdown_content(
    metadata: ArticleMetadata,
    pages_data: List[Dict[str, Any]]
) -> str:
    """
    Gera a string Markdown completa contendo o cabeçalho YAML Frontmatter
    e o texto estruturado com marcadores <!-- page: N -->.
    """
    # Prepara o dicionário de metadados para YAML
    yaml_dict = metadata.to_yaml_dict()

    # Formata o YAML Frontmatter
    yaml_str = yaml.dump(
        yaml_dict,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False
    )

    md_lines = ["---", yaml_str.strip(), "---", ""]

    # Adiciona o conteúdo página por página com marcador legível
    for page_info in pages_data:
        page_num = page_info["page"]
        text = page_info["text"].strip()
        if text:
            md_lines.append(f"<!-- page: {page_num} -->")
            md_lines.append(text)
            md_lines.append("")  # Linha em branco para separação visual

    return "\n".join(md_lines)


def write_markdown_file(
    metadata: ArticleMetadata,
    pages_data: List[Dict[str, Any]],
    output_dir: Path,
    overwrite_policy: str = "skip"
) -> Tuple[Path, bool]:
    """
    Salva o conteúdo em um arquivo `.md` no diretório de saída especificado.

    Retorna:
        Tuple[Path, bool]: (caminho_final_do_arquivo, foi_escrito_ou_sobrescrito)
    """
    ensure_directory(output_dir)

    base_name = sanitize_filename(Path(metadata.source_pdf).stem)
    target_path = output_dir / f"{base_name}.md"

    # Tratamento da política de sobrescrita
    if target_path.exists():
        if overwrite_policy == "skip":
            return target_path, False
        elif overwrite_policy == "timestamp":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_path = output_dir / f"{base_name}_{timestamp}.md"

    # Gera conteúdo e salva em UTF-8
    content = generate_markdown_content(metadata, pages_data)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)

    return target_path, True
