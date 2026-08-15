"""
Verificador de Integridade Diagnóstica para Projetos de Análise no SLD.
Realiza verificações não destrutivas detectando inconsistências entre registros, arquivos Markdown e índices.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from pydantic import BaseModel, Field
from src.sld.utils.atomic import validate_vector_index_files


class IntegrityReport(BaseModel):
    """Relatório estruturado de diagnóstico de integridade da análise."""
    project_dir: str
    checked_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: str = "valid"  # "valid", "warning", "inconsistent", "critical"

    integros: List[str] = Field(default_factory=list)
    avisos: List[str] = Field(default_factory=list)
    inconsistencias_recuperaveis: List[str] = Field(default_factory=list)
    erros_reconstrucao: List[str] = Field(default_factory=list)

    total_articles_registered: int = 0
    total_markdowns_found: int = 0
    total_segments_indexed: int = 0
    total_vectors_indexed: int = 0


from datetime import datetime


class AnalysisIntegrityChecker:
    """Executa checagens completas não destrutivas na pasta do projeto."""

    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.markdown_dir = self.project_dir / "markdown"
        self.index_dir = self.project_dir / "index"
        self.manifests_dir = self.project_dir / "manifests"

    def run_full_check(self, expected_model: Optional[str] = None) -> IntegrityReport:
        """Executa todas as verificações e retorna o relatório categorizado."""
        report = IntegrityReport(project_dir=str(self.project_dir))

        # 1. Verifica existência de arquivos base
        analysis_json_path = self.project_dir / "analysis.json"
        registry_path = self.manifests_dir / "article_registry.json"

        if not analysis_json_path.exists():
            report.erros_reconstrucao.append("Arquivo 'analysis.json' ausente na raiz do projeto.")

        if not registry_path.exists():
            report.avisos.append("Registro consolidado 'article_registry.json' ausente em manifests/.")
            registered_articles: Dict[str, Any] = {}
        else:
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    registered_articles = json.load(f)
                report.total_articles_registered = len(registered_articles)
                report.integros.append(f"{len(registered_articles)} artigos registrados em 'article_registry.json'.")
            except Exception as e:
                report.erros_reconstrucao.append(f"Registro 'article_registry.json' corrompido: {e}")
                registered_articles = {}

        # 2. Verifica arquivos Markdown
        md_files = list(self.markdown_dir.glob("*.md")) if self.markdown_dir.exists() else []
        report.total_markdowns_found = len(md_files)
        md_stems = {f.stem for f in md_files}

        if md_files:
            report.integros.append(f"{len(md_files)} arquivos Markdown encontrados em markdown/.")
        else:
            report.avisos.append("Nenhum arquivo Markdown encontrado em markdown/.")

        # Cruza registros x Markdowns
        for art_id, reg_data in registered_articles.items():
            if reg_data.get("processing_status") == "completed":
                md_path_str = reg_data.get("markdown_path", "")
                if md_path_str and not Path(md_path_str).exists():
                    report.inconsistencias_recuperaveis.append(
                        f"Artigo '{art_id}' marcado como concluído, mas o arquivo Markdown não foi localizado em disk."
                    )

        # 3. Verifica integridade do índice vetorial
        embeddings_path = self.index_dir / "embeddings.npy"
        segments_path = self.index_dir / "segments.jsonl"
        metadata_path = self.index_dir / "index_metadata.json"

        if embeddings_path.exists() or segments_path.exists() or metadata_path.exists():
            is_valid, index_errs = validate_vector_index_files(
                embeddings_path, segments_path, metadata_path, expected_model=expected_model
            )
            if is_valid:
                try:
                    meta = json.load(open(metadata_path, "r", encoding="utf-8"))
                    vecs = np.load(embeddings_path)
                    report.total_vectors_indexed = vecs.shape[0] if vecs.ndim >= 1 else 0
                    report.total_segments_indexed = meta.get("total_segments", 0)
                    report.integros.append(
                        f"Índice vetorial válido com {report.total_vectors_indexed:,} vetores ({meta.get('embedding_model')}).".replace(",", ".")
                    )
                except Exception:
                    pass
            else:
                for err in index_errs:
                    if "Divergência" in err or "incompatível" in err:
                        report.erros_reconstrucao.append(err)
                    else:
                        report.inconsistencias_recuperaveis.append(err)

        # 4. Checa por operações ativas sem finalização
        checkpoints_path = self.manifests_dir / "checkpoints.json"
        if checkpoints_path.exists():
            try:
                with open(checkpoints_path, "r", encoding="utf-8") as f:
                    chk_data = json.load(f)
                active_id = chk_data.get("active_operation_id")
                if active_id:
                    report.avisos.append(f"Operação ativa ou interrompida registrada em checkpoints: '{active_id}'.")
            except Exception:
                pass

        # Define status global
        if report.erros_reconstrucao:
            report.status = "critical"
        elif report.inconsistencias_recuperaveis:
            report.status = "inconsistent"
        elif report.avisos:
            report.status = "warning"
        else:
            report.status = "valid"

        return report
