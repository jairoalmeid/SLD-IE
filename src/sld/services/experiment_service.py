"""
Serviço de gerenciamento de experimentos, persistência isolada e gerador de relatórios METHODS.md.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

from config.settings import DEFAULT_OUTPUT_DIR
from src.sld.models.search_result import SearchResult, Segment
from src.sld.models.experiment import ExperimentConfig, ExperimentManifest, EnvironmentMetadata
from src.sld.models.evaluation import EvaluationSummary
from src.sld.semantic.semantic_reference import SemanticReferenceSet
from src.sld.semantic.embedding_service import EmbeddingService
from src.sld.utils.files import ensure_directory


class ExperimentService:
    """Gerencia a persistência reprodutiva de cada execução experimental sob output/experiments/<run_id>/."""

    def __init__(self, base_output_dir: Optional[Path] = None):
        self.base_output_dir = base_output_dir or DEFAULT_OUTPUT_DIR
        self.experiments_dir = self.base_output_dir / "experiments"
        ensure_directory(self.experiments_dir)

    def create_run_id(self) -> str:
        """Gera um identificador único de execução experimental (e.g. exp_20260810_214500_a1b2)."""
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:4]
        return f"exp_{now_str}_{short_id}"

    def save_experiment_run(
        self,
        config: ExperimentConfig,
        embedding_service: EmbeddingService,
        reference_set: SemanticReferenceSet,
        results: List[SearchResult],
        segments: List[Segment],
        evaluation_summary: Optional[EvaluationSummary] = None,
        performance_times: Optional[Dict[str, float]] = None
    ) -> Path:
        """
        Salva todos os arquivos do experimento sob output/experiments/<run_id>/ de forma auditável e imutável.
        """
        run_id = config.run_id
        exp_dir = self.experiments_dir / run_id
        ensure_directory(exp_dir)

        env_meta = embedding_service.get_environment_metadata()
        perf = performance_times or {}

        # Contagens de auditoria
        total_articles = len(set(s.article_id for s in segments))
        total_paragraphs = len(segments)
        valid_paragraphs = sum(1 for s in segments if s.status == "valid_paragraph")
        excluded_paragraphs = sum(1 for s in segments if s.status != "valid_paragraph")
        selected_paragraphs = sum(1 for r in results if r.selected)

        exclusion_reasons = {}
        for s in segments:
            if s.status != "valid_paragraph":
                reason = s.exclusion_reason or s.status
                exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1

        manifest = ExperimentManifest(
            run_id=run_id,
            created_at=datetime.now().isoformat(),
            config=config,
            environment=env_meta,
            semantic_references=reference_set.to_list_dicts(),
            total_articles=total_articles,
            total_paragraphs=total_paragraphs,
            valid_paragraphs=valid_paragraphs,
            excluded_paragraphs=excluded_paragraphs,
            selected_paragraphs=selected_paragraphs,
            exclusion_reasons=exclusion_reasons,
            processing_time_seconds=perf.get("processing_time", 0.0),
            embedding_time_seconds=perf.get("embedding_time", 0.0),
            similarity_time_seconds=perf.get("similarity_time", 0.0),
            paragraphs_per_second=perf.get("paragraphs_per_second", 0.0),
        )

        # 1. Salva config.json
        with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)

        # 2. Salva environment.json
        with open(exp_dir / "environment.json", "w", encoding="utf-8") as f:
            json.dump(env_meta.to_dict(), f, indent=2, ensure_ascii=False)

        # 3. Salva semantic_references.json
        with open(exp_dir / "semantic_references.json", "w", encoding="utf-8") as f:
            json.dump(reference_set.to_list_dicts(), f, indent=2, ensure_ascii=False)

        # 4. Salva manifest.json
        with open(exp_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)

        # 5. Salva metrics.json se houver calibração
        if evaluation_summary:
            with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(evaluation_summary.to_dict(), f, indent=2, ensure_ascii=False)

        # 6. Salva resultados em CSV e Markdown
        self._save_results_csv_and_md(exp_dir, results, config)

        # 7. Gera e salva o relatório metodológico METHODS.md
        methods_content = generate_methods_markdown_report(
            config=config,
            env_meta=env_meta,
            reference_set=reference_set,
            manifest=manifest,
            evaluation_summary=evaluation_summary
        )
        with open(exp_dir / "METHODS.md", "w", encoding="utf-8") as f:
            f.write(methods_content)

        return exp_dir

    def _save_results_csv_and_md(
        self,
        exp_dir: Path,
        results: List[SearchResult],
        config: ExperimentConfig
    ):
        """Salva resultados em CSV e Markdown estruturado com YAML Frontmatter."""
        csv_path = exp_dir / "results.csv"
        md_path = exp_dir / "results.md"

        # CSV
        df = pd.DataFrame([r.to_dict() for r in results])
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # Markdown Estruturado
        md_lines = [
            f"# Experimento de Recuperação Semântica — {config.run_id}",
            f"*Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            f"*Modelo:* `{config.embedding_model}` | *Estratégia:* `{config.aggregation_strategy}` | *Threshold:* `{config.threshold}`",
            "",
            "---",
            ""
        ]

        for r in results:
            if not r.selected and config.threshold > 0:
                continue

            md_lines.extend([
                "---",
                "---",
                f"run_id: \"{r.run_id}\"",
                f"article_id: \"{r.article_id}\"",
                f"paragraph_id: \"{r.paragraph_id}\"",
                f"chunk_id: \"{r.chunk_id or ''}\"",
                f"paragraph_hash: \"{r.paragraph_hash}\"",
                f"aggregate_score: {r.aggregate_score:.4f}",
                f"best_anchor: \"{r.best_anchor_id}\"",
                f"threshold_used: {r.threshold_used}",
                "---",
                "",
                f"### #{r.rank} — {r.title} (Score: {r.aggregate_score:.4f})",
                f"- **Seção:** `{r.section}` (Pág: `{r.page_range}`)",
                f"- **Melhor Âncora ({r.best_anchor_id}):** *\"{r.best_anchor_text}\"*",
                "",
                "#### Scores Detalhados por Âncora:",
            ])

            for q_id, q_score in r.anchor_scores.items():
                md_lines.append(f"- **{q_id}:** `{q_score:.4f}`")

            md_lines.extend([
                "",
                "#### Texto Original do Parágrafo:",
                f"> {r.text}",
                ""
            ])

            if r.context_before:
                md_lines.append(f"<details><summary>Contexto Anterior</summary>\n\n{r.context_before}\n</details>\n")
            if r.context_after:
                md_lines.append(f"<details><summary>Contexto Posterior</summary>\n\n{r.context_after}\n</details>\n")

            md_lines.append("")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))


def generate_methods_markdown_report(
    config: ExperimentConfig,
    env_meta: EnvironmentMetadata,
    reference_set: SemanticReferenceSet,
    manifest: ExperimentManifest,
    evaluation_summary: Optional[EvaluationSummary] = None
) -> str:
    """
    Gera deterministicamente a seção de metodologia (METHODS.md) para inclusão direta em tese de doutorado.
    """
    lines = [
        f"# Methodology Report: Semantic Retrieval of Disaster Vulnerability",
        f"**Run ID:** `{config.run_id}`  ",
        f"**Date:** `{manifest.created_at}`  ",
        "",
        "## 1. Scientific Context & Objectives",
        "This module performs high-sensitivity semantic retrieval as a preliminary screening step for scientific literature concerning vulnerability in disaster risk contexts. The goal is to maximize recall over candidate paragraphs extracted from scientific articles, ensuring relevant multidimensional content (social, economic, physical, institutional, coping capacity) is retrieved for subsequent qualitative and quantitative validation.",
        "",
        "## 2. Text Segmentation & Provenance Tracking",
        f"- **Corpus:** {manifest.total_articles} scientific articles processed into {manifest.total_paragraphs} total paragraphs.",
        f"- **Valid Paragraphs Analyzed:** {manifest.valid_paragraphs}",
        f"- **Excluded Short Fragments:** {manifest.excluded_paragraphs} (classified non-destructively under rules: minimum words = {config.min_words}, minimum characters = {config.min_characters}).",
        f"- **Long Paragraph Strategy:** `{config.long_text_strategy}` (maximum characters per unit = {config.max_characters}, chunk overlap = {config.chunk_overlap} chars).",
        "- **Provenance:** Each paragraph retains unique persistent identifiers (`article_id`, `section_id`, `paragraph_id`, `paragraph_hash = SHA256`).",
        "",
        "## 3. Sentence Embeddings & Representation Model",
        f"- **Model Name:** `{config.embedding_model}` (Revision: `{config.model_revision}`).",
        f"- **Vector Dimension:** 768 dimensions (normalized $L_2$).",
        f"- **Similarity Metric:** Cosine Similarity ($\text{{Cosine}}(E(P_i), E(Q_j)) = E(P_i) \\cdot E(Q_j)$).",
        f"- **Execution Hardware:** `{env_meta.device.upper()}` (Python {env_meta.python_version}, PyTorch {env_meta.torch_version}, Transformers {env_meta.transformers_version}).",
        "",
        "## 4. Multidimensional Semantic Reference Set (Anchors)",
        "Rather than relying on a single query term, candidate paragraphs were evaluated against a multi-anchor semantic reference set representing core dimensions of vulnerability in disaster risk:",
        ""
    ]

    for anchor in reference_set.anchors:
        lines.append(f"- **[{anchor.id}]**: *\"{anchor.text}\"* (Weight: {anchor.weight})")

    lines.extend([
        "",
        f"### Multi-Anchor Aggregation Strategy: `{config.aggregation_strategy.upper()}`",
        f"Individual anchor scores $sim(P_i, Q_j)$ were aggregated using the **{config.aggregation_strategy}** operator.",
        ""
    ])

    if evaluation_summary and evaluation_summary.metrics_per_threshold:
        lines.extend([
            "## 5. Threshold Calibration & Sensitivity Analysis",
            f"- **Calibration Criterion:** {evaluation_summary.calibration_criterion}",
            f"- **Target Recall:** {evaluation_summary.minimum_recall_target * 100:.1f}%",
            f"- **Selected Calibrated Threshold ($\theta$):** `{evaluation_summary.calibrated_threshold:.4f}`",
            f"- **Achieved Performance on Gold Standard:** Recall = {evaluation_summary.achieved_recall * 100:.1f}%, Precision = {evaluation_summary.achieved_precision * 100:.1f}%, F1-Score = {evaluation_summary.achieved_f1:.4f}",
            "",
            "| Threshold | Retrieved | % Corpus | Recall | Precision | F1-Score |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |"
        ])
        for m in evaluation_summary.metrics_per_threshold:
            lines.append(f"| {m.threshold:.2f} | {m.total_retrieved} | {m.pct_corpus * 100:.1f}% | {m.recall * 100:.1f}% | {m.precision * 100:.1f}% | {m.f1_score:.4f} |")
        lines.append("")
    else:
        lines.extend([
            "## 5. Threshold & Retrieval Decision",
            f"- **Threshold Type:** `{config.threshold_type}`",
            f"- **Selected Threshold ($\theta$):** `{config.threshold:.4f}`",
            f"- **Retrieved Candidate Paragraphs:** {manifest.selected_paragraphs} out of {manifest.valid_paragraphs} ({manifest.selected_paragraphs / max(1, manifest.valid_paragraphs) * 100:.1f}% of valid corpus).",
            ""
        ])

    lines.extend([
        "## 6. Methodological Caveats & Disclaimer",
        "1. **Semantic Similarity vs. Scientific Relevance:** Semantic similarity scores measure proximity in dense vector space relative to the anchor set and serve strictly as high-recall screening. High similarity does not automatically imply scientific evidence for the dissertation objectives without expert review.",
        "2. **Reproducibility Guarantee:** All parameters, random seeds (`seed = 42`), model revisions, and environment specifications are permanently logged under this experiment directory for independent replication."
    ])

    return "\n".join(lines)
