"""
Gerenciamento do estado do corpus e repositório não destrutivo de parágrafos.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
import pandas as pd
from src.sld.models.classification import ParagraphRecord
from src.sld.models.concept_label import CONCEPT_LABEL_SHORT_NAMES, binary_vector_to_labels
from src.sld.utils.files import ensure_directory


class CorpusRepository:
    """
    Gerencia a lista completa de parágrafos do acervo e seus estados sem exclusão física.
    Estados: RAW, SEMANTIC_CANDIDATE, SEMANTIC_REJECTED, MANUALLY_ANNOTATED, MODEL_RELEVANT, MODEL_NOT_RELEVANT, FINAL_CORPUS.
    """

    def __init__(self, corpus_jsonl_path: Path):
        p = Path(corpus_jsonl_path).expanduser().resolve()
        if p.is_dir() or not p.suffix:
            self.corpus_path = p / "corpus.jsonl"
        else:
            self.corpus_path = p
        ensure_directory(self.corpus_path.parent)

    def load_records(self) -> List[ParagraphRecord]:
        """Carrega todos os registros do corpus."""
        if not self.corpus_path.exists():
            return []

        records: List[ParagraphRecord] = []
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    records.append(ParagraphRecord(**json.loads(line_str)))
                except Exception:
                    continue

        return records

    def save_records(self, records: List[ParagraphRecord]) -> None:
        """Salva todos os registros preservando a integridade dos estados."""
        ensure_directory(self.corpus_path.parent)
        existing_map: Dict[str, ParagraphRecord] = {
            f"{r.article_id}_{r.paragraph_id}": r for r in self.load_records()
        }

        for r in records:
            key = f"{r.article_id}_{r.paragraph_id}"
            existing_map[key] = r

        with open(self.corpus_path, "w", encoding="utf-8") as f:
            for rec in existing_map.values():
                f.write(rec.model_dump_json() + "\n")

    def save_corpus_records(self, records: List[ParagraphRecord]) -> None:
        """Alias para save_records."""
        self.save_records(records)

    def get_final_corpus(self) -> List[ParagraphRecord]:
        """Retorna exclusivamente os parágrafos selecionados no FINAL_CORPUS."""
        records = self.load_records()
        return [r for r in records if r.status == "FINAL_CORPUS" or "MODEL_RELEVANT" in r.status]

    def export_corpus(self, output_path: Path, format_type: str = "csv") -> Path:
        """Exporta os registros do corpus final para CSV, JSONL, Parquet ou Markdown."""
        out_path = Path(output_path).expanduser().resolve()
        ensure_directory(out_path.parent)
        records = self.get_final_corpus()

        rows = []
        for r in records:
            row = {
                "article_id": r.article_id,
                "paragraph_id": r.paragraph_id,
                "section": r.section,
                "text": r.text,
                "semantic_score": r.semantic_score,
                "status": r.status,
                "predicted_labels": ", ".join(r.predicted_labels),
            }

            # Adiciona probabilidades de cada classe
            for c_name, p_val in r.predicted_probabilities.items():
                row[f"prob_{c_name}"] = p_val

            rows.append(row)

        df = pd.DataFrame(rows)

        if format_type == "csv":
            df.to_csv(out_path, index=False, encoding="utf-8")
        elif format_type == "jsonl":
            df.to_json(out_path, orient="records", lines=True, force_ascii=False)
        elif format_type == "parquet":
            df.to_parquet(out_path, index=False)
        elif format_type == "markdown":
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"# Corpus Final Selecionado ({len(records)} Parágrafos)\n\n")
                for r in records:
                    f.write(f"### Parágrafo `{r.paragraph_id}` (Artigo: `{r.article_id}`)\n")
                    f.write(f"- **Seção:** {r.section}\n")
                    f.write(f"- **Score Semântico:** {r.semantic_score:.4f}\n")
                    f.write(f"- **Rólutos Preditos:** {', '.join(r.predicted_labels)}\n\n")
                    f.write(f"> {r.text}\n\n---\n\n")

        return out_path
