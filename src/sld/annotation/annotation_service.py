"""
Serviço de gerenciamento, armazenamento, auditoria e consolidação do Gold Standard com suporte a múltiplos anotadores.
"""

import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from src.sld.models.classification import AnnotationRecord, MultilabelAnnotation, ParagraphRecord
from src.sld.models.concept_label import validate_and_sanitize_labels, labels_to_binary_vector
from src.sld.annotation.markdown_template import (
    generate_annotation_markdown,
    parse_annotation_markdown,
    SLD_ANNOTATION_FORMAT_VERSION
)
from src.sld.utils.files import ensure_directory


class AnnotationService:
    """Gerencia a leitura, gravação, auditoria, importação e exportação de anotações supervisionadas."""

    def __init__(self, annotations_dir_or_file: Path):
        p = Path(annotations_dir_or_file).expanduser().resolve()
        if p.suffix == ".jsonl":
            self.annotations_dir = p.parent
            self.legacy_file = p
        else:
            self.annotations_dir = p
            self.legacy_file = p / "gold_standard.jsonl"

        ensure_directory(self.annotations_dir)
        self.audit_log_path = self.annotations_dir / "annotation_audit_log.jsonl"
        self.exports_dir = self.annotations_dir / "exports"
        self.imports_dir = self.annotations_dir / "imports"
        ensure_directory(self.exports_dir)
        ensure_directory(self.imports_dir)

    def log_audit(self, action: str, paragraph_id: str, annotator_id: str, details: Dict[str, Any]):
        """Registra cada ação de anotação/importação/adjudicação no log auditável."""
        entry = {
            "audit_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "paragraph_id": paragraph_id,
            "annotator_id": annotator_id,
            "details": details
        }
        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load_annotations(self) -> List[AnnotationRecord]:
        """Carrega todas as anotações registradas no diretório / repositório."""
        records: List[AnnotationRecord] = []

        # Tenta carregar do arquivo principal legacy ou parquet
        jsonl_path = self.legacy_file
        candidate_paths = [jsonl_path]

        # Se for um diretório de run específico, adiciona o diretório de anotações global do projeto como fallback
        proj_ann_file = self.annotations_dir.parent.parent / "annotations" / "gold_standard.jsonl"
        if proj_ann_file.exists() and proj_ann_file != jsonl_path:
            candidate_paths.append(proj_ann_file)

        for j_path in candidate_paths:
            if j_path.exists():
                with open(j_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line_str = line.strip()
                        if not line_str:
                            continue
                        try:
                            data = json.loads(line_str)
                            # Suporta legado MultilabelAnnotation e novo AnnotationRecord
                            if "labels" in data and "label_0" not in data:
                                lbls = validate_and_sanitize_labels(data["labels"])
                                rec = AnnotationRecord(
                                    annotation_id=data.get("annotation_id", f"ANN_{data.get('paragraph_id')}_{data.get('annotator', 'anon')}"),
                                    dataset_id=data.get("dataset_id", "ANNOTATION_SET_001"),
                                    run_id=data.get("run_id", ""),
                                    document_id=data.get("article_id", ""),
                                    paragraph_id=data.get("paragraph_id", ""),
                                    annotator_id=data.get("annotator_id", data.get("annotator", "ANN_001")),
                                    annotator_name=data.get("annotator", None) if data.get("annotator") != "pesquisador" else None,
                                    annotation_source="legacy_jsonl",
                                    label_0=(0 in lbls),
                                    label_1=(1 in lbls),
                                    label_2=(2 in lbls),
                                    label_3=(3 in lbls),
                                    label_4=(4 in lbls),
                                    label_5=(5 in lbls),
                                    annotation_status="valid" if lbls else "unannotated",
                                    annotation_note=data.get("annotation_notes", ""),
                                    text_hash=data.get("text_sha256", ""),
                                    created_at=data.get("annotation_timestamp", datetime.now().isoformat()),
                                    included_in_gold_standard=True
                                )
                            else:
                                rec = AnnotationRecord(**data)
                            records.append(rec)
                        except Exception:
                            continue
                if records:
                    break

        return records

    def save_annotations(self, new_records: List[AnnotationRecord]) -> None:
        """Salva uma lista de registros de anotação preservando o rastreamento por (paragraph_id, annotator_id)."""
        existing = self.load_annotations()
        # Chave composta para permitir múltiplos anotadores por parágrafo
        existing_map: Dict[str, AnnotationRecord] = {
            f"{r.paragraph_id}_{r.annotator_id}": r for r in existing
        }

        for rec in new_records:
            key = f"{rec.paragraph_id}_{rec.annotator_id}"
            existing_map[key] = rec
            self.log_audit(
                action="save_annotation",
                paragraph_id=rec.paragraph_id,
                annotator_id=rec.annotator_id,
                details={"labels": rec.labels_list, "status": rec.annotation_status}
            )

        saved_records = list(existing_map.values())

        # Salva em JSONL
        with open(self.legacy_file, "w", encoding="utf-8") as f:
            for r in saved_records:
                f.write(r.model_dump_json() + "\n")

        # Salva também em Parquet para alta performance
        try:
            parquet_path = self.annotations_dir / "annotations.parquet"
            df_ann = pd.DataFrame([r.model_dump() for r in saved_records])
            df_ann.to_parquet(parquet_path, index=False)
        except Exception:
            pass

    def export_annotation_set(
        self,
        paragraphs: List[ParagraphRecord],
        dataset_id: str = "ANNOTATION_SET_001",
        run_id: str = "",
        concept: str = "conceito investigado",
        annotator_name: str = "",
        hide_scores: bool = True,
        blind_mode: bool = False
    ) -> Path:
        """Gera e exporta o arquivo Markdown de anotação supervisionada."""
        md_content = generate_annotation_markdown(
            paragraphs=paragraphs,
            dataset_id=dataset_id,
            run_id=run_id,
            concept=concept,
            annotator_name=annotator_name,
            hide_scores=hide_scores,
            blind_mode=blind_mode
        )

        out_path = self.exports_dir / f"{dataset_id}.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        self.log_audit(
            action="export_markdown",
            paragraph_id="BATCH",
            annotator_id=annotator_name or "ANN_EXPORT",
            details={"dataset_id": dataset_id, "n_paragraphs": len(paragraphs)}
        )

        return out_path

    def validate_import_file(
        self,
        file_content: str,
        file_name: str = "annotation.jsonl",
        corpus_records: Optional[List[ParagraphRecord]] = None
    ) -> Dict[str, Any]:
        """
        Valida e processa arquivos de anotação nos formatos .jsonl, .csv e .md de forma rápida.
        """
        ext = Path(file_name).suffix.lower()

        if ext == ".jsonl":
            return self._validate_import_jsonl(file_content)
        elif ext == ".csv":
            return self._validate_import_csv(file_content)
        else:
            return self.validate_import_markdown(file_content, corpus_records or [])

    def _validate_import_jsonl(self, file_content: str) -> Dict[str, Any]:
        """Validação rápida e direta para arquivos JSONL de anotação (ex: gold_standard.jsonl)."""
        validated_rows = []
        n_valid = 0
        n_invalid = 0
        n_unannotated = 0
        warnings = []

        lines = file_content.strip().split("\n")
        for line_idx, line in enumerate(lines, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                p_id = str(data.get("paragraph_id", ""))
                doc_id = str(data.get("document_id", data.get("article_id", "")))

                if "labels" in data and "label_0" not in data:
                    lbls = validate_and_sanitize_labels(data["labels"])
                    l0, l1, l2, l3, l4, l5 = (0 in lbls), (1 in lbls), (2 in lbls), (3 in lbls), (4 in lbls), (5 in lbls)
                else:
                    l0 = bool(data.get("label_0", False))
                    l1 = bool(data.get("label_1", False))
                    l2 = bool(data.get("label_2", False))
                    l3 = bool(data.get("label_3", False))
                    l4 = bool(data.get("label_4", False))
                    l5 = bool(data.get("label_5", False))
                    lbls = []
                    if l0: lbls.append(0)
                    if l1: lbls.append(1)
                    if l2: lbls.append(2)
                    if l3: lbls.append(3)
                    if l4: lbls.append(4)
                    if l5: lbls.append(5)

                is_unannotated = not (l0 or l1 or l2 or l3 or l4 or l5)
                is_invalid = l0 and (l1 or l2 or l3 or l4 or l5)

                if is_invalid:
                    status = "invalid"
                    error = "Regra da Classe 0 violada: Classe 0 marcada junto com outras classes."
                    n_invalid += 1
                elif is_unannotated:
                    status = "unannotated"
                    error = "Nenhuma classe marcada."
                    n_unannotated += 1
                else:
                    status = data.get("annotation_status", "valid")
                    error = ""
                    n_valid += 1

                validated_rows.append({
                    "paragraph_id": p_id,
                    "document_id": doc_id,
                    "checked_classes": lbls,
                    "labels_str": ", ".join(str(c) for c in lbls) if lbls else "Nenhuma (unannotated)",
                    "status": status,
                    "hash_valid": True,
                    "error": error,
                    "text": data.get("text", ""),
                    "note": data.get("annotation_note", data.get("annotation_notes", "")),
                    "annotator_id": data.get("annotator_id", data.get("annotator", "ANN_001")),
                    "annotator_name": data.get("annotator_name", None),
                    "dataset_id": data.get("dataset_id", "IMPORT_JSONL"),
                    "run_id": data.get("run_id", "")
                })
            except Exception as e:
                warnings.append(f"Linha {line_idx}: formato JSON inválido ({e})")
                n_invalid += 1

        return {
            "format_version": 1,
            "dataset_id": "IMPORT_JSONL",
            "run_id": "",
            "annotator_name": "",
            "total_items": len(validated_rows),
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "n_unannotated": n_unannotated,
            "n_missing_id": 0,
            "n_hash_mismatch": 0,
            "n_conflicts": 0,
            "warnings": warnings,
            "validated_rows": validated_rows
        }

    def _validate_import_csv(self, file_content: str) -> Dict[str, Any]:
        """Validação rápida para arquivos CSV de anotação."""
        import io
        validated_rows = []
        n_valid = 0
        n_invalid = 0
        n_unannotated = 0
        warnings = []

        try:
            df = pd.read_csv(io.StringIO(file_content))
            for _, row in df.iterrows():
                p_id = str(row.get("paragraph_id", ""))
                doc_id = str(row.get("document_id", row.get("article_id", "")))

                if "labels" in row and pd.notna(row["labels"]):
                    lbls_raw = str(row["labels"]).replace(";", ",").split(",")
                    lbls = []
                    for x in lbls_raw:
                        try:
                            lbls.append(int(x.strip()))
                        except Exception:
                            pass
                    l0, l1, l2, l3, l4, l5 = (0 in lbls), (1 in lbls), (2 in lbls), (3 in lbls), (4 in lbls), (5 in lbls)
                else:
                    l0 = bool(row.get("label_0", False))
                    l1 = bool(row.get("label_1", False))
                    l2 = bool(row.get("label_2", False))
                    l3 = bool(row.get("label_3", False))
                    l4 = bool(row.get("label_4", False))
                    l5 = bool(row.get("label_5", False))
                    lbls = []
                    if l0: lbls.append(0)
                    if l1: lbls.append(1)
                    if l2: lbls.append(2)
                    if l3: lbls.append(3)
                    if l4: lbls.append(4)
                    if l5: lbls.append(5)

                is_unannotated = not (l0 or l1 or l2 or l3 or l4 or l5)
                is_invalid = l0 and (l1 or l2 or l3 or l4 or l5)

                if is_invalid:
                    status = "invalid"
                    error = "Regra da Classe 0 violada."
                    n_invalid += 1
                elif is_unannotated:
                    status = "unannotated"
                    error = "Nenhuma classe marcada."
                    n_unannotated += 1
                else:
                    status = "valid"
                    error = ""
                    n_valid += 1

                validated_rows.append({
                    "paragraph_id": p_id,
                    "document_id": doc_id,
                    "checked_classes": lbls,
                    "labels_str": ", ".join(str(c) for c in lbls) if lbls else "Nenhuma (unannotated)",
                    "status": status,
                    "hash_valid": True,
                    "error": error,
                    "text": str(row.get("text", "")),
                    "note": str(row.get("annotation_note", row.get("annotation_notes", ""))),
                    "annotator_id": str(row.get("annotator_id", row.get("annotator", "ANN_001"))),
                    "annotator_name": str(row.get("annotator_name", "")),
                    "dataset_id": str(row.get("dataset_id", "IMPORT_CSV")),
                    "run_id": ""
                })
        except Exception as e:
            warnings.append(f"Erro ao ler arquivo CSV: {e}")

        return {
            "format_version": 1,
            "dataset_id": "IMPORT_CSV",
            "run_id": "",
            "annotator_name": "",
            "total_items": len(validated_rows),
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "n_unannotated": n_unannotated,
            "n_missing_id": 0,
            "n_hash_mismatch": 0,
            "n_conflicts": 0,
            "warnings": warnings,
            "validated_rows": validated_rows
        }

    def validate_import_markdown(
        self,
        markdown_content: str,
        corpus_records: List[ParagraphRecord]
    ) -> Dict[str, Any]:
        """
        Valida um arquivo Markdown de anotação de forma eficiente.
        """
        metadata, items, parse_warnings = parse_annotation_markdown(markdown_content)
        # Otimização: Apenas cria mapa se corpus for menor que 50.000 itens para não sobrecarregar memória
        corpus_map = {r.paragraph_id: r for r in corpus_records} if (corpus_records and len(corpus_records) <= 50000) else {}
        existing_annos = self.load_annotations()
        existing_p_ids = {a.paragraph_id for a in existing_annos}

        validated_rows = []
        n_valid = 0
        n_invalid = 0
        n_unannotated = 0
        n_missing_id = 0
        n_hash_mismatch = 0
        n_conflicts = 0

        for item in items:
            p_id = item["paragraph_id"]
            checked_lbls = item["checked_classes"]
            status = item["status"]
            error_text = item["error"] or ""
            hash_valid = True

            if corpus_map and p_id not in corpus_map:
                status = "invalid"
                error_text = "ID do parágrafo não encontrado no corpus original."
                n_missing_id += 1
            elif corpus_map:
                orig_r = corpus_map.get(p_id)
                if orig_r and item.get("text_hash"):
                    expected_hash = orig_r.text_sha256
                    if expected_hash and item["text_hash"] != expected_hash:
                        hash_valid = False
                        n_hash_mismatch += 1
                        error_text = (error_text + " | Hash do texto divergente do corpus original.").strip(" | ")

            if p_id in existing_p_ids:
                n_conflicts += 1

            if status == "valid":
                n_valid += 1
            elif status == "unannotated":
                n_unannotated += 1
            elif status == "invalid":
                n_invalid += 1

            validated_rows.append({
                "paragraph_id": p_id,
                "document_id": item["document_id"],
                "checked_classes": checked_lbls,
                "labels_str": ", ".join(str(c) for c in checked_lbls) if checked_lbls else "Nenhuma (unannotated)",
                "status": status,
                "hash_valid": hash_valid,
                "error": error_text,
                "text": item["text"] or (corpus_map[p_id].text if p_id in corpus_map else ""),
                "note": item["note"]
            })

        return {
            "format_version": metadata.get("sld_annotation_format", 1),
            "dataset_id": metadata.get("dataset_id", "IMPORT_SET"),
            "run_id": metadata.get("run_id", ""),
            "annotator_name": metadata.get("annotator", ""),
            "total_items": len(items),
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "n_unannotated": n_unannotated,
            "n_missing_id": n_missing_id,
            "n_hash_mismatch": n_hash_mismatch,
            "n_conflicts": n_conflicts,
            "warnings": parse_warnings,
            "validated_rows": validated_rows
        }

    def import_validated_markdown(
        self,
        validation_result: Dict[str, Any],
        annotator_name_override: str = "",
        annotator_id_override: str = ""
    ) -> List[AnnotationRecord]:
        """Incorpora as anotações validadas (Markdown, JSONL ou CSV) ao acervo."""
        dataset_id = validation_result.get("dataset_id", "ANNOTATION_SET_001")
        run_id = validation_result.get("run_id", "")
        ann_name = annotator_name_override or validation_result.get("annotator_name", "")
        ann_id = annotator_id_override or f"ANN_{uuid.uuid4().hex[:6]}"

        imported_records = []
        for row in validation_result["validated_rows"]:
            if row["status"] not in ["valid", "invalid"]:
                continue

            lbls = row["checked_classes"]
            anno_rec = AnnotationRecord(
                annotation_id=f"ANN_{row['paragraph_id']}_{row.get('annotator_id', ann_id)}",
                dataset_id=dataset_id,
                run_id=run_id,
                document_id=row["document_id"],
                paragraph_id=row["paragraph_id"],
                annotator_id=row.get("annotator_id", ann_id),
                annotator_name=row.get("annotator_name", ann_name if ann_name else None),
                annotation_source=f"imported_{dataset_id.lower()}",
                label_0=(0 in lbls),
                label_1=(1 in lbls),
                label_2=(2 in lbls),
                label_3=(3 in lbls),
                label_4=(4 in lbls),
                label_5=(5 in lbls),
                annotation_status=row["status"],
                annotation_note=row.get("note", ""),
                text_hash="",
                imported_at=datetime.now().isoformat(),
                included_in_gold_standard=(row["status"] == "valid")
            )
            imported_records.append(anno_rec)

        self.save_annotations(imported_records)
        self.log_audit(
            action="import_annotations",
            paragraph_id="BATCH",
            annotator_id=ann_id,
            details={"dataset_id": dataset_id, "imported_count": len(imported_records)}
        )

        return imported_records

    def get_gold_standard_records(self) -> List[AnnotationRecord]:
        """Retorna exclusivamente registros de anotação VÁLIDOS e APROVADOS no Gold Standard para treino."""
        all_recs = self.load_annotations()
        return [
            r for r in all_recs
            if r.annotation_status == "valid" and r.included_in_gold_standard and not r.label_0
        ]
