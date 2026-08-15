"""
Módulo de Gerenciamento de Persistência Completa de Sessão e Desacoplamento Modular.
Permite salvar atomicamente e restaurar integralmente o estado de pesquisa do SLD
(artigos, embeddings mmap, similaridades, Gold Standard, modelos supervisionados,
classificação conceitual, índice RAG e extrações LLM) com liberação ativa de memória.
"""

import gc
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.sld.models.article import ArticleMetadata, ProcessedArticle
from src.sld.models.classification import ParagraphRecord, AnnotationRecord
from src.sld.models.search_result import Segment
from src.sld.corpus.analysis_project import AnalysisProject
from src.sld.annotation.annotation_service import AnnotationService
from src.sld.semantic.vector_index import VectorIndex
from src.sld.classification.baseline_classifier import MultilabelLogisticClassifier
from src.sld.rag_index import (
    RAGIndexRetriever,
    RAGIndexManifest,
    IndexStats,
)
from src.sld.utils.atomic import atomic_write_json

logger = logging.getLogger(__name__)


def save_full_session_state(project: AnalysisProject, session_state: Any) -> Dict[str, Any]:
    """
    Persiste em disco o snapshot completo da sessão atual, incluindo:
    - session_state.json no diretório manifests/
    - candidates.parquet no diretório semantic/
    - classified_corpus.parquet no diretório classification/
    - gold_standard.jsonl / annotations.parquet no diretório annotations/
    - model.joblib no diretório classification/
    """
    snapshot = {
        "saved_at": datetime.now().isoformat(),
        "run_id": getattr(session_state, "run_id", "run_default"),
        "completed_steps": list(getattr(session_state, "completed_steps", [])),
        "config": getattr(session_state, "config", {}),
        "funnel_counts": getattr(session_state, "funnel_counts", {}),
        "selected_output_dir": str(project.output_dir),
    }

    # 1. Âncoras e configuração de busca semântica
    sem_ref = getattr(session_state, "reference_set", None) or getattr(session_state, "semantic_reference", None)
    if sem_ref is not None and hasattr(sem_ref, "anchors"):
        snapshot["semantic_reference"] = {
            "anchors": [a.to_dict() if hasattr(a, "to_dict") else (a if isinstance(a, dict) else {"id": getattr(a, "id", "Q1"), "text": getattr(a, "text", "")}) for a in sem_ref.anchors],
            "aggregation_strategy": snapshot["config"].get("aggregation_strategy", "maximum"),
            "similarity_threshold": float(snapshot["config"].get("similarity_threshold", 0.50)),
        }

    # 2. Candidatos da Busca Semântica
    candidates = getattr(session_state, "semantic_candidates", [])
    if candidates:
        cand_parquet_path = project.semantic_dir / "candidates.parquet"
        try:
            cand_rows = []
            for r in candidates:
                cand_rows.append({
                    "paragraph_id": r.paragraph_id,
                    "article_id": r.article_id,
                    "semantic_score": float(r.semantic_score or 0.0),
                    "text": r.text,
                    "status": r.status
                })
            df_cand = pd.DataFrame(cand_rows)
            df_cand.to_parquet(cand_parquet_path, index=False)
            snapshot["semantic_candidates_file"] = str(cand_parquet_path.relative_to(project.output_dir))
            snapshot["semantic_candidates_count"] = len(candidates)
        except Exception as e:
            logger.warning(f"Erro ao salvar candidates.parquet: {e}")

    # 3. Gold Standard e Anotações
    gold_annotations = getattr(session_state, "gold_annotations", [])
    if gold_annotations:
        try:
            ann_srv = AnnotationService(project.annotations_dir)
            ann_srv.save_annotations(gold_annotations)
            snapshot["gold_annotations_count"] = len(gold_annotations)
        except Exception as e:
            logger.warning(f"Erro ao persistir gold standard no projeto: {e}")

    # 4. Modelo de Classificação Supervisionada
    clf = getattr(session_state, "logistic_classifier", None)
    if clf is not None and getattr(clf, "is_fitted", False):
        try:
            clf.save(project.classification_dir)
            snapshot["has_trained_model"] = True
        except Exception as e:
            logger.warning(f"Erro ao salvar classificador supervisionado: {e}")

    opt_thresh = getattr(session_state, "optimal_thresholds", None)
    if opt_thresh:
        snapshot["optimal_thresholds"] = opt_thresh
        atomic_write_json(project.classification_dir / "optimal_thresholds.json", opt_thresh)

    # 5. Corpus Classificado Conceitualmente
    classified = getattr(session_state, "classified_records", [])
    if classified:
        class_parquet_path = project.classification_dir / "classified_corpus.parquet"
        try:
            class_rows = []
            for r in classified:
                class_rows.append({
                    "paragraph_id": r.paragraph_id,
                    "article_id": r.article_id,
                    "status": r.status,
                    "semantic_score": float(r.semantic_score or 0.0),
                    "predicted_labels": json.dumps(r.predicted_labels or []),
                    "predicted_probabilities": json.dumps(r.predicted_probabilities or {}),
                    "text": r.text
                })
            df_class = pd.DataFrame(class_rows)
            df_class.to_parquet(class_parquet_path, index=False)
            snapshot["classified_corpus_file"] = str(class_parquet_path.relative_to(project.output_dir))
            snapshot["classified_records_count"] = len(classified)
        except Exception as e:
            logger.warning(f"Erro ao salvar classified_corpus.parquet: {e}")

    # 6. Metadados do Índice RAG
    rag_stats = getattr(session_state, "rag_index_stats", None)
    if rag_stats is not None:
        snapshot["rag_index_stats"] = rag_stats.model_dump()

    rag_manifest = getattr(session_state, "rag_index_manifest", None)
    if rag_manifest is not None:
        snapshot["rag_index_manifest"] = rag_manifest.model_dump()

    snapshot["rag_index_zip_path"] = getattr(session_state, "rag_index_zip_path", None)

    # 7. Salva o snapshot atômico
    project.save_session_snapshot(snapshot)

    # Coleta de lixo ativa
    gc.collect()

    return snapshot


def restore_full_session_state(project: AnalysisProject, session_state: Any) -> Dict[str, Any]:
    """
    Restaura integralmente o estado da pesquisa a partir dos arquivos salvos na pasta de saída.
    Aplica desacoplamento e memória mapeada (mmap) para evitar sobrecarga de RAM.
    """
    restored_items = []

    # 1. Restaura snapshot de configuração se existir
    snapshot = project.load_session_snapshot() or {}

    if "run_id" in snapshot and hasattr(session_state, "run_id"):
        session_state.run_id = snapshot["run_id"]

    if "completed_steps" in snapshot and hasattr(session_state, "completed_steps"):
        for step in snapshot["completed_steps"]:
            if step not in session_state.completed_steps:
                session_state.completed_steps.append(step)

    if "config" in snapshot and hasattr(session_state, "config"):
        session_state.config.update(snapshot["config"])

    if "funnel_counts" in snapshot and hasattr(session_state, "funnel_counts"):
        session_state.funnel_counts.update(snapshot["funnel_counts"])

    # 2. Restaura Artigos e Markdowns (Etapa 1)
    if not getattr(session_state, "articles_records", None):
        reg_records = project.registry.load_registry()
        articles_recs = []
        if reg_records:
            for reg in reg_records.values():
                md_p = Path(reg.markdown_path)
                if not md_p.is_absolute():
                    md_p = project.output_dir / reg.markdown_path
                md_txt = ""
                if md_p.exists():
                    try:
                        with open(md_p, "r", encoding="utf-8") as f:
                            md_txt = f.read()
                    except Exception:
                        pass

                meta_obj = ArticleMetadata(
                    sld_id=reg.article_id,
                    source_pdf=reg.source_filename,
                    source_path=reg.source_path,
                    pdf_sha256=reg.pdf_sha256,
                    processed_at=reg.last_updated_at,
                    references_removed=(reg.reference_removal_status == "success")
                )
                articles_recs.append(ProcessedArticle(
                    metadata=meta_obj,
                    markdown_content=md_txt,
                    markdown_path=str(md_p),
                    page_count=reg.page_count,
                    char_count=reg.character_count,
                    status="success" if reg.processing_status == "completed" else reg.processing_status,
                    error_message=reg.error_message
                ))
        else:
            # Fallback: Varre arquivos .md existentes no diretório markdown/ ou raiz
            md_files = list(project.markdown_dir.glob("*.md")) or list(project.output_dir.glob("*.md"))
            for mf in md_files:
                if mf.name in ["README.md", "index_report.md"]:
                    continue
                try:
                    with open(mf, "r", encoding="utf-8") as f:
                        md_txt = f.read()
                except Exception:
                    md_txt = ""
                doc_id = mf.stem
                meta_obj = ArticleMetadata(
                    sld_id=doc_id,
                    source_pdf=f"{doc_id}.pdf",
                    source_path=str(mf),
                    processed_at=datetime.now().isoformat(),
                    references_removed=True
                )
                articles_recs.append(ProcessedArticle(
                    metadata=meta_obj,
                    markdown_content=md_txt,
                    markdown_path=str(mf),
                    page_count=1,
                    char_count=len(md_txt),
                    status="success"
                ))

        session_state.articles_records = articles_recs
        if articles_recs:
            restored_items.append(f"{len(articles_recs)} artigos processados")
            if 1 not in session_state.completed_steps:
                session_state.completed_steps.append(1)

    # 3. Restaura VectorIndex e Matriz de Embeddings via mmap (Etapa 2 e 3)
    vec_index = VectorIndex(project.index_dir)
    segments_path = project.index_dir / "segments.jsonl"
    if not segments_path.exists():
        for fallback_seg in [project.output_dir / "segments.jsonl", project.output_dir / "corpus.jsonl", project.output_dir / "paragraphs" / "corpus.jsonl"]:
            if fallback_seg.exists():
                segments_path = fallback_seg
                break

    # Carrega segmentos do disco se existirem
    if segments_path.exists() and not getattr(session_state, "corpus_records", None):
        seg_paras = []
        try:
            with open(segments_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        p_rec = ParagraphRecord(
                            paragraph_id=data.get("paragraph_id", data.get("segment_id", "")),
                            article_id=data.get("article_id", data.get("document_id", "")),
                            text=data.get("text", ""),
                            status="INGESTED"
                        )
                        seg_paras.append(p_rec)
            session_state.corpus_records = seg_paras
            restored_items.append(f"{len(seg_paras):,} parágrafos carregados do arquivo segments.jsonl")
        except Exception as e:
            logger.warning(f"Erro ao ler segments.jsonl: {e}")

    # Fallback para gerar parágrafos a partir dos markdowns se segments.jsonl não existir
    if not getattr(session_state, "corpus_records", None) and getattr(session_state, "articles_records", None):
        from src.sld.segmentation.segmenter import Segmenter
        cfg_seg = getattr(session_state, "config", {})
        segmenter = Segmenter(
            min_words=cfg_seg.get("min_words", 8),
            min_characters=cfg_seg.get("min_characters", 40),
            max_characters=cfg_seg.get("max_characters", 500),
            long_text_strategy=cfg_seg.get("long_text_strategy", "chunk")
        )
        gen_paras = []
        for art in session_state.articles_records:
            if art.markdown_content:
                segs = segmenter.segment_markdown(
                    markdown_text=art.markdown_content,
                    article_id=art.metadata.sld_id,
                    source_pdf=art.metadata.source_pdf,
                    markdown_path=art.markdown_path
                )
                for s in segs:
                    gen_paras.append(ParagraphRecord(
                        paragraph_id=s.paragraph_id,
                        article_id=s.article_id,
                        text=s.text,
                        status="INGESTED"
                    ))
        if gen_paras:
            session_state.corpus_records = gen_paras
            restored_items.append(f"{len(gen_paras):,} parágrafos gerados a partir dos Markdowns")

    # Carrega matriz de embeddings com mmap se existir no disco
    emb_path = project.index_dir / "embeddings.npy"
    if not emb_path.exists() and (project.output_dir / "embeddings.npy").exists():
        emb_path = project.output_dir / "embeddings.npy"

    if emb_path.exists() and getattr(session_state, "embeddings_matrix", None) is None:
        try:
            session_state.embeddings_matrix = np.load(emb_path, mmap_mode="r")
            restored_items.append(f"Matriz de embeddings ({session_state.embeddings_matrix.shape[0]:,} vetores D={session_state.embeddings_matrix.shape[1]}) carregada via mmap")
            if 2 not in session_state.completed_steps:
                session_state.completed_steps.append(2)
        except Exception as e:
            logger.warning(f"Erro ao carregar embeddings.npy: {e}")

    # 4. Restaura Âncoras e Candidatos de Similaridade Semântica (Etapa 4)
    if "semantic_reference" in snapshot:
        sem_info = snapshot["semantic_reference"]
        if "anchors" in sem_info:
            from src.sld.semantic.semantic_reference import SemanticReferenceSet, SemanticAnchor
            loaded_anchors = [
                SemanticAnchor.from_dict(d) if isinstance(d, dict) else d
                for d in sem_info["anchors"]
            ]
            new_ref_set = SemanticReferenceSet(anchors=loaded_anchors)
            if hasattr(session_state, "reference_set"):
                session_state.reference_set = new_ref_set
            if hasattr(session_state, "semantic_reference"):
                session_state.semantic_reference = new_ref_set

        if "aggregation_strategy" in sem_info and hasattr(session_state, "config"):
            session_state.config["aggregation_strategy"] = sem_info["aggregation_strategy"]

    cand_parquet = project.semantic_dir / "candidates.parquet"
    if cand_parquet.exists() and not getattr(session_state, "semantic_candidates", None):
        try:
            df_cand = pd.read_parquet(cand_parquet)
            cand_list = []
            scores_map = {}
            for _, row in df_cand.iterrows():
                p_rec = ParagraphRecord(
                    paragraph_id=str(row["paragraph_id"]),
                    article_id=str(row["article_id"]),
                    text=str(row["text"]),
                    status=str(row.get("status", "VALID_PARAGRAPH")),
                    semantic_score=float(row.get("semantic_score", 0.0))
                )
                cand_list.append(p_rec)
                composite_key = f"{p_rec.article_id}_{p_rec.paragraph_id}"
                scores_map[composite_key] = p_rec.semantic_score
                scores_map[p_rec.paragraph_id] = p_rec.semantic_score

            session_state.semantic_candidates = cand_list
            session_state.semantic_scores_map = scores_map

            # Atualiza scores no corpus geral se disponível
            if getattr(session_state, "corpus_records", None):
                for r in session_state.corpus_records:
                    if r.paragraph_id in scores_map:
                        r.semantic_score = scores_map[r.paragraph_id]

            restored_items.append(f"{len(cand_list):,} candidatos semânticos")
            if 4 not in session_state.completed_steps:
                session_state.completed_steps.append(4)
        except Exception as e:
            logger.warning(f"Erro ao restaurar candidates.parquet: {e}")

    # 5. Restaura Gold Standard (Etapa 5)
    ann_srv = AnnotationService(project.annotations_dir)
    loaded_anns = ann_srv.load_annotations()
    if loaded_anns:
        session_state.gold_annotations = loaded_anns
        restored_items.append(f"{len(loaded_anns)} anotações Gold Standard")

    # 6. Restaura Modelo Treinado de Regressão Logística (Etapa 5)
    model_exists = (project.classification_dir / "logistic_classifier.joblib").exists() or (project.classification_dir / "baseline_model.joblib").exists()
    if model_exists and not getattr(session_state, "logistic_classifier", None):
        try:
            loaded_clf = MultilabelLogisticClassifier.load(project.classification_dir)
            session_state.logistic_classifier = loaded_clf
            restored_items.append("Classificador supervisionado treinado")
            if 5 not in session_state.completed_steps:
                session_state.completed_steps.append(5)
        except Exception as e:
            logger.warning(f"Erro ao carregar modelo de classificação: {e}")

    opt_thresh_file = project.classification_dir / "optimal_thresholds.json"
    if opt_thresh_file.exists():
        try:
            with open(opt_thresh_file, "r", encoding="utf-8") as f:
                session_state.optimal_thresholds = json.load(f)
        except Exception:
            pass

    # 7. Restaura Corpus Classificado Conceitualmente (Etapa 6)
    class_parquet = project.classification_dir / "classified_corpus.parquet"
    if class_parquet.exists() and not getattr(session_state, "classified_records", None):
        try:
            df_class = pd.read_parquet(class_parquet)
            class_list = []
            class_map = {}
            for _, row in df_class.iterrows():
                lbls = json.loads(row["predicted_labels"]) if isinstance(row["predicted_labels"], str) else list(row["predicted_labels"])
                probs = json.loads(row["predicted_probabilities"]) if isinstance(row["predicted_probabilities"], str) else dict(row["predicted_probabilities"])
                p_rec = ParagraphRecord(
                    paragraph_id=str(row["paragraph_id"]),
                    article_id=str(row["article_id"]),
                    text=str(row["text"]),
                    status=str(row.get("status", "MODEL_RELEVANT")),
                    semantic_score=float(row.get("semantic_score", 0.0)),
                    predicted_labels=lbls,
                    predicted_probabilities=probs
                )
                class_list.append(p_rec)
                composite_k = f"{p_rec.article_id}_{p_rec.paragraph_id}"
                class_map[composite_k] = p_rec
                class_map[p_rec.paragraph_id] = p_rec

            session_state.classified_records = class_list

            # Sincroniza status no corpus geral
            if getattr(session_state, "corpus_records", None):
                for r in session_state.corpus_records:
                    if r.paragraph_id in class_map:
                        matched = class_map[r.paragraph_id]
                        r.status = matched.status
                        r.predicted_labels = matched.predicted_labels
                        r.predicted_probabilities = matched.predicted_probabilities

            restored_items.append(f"{len(class_list):,} parágrafos classificados")
            if 6 not in session_state.completed_steps:
                session_state.completed_steps.append(6)
        except Exception as e:
            logger.warning(f"Erro ao restaurar classified_corpus.parquet: {e}")

    # 8. Restaura Índice RAG (Etapa 7)
    rag_dir = project.rag_index_dir
    if not (rag_dir / "corpus_refinado.faiss").exists() and (rag_dir / "rag_index" / "corpus_refinado.faiss").exists():
        rag_dir = rag_dir / "rag_index"

    rag_faiss_file = rag_dir / "corpus_refinado.faiss"
    rag_parquet_file = rag_dir / "metadata.parquet"
    if rag_faiss_file.exists() and rag_parquet_file.exists() and not getattr(session_state, "rag_retriever", None):
        try:
            retriever = RAGIndexRetriever()
            retriever.load_from_dir(rag_dir)
            session_state.rag_retriever = retriever

            if retriever.manifest:
                session_state.rag_index_manifest = retriever.manifest

            if "rag_index_stats" in snapshot:
                session_state.rag_index_stats = IndexStats(**snapshot["rag_index_stats"])

            if "rag_index_zip_path" in snapshot and snapshot["rag_index_zip_path"]:
                session_state.rag_index_zip_path = snapshot["rag_index_zip_path"]

            restored_items.append(f"Índice RAG FAISS ({retriever.faiss_index.ntotal:,} vetores)")
            if 7 not in session_state.completed_steps:
                session_state.completed_steps.append(7)
        except Exception as e:
            logger.warning(f"Erro ao carregar índice RAG existente: {e}")

    # Libera memória imediatamente
    gc.collect()

    return {
        "restored_count": len(restored_items),
        "restored_items": restored_items,
        "completed_steps": getattr(session_state, "completed_steps", [])
    }


def inspect_stage_files(project: AnalysisProject) -> Dict[str, Any]:
    """
    Inspeciona a pasta de saída para descobrir quais artefatos de cada etapa já estão disponíveis no disco.
    """
    # Etapa 1
    md_files = list(project.markdown_dir.glob("*.md")) or list(project.output_dir.glob("*.md"))
    md_files = [f for f in md_files if f.name not in ["README.md", "index_report.md"]]
    has_registry = (project.manifests_dir / "article_registry.jsonl").exists()

    # Etapa 2 / 3
    emb_file = project.index_dir / "embeddings.npy" if (project.index_dir / "embeddings.npy").exists() else (project.output_dir / "embeddings.npy" if (project.output_dir / "embeddings.npy").exists() else None)
    seg_file = project.index_dir / "segments.jsonl" if (project.index_dir / "segments.jsonl").exists() else (project.output_dir / "segments.jsonl" if (project.output_dir / "segments.jsonl").exists() else None)

    # Etapa 4
    cand_file = project.semantic_dir / "candidates.parquet" if (project.semantic_dir / "candidates.parquet").exists() else (project.output_dir / "candidates.parquet" if (project.output_dir / "candidates.parquet").exists() else None)

    # Etapa 5
    gold_file = project.annotations_dir / "gold_standard.jsonl" if (project.annotations_dir / "gold_standard.jsonl").exists() else (project.output_dir / "gold_standard.jsonl" if (project.output_dir / "gold_standard.jsonl").exists() else None)
    ann_parquet = project.annotations_dir / "annotations.parquet" if (project.annotations_dir / "annotations.parquet").exists() else None
    model_file = project.classification_dir / "logistic_classifier.joblib" if (project.classification_dir / "logistic_classifier.joblib").exists() else (project.classification_dir / "baseline_model.joblib" if (project.classification_dir / "baseline_model.joblib").exists() else None)

    # Etapa 6
    classified_file = project.classification_dir / "classified_corpus.parquet" if (project.classification_dir / "classified_corpus.parquet").exists() else (project.output_dir / "classified_corpus.parquet" if (project.output_dir / "classified_corpus.parquet").exists() else None)

    # Etapa 7
    rag_dir = project.rag_index_dir
    if not (rag_dir / "corpus_refinado.faiss").exists() and (rag_dir / "rag_index" / "corpus_refinado.faiss").exists():
        rag_dir = rag_dir / "rag_index"
    rag_faiss = rag_dir / "corpus_refinado.faiss" if (rag_dir / "corpus_refinado.faiss").exists() else None
    rag_parquet = rag_dir / "metadata.parquet" if (rag_dir / "metadata.parquet").exists() else None

    # Etapa 8
    llm_file = project.llm_dir / "refined_corpus_llm.jsonl" if (project.llm_dir / "refined_corpus_llm.jsonl").exists() else (project.output_dir / "refined_corpus_llm.jsonl" if (project.output_dir / "refined_corpus_llm.jsonl").exists() else None)

    # Snapshot
    snapshot_file = project.session_snapshot_path if project.session_snapshot_path.exists() else None

    return {
        "stage_1": {
            "has_files": bool(md_files or has_registry),
            "md_count": len(md_files),
            "has_registry": has_registry,
            "description": f"{len(md_files)} arquivos .md encontrados" if md_files else ("Registro de artigos encontrado" if has_registry else "Nenhum markdown encontrado")
        },
        "stage_2": {
            "has_files": bool(emb_file and seg_file),
            "has_embeddings": bool(emb_file),
            "has_segments": bool(seg_file),
            "emb_path": str(emb_file) if emb_file else None,
            "seg_path": str(seg_file) if seg_file else None,
            "description": "Matriz embeddings.npy e segments.jsonl encontrados" if (emb_file and seg_file) else ("Apenas segments.jsonl" if seg_file else ("Apenas embeddings.npy" if emb_file else "Índice vetorial ausente"))
        },
        "stage_3": {
            "has_files": bool(seg_file or md_files),
            "description": "Segmentos prontos para análise exploratória" if seg_file else (f"{len(md_files)} markdowns prontos para segmentação e análise" if md_files else "Corpus não encontrado")
        },
        "stage_4": {
            "has_files": bool(cand_file),
            "cand_path": str(cand_file) if cand_file else None,
            "description": "Arquivo candidates.parquet encontrado" if cand_file else "Busca semântica ainda não executada"
        },
        "stage_5": {
            "has_files": bool(gold_file or ann_parquet or model_file),
            "has_gold": bool(gold_file or ann_parquet),
            "has_model": bool(model_file),
            "description": "Gold Standard e Modelo Treinado encontrados" if ((gold_file or ann_parquet) and model_file) else ("Apenas Gold Standard encontrado" if (gold_file or ann_parquet) else ("Apenas Modelo Treinado" if model_file else "Anotações ausentes"))
        },
        "stage_6": {
            "has_files": bool(classified_file),
            "classified_path": str(classified_file) if classified_file else None,
            "description": "Arquivo classified_corpus.parquet encontrado" if classified_file else "Classificação conceitual ainda não executada"
        },
        "stage_7": {
            "has_files": bool(rag_faiss and rag_parquet),
            "rag_path": str(rag_faiss) if rag_faiss else None,
            "description": "Índice FAISS corpus_refinado.faiss e metadata.parquet encontrados" if (rag_faiss and rag_parquet) else "Índice RAG ausente"
        },
        "stage_8": {
            "has_files": bool(llm_file),
            "description": "Extrações LLM salvas em disco" if llm_file else "Extração LLM não executada"
        },
        "snapshot": {
            "has_snapshot": bool(snapshot_file),
            "snapshot_path": str(snapshot_file) if snapshot_file else None
        }
    }

