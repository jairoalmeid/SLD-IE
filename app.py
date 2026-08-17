"""
Aplicação Principal Streamlit — SLD (Scientific Literature Decoder)
Pipeline Metodológico em 8 Abas Científicas para Pesquisa de Doutorado.
Refatoração Visual de Alta Sobriedade Acadêmica e Download em Todas as Etapas (.md, .csv, .jsonl).
"""

import time
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

from config.settings import (
    BASE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_INDEX_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_SEMANTIC_BATCH_SIZE,
    SYSTEM_PROMPT_VERSION,
    create_run_id,
    get_run_dir_structure,
    get_default_config
)
from src.sld.models.concept_label import (
    MULTILABEL_CLASSES,
    CONCEPT_LABEL_NAMES,
    CONCEPT_LABEL_SHORT_NAMES,
    CONCEPT_LABEL_DESCRIPTIONS,
    validate_and_sanitize_labels
)
from src.sld.models.article import ArticleMetadata, ProcessedArticle
from src.sld.models.search_result import Segment
from src.sld.models.classification import ParagraphRecord, MultilabelAnnotation, AnnotationRecord, ModelVersionMetadata
from src.sld.utils.files import validate_directory, find_pdf_files, ensure_directory, open_folder_dialog
from src.sld.ingestion.pdf_reader import read_pdf_content
from src.sld.ingestion.zip_extractor import extract_zip_safely, find_all_inputs
from src.sld.ingestion.reference_remover import remove_references
from src.sld.ingestion.markdown_writer import write_markdown_file, generate_markdown_content
from src.sld.ingestion.segmenter import segment_markdown_paragraphs
from src.sld.semantic.embedding_service import EmbeddingService
from src.sld.semantic.semantic_reference import SemanticReferenceSet
from src.sld.semantic.semantic_search import perform_multi_anchor_search, perform_semantic_search, compute_per_anchor_statistics
from src.sld.analysis.corpus_analysis import (
    compute_corpus_descriptors,
    compute_top_terms,
    generate_wordcloud_image,
    compute_cooccurrence_matrix
)
from src.sld.classification.baseline_classifier import MultilabelLogisticClassifier
from src.sld.annotation.active_learning_sampling import sample_paragraphs
from src.sld.evaluation.group_splitter import split_annotations_by_article, verify_no_data_leakage
from src.sld.evaluation.multilabel_evaluator import (
    compute_multilabel_evaluation,
    compute_pr_curves_data,
    generate_evaluation_tables,
    generate_evaluation_markdown
)
from src.sld.evaluation.inter_annotator_agreement import compute_inter_annotator_agreement
from src.sld.models.concept_label import (
    MULTILABEL_CLASSES,
    CONCEPT_LABEL_NAMES,
    CONCEPT_LABEL_SHORT_NAMES
)
from src.sld.models.classification import EvaluationReport
from src.sld.annotation.annotation_service import AnnotationService
from src.sld.corpus.corpus_repository import CorpusRepository
from src.sld.models.llm_extraction import ExtractionOutput, LLMParagraphResult
from src.sld.llm.llm_provider import OllamaProvider, MockLLMProvider
from src.sld.llm.extraction_service import LLMExtractionService
from src.sld.corpus.analysis_project import AnalysisProject, AnalysisMetadata
from src.sld.corpus.duplicate_controller import ArticleRegistry, ArticleRegistryRecord, DuplicateSummary
from src.sld.corpus.checkpoint_manager import CheckpointManager, OperationCheckpoint
from src.sld.corpus.integrity_checker import AnalysisIntegrityChecker, IntegrityReport
from src.sld.semantic.vector_index import VectorIndex
from src.sld.semantic.embeddings_tracker import EmbeddingsTracker, PerArticleEmbeddingRecord
from src.sld.utils.atomic import atomic_write_json
from src.sld.utils.hashing import calculate_file_sha256, calculate_text_sha256, generate_article_id
from src.sld.ui.styles import inject_custom_styles
from src.sld.ui.tracker import ProgressTracker
from src.sld.ui.components import (
    render_institutional_header,
    render_pipeline_stepper,
    render_methodology_header,
    render_pipeline_metrics,
    render_completion_panel,
    render_export_section,
    render_empty_state,
    render_status_card,
    render_formula,
    render_descriptive_statistics,
    render_interpretation_box,
    render_methodological_alert,
    dataframe_to_markdown,
    render_stage_disk_loader
)
from src.sld.rag_index import (
    RAGIndexConfig,
    RAGIndexManifest,
    IndexStats,
    CorpusDistributionStats,
    CoverageStats,
    RAGQueryResult,
    RAGIndexBuilder,
    RAGIndexRetriever,
    compute_corpus_distribution_stats,
    compute_coverage_stats,
)
from src.sld.corpus.session_manager import (
    save_full_session_state,
    restore_full_session_state,
    inspect_stage_files,
    export_classified_corpus_to_disk,
    export_semantic_candidates_to_disk,
    export_anchor_statistics_to_disk,
    export_evaluation_report_to_disk
)
from src.sld.reports.methodology_report import generate_methodology_report


# Configuração Global da Página Streamlit
st.set_page_config(
    page_title="SLD — Scientific Literature Decoder",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Inicialização de Estado da Sessão (Session State)
if "run_id" not in st.session_state:
    st.session_state.run_id = create_run_id()

if "run_dirs" not in st.session_state:
    st.session_state.run_dirs = get_run_dir_structure(DEFAULT_OUTPUT_DIR, st.session_state.run_id)

if "config" not in st.session_state:
    st.session_state.config = get_default_config()

if "last_session_save" not in st.session_state:
    st.session_state.last_session_save = None

if "articles_records" not in st.session_state:
    st.session_state.articles_records = []

if "corpus_records" not in st.session_state:
    st.session_state.corpus_records = []

if "embeddings_matrix" not in st.session_state:
    st.session_state.embeddings_matrix = None

if "reference_set" not in st.session_state:
    st.session_state.reference_set = SemanticReferenceSet(anchors=[])

if "semantic_reference" not in st.session_state:
    st.session_state.semantic_reference = st.session_state.reference_set

if "semantic_candidates" not in st.session_state:
    st.session_state.semantic_candidates = []

if "semantic_scores_map" not in st.session_state:
    st.session_state.semantic_scores_map = {}

if "gold_annotations" not in st.session_state:
    st.session_state.gold_annotations = []

if "logistic_classifier" not in st.session_state:
    st.session_state.logistic_classifier = None

if "classified_records" not in st.session_state:
    st.session_state.classified_records = []

if "llm_results" not in st.session_state:
    st.session_state.llm_results = []

if "completed_steps" not in st.session_state:
    st.session_state.completed_steps = []

if "funnel_counts" not in st.session_state:
    st.session_state.funnel_counts = {}

if "active_learning_step" not in st.session_state:
    st.session_state.active_learning_step = 1

if "active_sampling_batch" not in st.session_state:
    st.session_state.active_sampling_batch = []

if "current_uncertain_samples" not in st.session_state:
    st.session_state.current_uncertain_samples = []

if "optimal_thresholds" not in st.session_state:
    st.session_state.optimal_thresholds = None

if "evaluation_report" not in st.session_state:
    st.session_state.evaluation_report = None

if "eval_report" not in st.session_state:
    st.session_state.eval_report = None

if "exploratory_results" not in st.session_state:
    st.session_state.exploratory_results = None

if "llm_stats" not in st.session_state:
    st.session_state.llm_stats = None

if "rag_index_stats" not in st.session_state:
    st.session_state.rag_index_stats = None

if "rag_index_manifest" not in st.session_state:
    st.session_state.rag_index_manifest = None

if "rag_index_zip_path" not in st.session_state:
    st.session_state.rag_index_zip_path = None

if "rag_retriever" not in st.session_state:
    st.session_state.rag_retriever = None

if "rag_query_results" not in st.session_state:
    st.session_state.rag_query_results = []


def sync_project_state(output_dir_path: Path):
    """
    Sincroniza e restaura integralmente o estado do projeto a partir da pasta de saída
    (Artigos, Embeddings mmap, Similaridade, Gold Standard, Modelos, Classificados e RAG Index).
    """
    project = AnalysisProject(output_dir_path)
    st.session_state.analysis_project = project

    if project.is_existing_project():
        meta = project.load_metadata()
        st.session_state.analysis_metadata = meta

        if meta.embedding_model:
            st.session_state.config["embedding_model"] = meta.embedding_model
        if meta.segmentation_config:
            st.session_state.config.update(meta.segmentation_config)
        if meta.reference_removal_config:
            st.session_state.config.update(meta.reference_removal_config)

    # Executa restauração completa desacoplada de tudo que estiver no disco
    res_info = restore_full_session_state(project, st.session_state)
    return res_info


def render_supervised_evaluation_panel(
    eval_report: EvaluationReport,
    project: AnalysisProject,
    run_id: str,
    key_suffix: str = "main"
):
    """
    Renderiza painel quantitativo e metodológico de avaliação supervisionada com:
    - Alertas de Integridade e Vazamento de Dados (Data Leakage)
    - Métricas Globais das Classes Ativas 1 a 5 (Macro/Micro F1 com IC95% Bootstrap)
    - Tabela Principal das 5 Classes Ativas (com suporte positivo, prevalência, especificidade, IC95%, AP e ROC-AUC)
    - Diagnóstico Separado da Classe 0 (Não Relevante — Derivada e Excludente)
    - Variação da Validação Cruzada (se houver)
    - Gráficos Comparativos Plotly e Matrizes de Confusão 2×2
    - Persistência em Disco e Botões de Download
    """
    df_g, df_c, df_c0, df_cv = generate_evaluation_tables(eval_report)
    md_text = generate_evaluation_markdown(eval_report)

    # 1. Alertas de Integridade Metodológica e Vazamento de Dados
    if eval_report.methodological_alerts:
        for alert in eval_report.methodological_alerts:
            if "🚨" in alert or "❌" in alert:
                st.error(alert)
            elif "⚠️" in alert:
                st.warning(alert)
            else:
                st.info(alert)

    # 2. Métricas Globais das Classes Ativas (1 a 5)
    st.markdown("#### 🎯 Métricas Globais das Classes Ativas (1 a 5)")
    st.caption("As métricas agregadas principais são computadas estritamente sobre as 5 classes ativas.")

    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
    m_f1_str = f"{eval_report.macro_f1:.4f}" if eval_report.macro_f1 is not None else "N/A"
    if eval_report.macro_f1_ci95:
        m_f1_help = f"Macro-F1: Média não ponderada do F1 das classes ativas (IC 95%: [{eval_report.macro_f1_ci95[0]:.4f}, {eval_report.macro_f1_ci95[1]:.4f}])."
    else:
        m_f1_help = "Macro-F1: Média não ponderada do F1 das classes ativas."
    col_m1.metric("Macro-F1", m_f1_str, help=m_f1_help)

    mic_f1_str = f"{eval_report.micro_f1:.4f}" if eval_report.micro_f1 is not None else "N/A"
    col_m2.metric("Micro-F1", mic_f1_str, help="F1 global calculado a partir da soma agregada de TP, FP e FN.")

    m_p_str = f"{eval_report.macro_precision:.4f}" if eval_report.macro_precision is not None else "N/A"
    col_m3.metric("Macro-Precision", m_p_str, help="Média das taxas de precisão positiva entre as classes 1 a 5.")

    m_r_str = f"{eval_report.macro_recall:.4f}" if eval_report.macro_recall is not None else "N/A"
    col_m4.metric("Macro-Recall", m_r_str, help="Média das taxas de sensibilidade/recuperação entre as classes 1 a 5.")

    em_str = f"{eval_report.subset_accuracy:.4f}"
    col_m5.metric("Exact Match", em_str, help="Subset Accuracy: % de parágrafos com acerto em todas as 5 classes.")

    hl_str = f"{eval_report.hamming_loss:.4f}"
    col_m6.metric("Hamming Loss", hl_str, help="Fração de decisões binárias individuais incorretas.")

    col_meta1, col_meta2 = st.columns(2)
    col_meta1.caption(f"📊 **Cardinalidade de Rótulos ($LC$):** `{eval_report.label_cardinality:.4f}` rótulos/parágrafo")
    col_meta2.caption(f"📈 **Densidade de Rótulos ($LD$):** `{eval_report.label_density:.4f}`")

    with st.expander("📋 Tabela Estruturada de Métricas Globais", expanded=False):
        st.dataframe(df_g, use_container_width=True)

    st.divider()

    # 3. Tabela Principal de Desempenho das Classes Ativas (1 a 5)
    st.markdown("#### 🏷️ Desempenho Detalhado por Dimensão Conceitual (Classes Ativas 1 a 5)")
    st.caption("Tabela com limiares calibrados, prevalência, suporte, matriz de confusão e métricas discriminativas.")
    st.dataframe(df_c, use_container_width=True)

    # 4. Seção Separada: Diagnóstico da Classe 0 (Não Relevante)
    st.divider()
    st.markdown("#### 🚫 Diagnóstico Isolado da Classe 0 (Não Relevante — Derivada)")
    st.caption("A Classe 0 é obtida por exclusão lógica mútua (quando nenhuma das classes ativas 1 a 5 é atribuída).")
    if df_c0 is not None and not df_c0.empty:
        st.dataframe(df_c0, use_container_width=True)
    if eval_report.class_0_metrics and eval_report.class_0_metrics.support_positive == 0:
        st.info(f"ℹ️ **Nota Metodológica:** {eval_report.class_0_metrics.note}")

    # 5. Tabela de Validação Cruzada (se houver)
    if df_cv is not None and not df_cv.empty:
        st.divider()
        st.markdown("#### 🔄 Variação entre Folds da Validação Cruzada (K-Fold CV)")
        st.dataframe(df_cv, use_container_width=True)

    st.divider()

    # 6. Gráficos Comparativos (Plotly)
    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        df_c_plot = df_c.copy()
        try:
            df_c_plot["Precisão"] = pd.to_numeric(df_c_plot["Precisão"], errors="coerce")
            df_c_plot["Recall"] = pd.to_numeric(df_c_plot["Recall"], errors="coerce")
            df_c_plot["F1"] = pd.to_numeric(df_c_plot["F1"], errors="coerce")

            fig_bar = px.bar(
                df_c_plot,
                x="Classe",
                y=["Precisão", "Recall", "F1"],
                barmode="group",
                hover_data=["Dimensão", "Suporte positivo", "TP", "FP", "FN", "TN"],
                title="Comparativo de Precisão, Recall e F1 por Classe",
                labels={"value": "Score (0 a 1)", "variable": "Métrica"}
            )
            fig_bar.update_layout(yaxis=dict(range=[0, 1.05]), legend_title_text="Métrica")
            st.plotly_chart(fig_bar, use_container_width=True)
        except Exception:
            pass

    with col_plot2:
        try:
            fig_sup = px.bar(
                df_c,
                x="Classe",
                y=["TP", "FP", "FN"],
                barmode="group",
                hover_data=["Dimensão", "Suporte positivo"],
                title="Distribuição de Contagens (TP, FP, FN) por Dimensão",
                labels={"value": "Quantidade de Parágrafos", "variable": "Decisão"}
            )
            fig_sup.update_layout(legend_title_text="Contagem")
            st.plotly_chart(fig_sup, use_container_width=True)
        except Exception:
            pass

    st.divider()

    # 7. Matrizes de Confusão Binárias 2x2 por Classe
    st.markdown("#### 🔲 Matrizes de Confusão Binárias 2×2 por Dimensão Conceitual")
    matrix_classes = [c for c in MULTILABEL_CLASSES if CONCEPT_LABEL_SHORT_NAMES[c] in eval_report.per_class_metrics]
    for i in range(0, len(matrix_classes), 2):
        cols_cm = st.columns(2)
        for j in range(2):
            if i + j < len(matrix_classes):
                c_idx = matrix_classes[i + j]
                c_s = CONCEPT_LABEL_SHORT_NAMES[c_idx]
                m_info = eval_report.per_class_metrics[c_s]
                tp, fp, fn, tn = m_info.true_positives, m_info.false_positives, m_info.false_negatives, m_info.true_negatives
                total_k = tp + fp + fn + tn
                acc_k = (tp + tn) / max(1, total_k)
                fpr_k = (fp / max(1, fp + tn)) if (fp + tn) > 0 else 0.0
                fnr_k = (fn / max(1, fn + tp)) if (fn + tp) > 0 else 0.0

                with cols_cm[j]:
                    st.markdown(f"**Classe {c_idx} — {CONCEPT_LABEL_NAMES[c_idx]}**")
                    df_cm_display = pd.DataFrame(
                        [
                            {"Predição": "Previsto Positivo (+)", "Real Positivo (1)": f"TP = {tp}", "Real Negativo (0)": f"FP = {fp}"},
                            {"Predição": "Previsto Negativo (-)", "Real Positivo (1)": f"FN = {fn}", "Real Negativo (0)": f"TN = {tn}"}
                        ]
                    )
                    st.dataframe(df_cm_display, use_container_width=True, hide_index=True)
                    st.caption(f"✓ Acurácia Binária: `{acc_k:.2%}` | Taxa FP (FPR): `{fpr_k:.2%}` | Taxa FN (FNR): `{fnr_k:.2%}` | Suporte Positivo: `{m_info.support_positive}`")
                    st.divider()

    # 8. Persistência Física no Disco e Opções de Download
    saved_paths = export_evaluation_report_to_disk(project, eval_report, run_id)

    st.success(
        f"💾 **Relatórios de Avaliação e Matrizes Salvos Fisicamente no Disco!**\n\n"
        f"Os arquivos de métricas foram gravados na pasta de saída configurada do projeto:\n\n"
        f"- 🎯 **Métricas Globais (CSV):** `{project.classification_dir / 'metricas_avaliacao_globais.csv'}`\n"
        f"- 🏷️ **Métricas por Classe (CSV):** `{project.classification_dir / 'metricas_avaliacao_classes.csv'}`\n"
        f"- 🚫 **Diagnóstico da Classe 0 (CSV):** `{project.classification_dir / 'diagnostico_classe_0.csv'}`\n"
        f"- 📝 **Relatório Completo em Markdown:** `{project.classification_dir / 'relatorio_avaliacao.md'}`\n\n"
        f"📍 **Pasta Local dos Arquivos:** `{project.classification_dir.resolve()}`"
    )

    col_dl_ev1, col_dl_ev2, col_dl_ev3, col_dl_ev4 = st.columns(4)
    p_glob = project.classification_dir / 'metricas_avaliacao_globais.csv'
    col_dl_ev1.download_button(
        label="📥 Métricas Globais (.csv)",
        data=p_glob.read_bytes() if p_glob.exists() else df_g.to_csv(index=False).encode("utf-8"),
        file_name=f"metricas_avaliacao_globais_{run_id}.csv",
        mime="text/csv",
        key=f"dl_eval_glob_{key_suffix}",
        use_container_width=True
    )
    p_cls = project.classification_dir / 'metricas_avaliacao_classes.csv'
    col_dl_ev2.download_button(
        label="📥 Métricas por Classe (.csv)",
        data=p_cls.read_bytes() if p_cls.exists() else df_c.to_csv(index=False).encode("utf-8"),
        file_name=f"metricas_avaliacao_classes_{run_id}.csv",
        mime="text/csv",
        key=f"dl_eval_cls_{key_suffix}",
        use_container_width=True
    )
    p_c0 = project.classification_dir / 'diagnostico_classe_0.csv'
    col_dl_ev3.download_button(
        label="📥 Diagnóstico Classe 0 (.csv)",
        data=p_c0.read_bytes() if p_c0.exists() else (df_c0.to_csv(index=False).encode("utf-8") if df_c0 is not None and not df_c0.empty else b""),
        file_name=f"diagnostico_classe_0_{run_id}.csv",
        mime="text/csv",
        key=f"dl_eval_c0_{key_suffix}",
        use_container_width=True
    )
    p_md = project.classification_dir / 'relatorio_avaliacao.md'
    col_dl_ev4.download_button(
        label="📥 Relatório Completo (.md)",
        data=p_md.read_text(encoding="utf-8") if p_md.exists() else md_text,
        file_name=f"relatorio_avaliacao_{run_id}.md",
        mime="text/markdown",
        key=f"dl_eval_md_{key_suffix}",
        use_container_width=True
    )


def main():
    inject_custom_styles()

    output_dir_path = Path(st.session_state.get("selected_output_dir", DEFAULT_OUTPUT_DIR))
    project = AnalysisProject(output_dir_path)
    st.session_state.analysis_project = project

    # Sincroniza automaticamente na inicialização se ainda não tiver registros carregados
    if not st.session_state.get("project_data_loaded", False):
        res = sync_project_state(output_dir_path)
        if res.get("restored_count", 0) > 0:
            st.session_state.project_data_loaded = True

    with st.sidebar:
        logo_path = BASE_DIR / "logo.png"
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)
            st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

        st.markdown(f"### **Execução:** `{st.session_state.run_id}`")
        st.caption(f"Diretório: `{project.output_dir.name}`")

        n_completed = len(set(st.session_state.completed_steps))
        status_state = "completed" if n_completed == 8 else ("running" if n_completed > 0 else "idle")
        render_status_card(
            title="Status da Pesquisa",
            status_state=status_state,
            details=f"{n_completed} de 8 etapas concluídas no pipeline."
        )

        # Botão principal de Salvar Status da Sessão
        if st.button("💾 Salvar Status da Sessão", type="primary", use_container_width=True, key="btn_save_session_sidebar"):
            snapshot = save_full_session_state(project, st.session_state)
            st.session_state.last_session_save = snapshot.get("saved_at")
            st.toast(f"✓ Sessão salva com sucesso! ({len(snapshot.get('completed_steps', []))} etapas persistidas)")
            st.success("✓ Estado completo da sessão e Gold Standard gravados em disco!")

        if st.session_state.get("last_session_save"):
            st.caption(f"💾 Último salvamento: `{st.session_state.last_session_save[:19].replace('T', ' ')}`")

        if hasattr(st.session_state, "analysis_project") and st.session_state.analysis_project.is_existing_project() and st.session_state.get("project_data_loaded", False):
            stats = st.session_state.analysis_project.get_summary_stats()
            with st.expander("📊 Resumo da Análise Persistente", expanded=True):
                st.caption(f"**ID da Análise:** `{stats['analysis_id']}`")
                st.markdown(f"- **Artigos Registrados:** {stats['total_registered']}")
                st.markdown(f"- **Concluídos:** {stats['completed_articles']} | **Pendentes:** {stats['pending_articles']}")
                st.markdown(f"- **Com Erro:** {stats['error_articles']}")
                st.markdown(f"- **Segmentos Indexados:** {stats['total_segments']:,}".replace(",", "."))
                st.markdown(f"- **Modelo:** `{stats['embedding_model']}`")
                st.markdown(f"- **Status do Índice:** `{stats['index_status']}` (v{stats['index_version']})")
                st.caption(f"Última atualização: {stats['last_updated'][:19].replace('T', ' ')}")

        with st.expander("Parâmetros Metodológicos Gerais", expanded=True):
            emb_model = st.selectbox(
                "Modelo de Embedding",
                options=["nomic-embed-text", "all-MiniLM-L6-v2", "paraphrase-multilingual-mpnet-base-v2"],
                index=0,
                help="Modelo responsável pela transformação dos textos em representações vetoriais semânticas d-dimensionais."
            )
            st.session_state.config["embedding_model"] = emb_model

            sim_th = st.slider(
                "Limiar de Similaridade (θ_s)",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.config.get("similarity_threshold", 0.50)),
                step=0.05,
                help="Define a similaridade mínima necessária (cosseno) para que um parágrafo pertença ao corpus candidato. Valores maiores aumentam a seletividade e podem reduzir o recall."
            )
            st.session_state.config["similarity_threshold"] = sim_th

            max_chars = st.number_input(
                "Max Caracteres/Parágrafo",
                min_value=100,
                max_value=2000,
                value=int(st.session_state.config.get("max_characters", 500)),
                step=50,
                help="Tamanho máximo do bloco de texto do parágrafo para garantir coerência semântica durante a vetorização."
            )
            st.session_state.config["max_characters"] = max_chars

        with st.expander("Configurações de Ambiente e Desempenho", expanded=False):
            if "selected_output_dir" not in st.session_state:
                st.session_state["selected_output_dir"] = str(DEFAULT_OUTPUT_DIR)

            st.text_input(
                "Diretório de Saída (Apenas Seleção):",
                value=st.session_state["selected_output_dir"],
                disabled=True,
                help="Diretório onde os artefatos da execução isolada serão persistidos por run_id."
            )

            if st.button("📁 Buscar Pasta de Saída", use_container_width=True, help="Abre o seletor nativo do sistema operacional para escolher o diretório de saída."):
                chosen_out = open_folder_dialog(st.session_state["selected_output_dir"], title="Selecionar Diretório de Saída dos Artefatos")
                if chosen_out:
                    st.session_state["selected_output_dir"] = chosen_out
                    st.session_state.config["output_dir"] = chosen_out
                    st.session_state.run_dirs = get_run_dir_structure(Path(chosen_out), st.session_state.run_id)
                    st.session_state.project_data_loaded = True
                    sync_project_state(Path(chosen_out))
                    st.toast("✓ Diretório alterado e arquivos sincronizados com sucesso!")
                    st.rerun()

            st.session_state.config["output_dir"] = st.session_state["selected_output_dir"]

        with st.expander("Zona de Manutenção e Opções Avançadas", expanded=False):
            if st.button("🔍 Verificar Integridade da Análise", use_container_width=True):
                rep = project.integrity_checker.run_full_check(st.session_state.config.get("embedding_model"))
                st.session_state.integrity_report = rep
                st.rerun()

            if "integrity_report" in st.session_state:
                rep: IntegrityReport = st.session_state.integrity_report
                st.markdown(f"**Diagnóstico de Integridade:** `{rep.status.upper()}`")
                if rep.integros:
                    st.success("**Íntegros:**\n" + "\n".join(f"- {x}" for x in rep.integros))
                if rep.avisos:
                    st.warning("**Avisos:**\n" + "\n".join(f"- {x}" for x in rep.avisos))
                if rep.inconsistencias_recuperaveis:
                    st.info("**Inconsistências Recuperáveis:**\n" + "\n".join(f"- {x}" for x in rep.inconsistencias_recuperaveis))
                if rep.erros_reconstrucao:
                    st.error("**Erros que Exigem Reconstrução:**\n" + "\n".join(f"- {x}" for x in rep.erros_reconstrucao))

            if st.button("Limpar Cache e Reiniciar Sessão", use_container_width=True):
                st.session_state.clear()
                st.rerun()

    # Header Institucional Principal na Área Central
    st.markdown(
        "<h1 style='margin-bottom: 0px; font-weight: 700; color: #0f172a;'>SLD — Scientific Literature Decoder</h1>"
        "<p style='margin-top: 0px; color: #475569; font-size: 1.05rem; font-weight: 500;'>"
        "Análise semântica, classificação supervisionada e extração conceitual de literatura científica"
        "</p>",
        unsafe_allow_html=True
    )

    # Solicitação explícita ao usuário ao detectar dados no diretório de saída
    if project.is_existing_project() and not st.session_state.get("project_data_loaded", False):
        st.info("📁 **Análise Existente Detectada:** A pasta selecionada contém uma análise gravada anteriormente. Escolha como deseja prosseguir:")
        col_load1, col_load2 = st.columns([1, 1])
        if col_load1.button("📥 Carregar Dados Existentes desta Pasta", use_container_width=True, type="primary"):
            sync_project_state(output_dir_path)
            st.session_state.project_data_loaded = True
            st.toast("✓ Dados da análise carregados com sucesso!")
            st.rerun()
        if col_load2.button("✨ Iniciar Nova Análise Limpa nesta Pasta", use_container_width=True):
            meta = project.initialize_new_project(st.session_state.config)
            st.session_state.analysis_metadata = meta
            st.session_state.project_data_loaded = True
            st.toast("✓ Nova análise inicializada sem carregar dados anteriores.")
            st.rerun()

    # Alerta de migração para pastas legadas
    if st.session_state.get("legacy_folder_detected", False):
        st.warning("⚠️ **Análise Legada Detectada:** Esta pasta foi criada por uma versão anterior do SLD. Deseja migrar os dados para a estrutura persistente?")
        col_m1, col_m2 = st.columns([1, 4])
        if col_m1.button("Confirmar Migração de Pasta"):
            meta = project.migrate_legacy_folder(st.session_state.config)
            st.session_state.legacy_folder_detected = False
            st.session_state.analysis_metadata = meta
            st.toast("✓ Migração de pasta legada realizada com sucesso!")
            st.rerun()

    # Alerta de operação ativa/interrompida (Checkpoints)
    active_chk = project.checkpoint_mgr.get_active_checkpoint()
    if active_chk and active_chk.status in ["active", "interrupted"]:
        st.info(
            f"⚠️ **Operação Interrompida Encontrada:** `{active_chk.operation_type}` | "
            f"Progresso salvo: **{active_chk.completed_items}** de **{active_chk.total_items}** itens | "
            f"Última atualização: `{active_chk.last_updated_at[:19].replace('T', ' ')}`"
        )
        col_c1, col_c2, col_c3 = st.columns([2, 2, 2])
        if col_c1.button("▶️ Retomar do Último Checkpoint"):
            st.session_state.resume_checkpoint_active = active_chk
            st.toast("✓ Retomada ativada a partir do último checkpoint.")
            st.rerun()
        if col_c2.button("🔄 Reiniciar Apenas Esta Etapa"):
            project.checkpoint_mgr.mark_cancelled(active_chk)
            st.toast("✓ Operação anterior descartada. Pronto para reiniciar.")
            st.rerun()
        if col_c3.button("❌ Cancelar Operação Pendente"):
            project.checkpoint_mgr.mark_cancelled(active_chk)
            st.toast("✓ Operação cancelada.")
            st.rerun()
    st.divider()

    # Abas Metodológicas do Pipeline Científico
    t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs([
        "1. ETL — Extração, Transformação e Carregamento",
        "2. Embeddings",
        "3. Análise Exploratória",
        "4. Similaridade Semântica",
        "5. Treinamento Supervisionado",
        "6. Classificação Conceitual",
        "7. Índice de Recuperação do Corpus Refinado",
        "8. Corpus Final e LLM",
        "9. Relatório Metodológico"
    ])

    # ==========================================
    # ABA 1: ETL — EXTRAÇÃO, TRANSFORMAÇÃO E CARREGAMENTO
    # ==========================================
    with t1:
        render_methodology_header(
            title="1. ETL — Extração, Transformação e Carregamento",
            description=(
                "**Extração:** refere-se à coleta das informações contidas nos arquivos PDF e ZIP de origem.\n\n"
                "**Transformação:** é o processo de limpeza, extração de texto estruturado, conversão para Markdown (.md), remoção de referências bibliográficas e segmentação em parágrafos.\n\n"
                "**Carregamento:** a etapa final da construção do corpus se concentra no carregamento dos arquivos já convertidos para Markdown em um repositório local, formando uma nova amostra com o conteúdo que será processado nas etapas seguintes. Além disso, a etapa de carregamento incorpora a função de versionar e auditar a amostra, assegurando que não ocorram duplicações nem perda de arquivos, onde o sucesso da operação é mensurado pela fórmula:"
            ),
            objective="Construir o corpus inicial em Markdown (.md) com auditoria de integridade, versionamento local e medição de redução de tamanho de armazenamento.",
            method="Varredura recursiva de diretórios, extração segura de arquivos ZIP com checagem de ZIP-Slip, conversão via PyMuPDF, remoção heurística de referências e salvamento no repositório de corpus local.",
            formula_latex=r"T_{\text{sucesso}} = \frac{N_{\text{processados}}}{N_{\text{válidos}}} \times 100 \quad \land \quad \text{Redução}_{\text{Tamanho}}(\%) = \left(1 - \frac{\text{Tamanho}_{\text{MD}}}{\text{Tamanho}_{\text{Inicial}}}\right) \times 100",
            legend_dict={
                "N_{processados}": "número de arquivos PDF extraídos e convertidos com sucesso para Markdown",
                "N_{válidos}": "quantidade total de arquivos PDF válidos identificados no diretório",
                "T_{sucesso}": "taxa percentual de sucesso na ingestão e conversão do lote",
                "Tamanho_{Inicial}": "tamanho total acumulado dos arquivos de origem (PDFs/ZIPs em bytes)",
                "Tamanho_{MD}": "tamanho do corpus final em arquivos Markdown gerados (bytes)",
                "Redução_{Tamanho}": "percentual de redução no volume de dados armazenado"
            },
            interpretation="Mede a taxa de aproveitamento, a integridade da conversão e a eficiência no volume de dados armazenado no repositório local de corpus."
        )

        render_stage_disk_loader(1, "ETL — Ingestão e Markdowns", project, st.session_state)

        if "selected_input_dir" not in st.session_state:
            st.session_state["selected_input_dir"] = ""

        col_dir_in, col_dir_btn = st.columns([3, 1])
        with col_dir_in:
            st.text_input(
                "Diretório de Origem dos PDFs ou ZIPs (Apenas Seleção):",
                value=st.session_state["selected_input_dir"],
                disabled=True,
                help="Caminho absoluto do diretório contendo os artigos científicos. Clique no botão ao lado para alterar."
            )
            input_dir_path = st.session_state["selected_input_dir"]

        with col_dir_btn:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            if st.button("📁 Buscar Pasta de Origem", use_container_width=True, help="Abre a janela nativa do sistema operacional (macOS Finder) para escolher a pasta."):
                chosen_dir = open_folder_dialog(st.session_state["selected_input_dir"], title="Selecionar Diretório de Origem dos PDFs ou ZIPs")
                if chosen_dir:
                    st.session_state["selected_input_dir"] = chosen_dir
                    st.rerun()

        # Inspeciona diretório de origem de forma não mutativa (sem extrair ZIPs no layout)
        input_p = Path(input_dir_path) if input_dir_path else None
        if input_p and input_p.exists() and input_p != BASE_DIR:
            direct_pdfs, zip_files = find_all_inputs(input_p)
            if direct_pdfs or zip_files:
                dup_summary = project.registry.analyze_batch(list(direct_pdfs))

                st.markdown("### 📋 Resumo Pré-Ingestão e Controle de Duplicidades")
                c_d1, c_d2, c_d3, c_d4, c_d5 = st.columns(5)
                c_d1.metric("PDFs Diretos", f"{len(direct_pdfs)}")
                c_d2.metric("Arquivos ZIP", f"{len(zip_files)}")
                c_d3.metric("Novos Conteúdos", f"{dup_summary.new_files}")
                c_d4.metric("Duplicados (SHA-256)", f"{dup_summary.duplicate_content}")
                c_d5.metric("Interrompidos / Erros", f"{dup_summary.previously_interrupted + dup_summary.previous_error}")

                dup_policy = st.selectbox(
                    "Política de Tratamento de Duplicados:",
                    options=["pular", "sobrescrever", "versao"],
                    format_func=lambda x: {
                        "pular": "Pular duplicados por SHA-256 (Recomendado — Economiza tempo e evita reprocessamento)",
                        "sobrescrever": "Sobrescrever duplicados (Reprocessa o arquivo e atualiza registros)",
                        "versao": "Criar nova versão (Registra nova versão v2 para mesmo nome com conteúdo distinto)"
                    }[x],
                    index=0,
                    help="Define a ação a tomar para arquivos cujo hash SHA-256 já foi processado."
                )
                st.session_state.config["duplicate_policy"] = dup_policy

        if st.button("Iniciar Ingestão e Extração de Documentos", type="primary"):
            start_t = time.time()
            if not input_p or not input_p.exists():
                st.error("Selecione um diretório de origem válido contendo arquivos PDF ou ZIP.")
            else:
                if not project.is_existing_project():
                    project.initialize_new_project(st.session_state.config)

                interim_dir = project.output_dir / "interim"
                direct_pdfs, zip_files = find_all_inputs(input_p)
                found_pdfs = list(direct_pdfs)
                for z_file in zip_files:
                    extracted_recs = extract_zip_safely(z_file, interim_dir / "extracted_zips")
                    for rec in extracted_recs:
                        found_pdfs.append(rec["extracted_path"])
                unique_input_pdfs = sorted(list(set(found_pdfs)))

                chk = project.checkpoint_mgr.create_checkpoint(
                    operation_type="ingestion",
                    total_items=len(unique_input_pdfs),
                    all_item_keys=[str(p) for p in unique_input_pdfs],
                    config=st.session_state.config
                )

                tracker = ProgressTracker(
                    title="Convertendo PDFs para Arquivos .MD Individuais",
                    total=len(unique_input_pdfs),
                    steps=["Filtrar PDFs & Hashes", "Remover Referências Bibliográficas", "Gravar Arquivos .MD Persistentes"],
                    update_interval=1
                )
                tracker.set_step(0, "Filtrar PDFs & Hashes")

                processed_dir = project.markdown_dir
                ensure_directory(processed_dir)

                articles_records = list(st.session_state.articles_records)
                success_cnt = 0
                skipped_cnt = 0
                error_cnt = 0

                for idx, pdf_path in enumerate(unique_input_pdfs, start=1):
                    if tracker.is_cancelled():
                        project.checkpoint_mgr.mark_interrupted(chk)
                        st.warning("⚠️ Operação interrompida pelo usuário. Progresso preservado no último checkpoint!")
                        break

                    article_id = pdf_path.stem
                    pdf_sha256 = calculate_file_sha256(pdf_path)
                    existing_rec = project.registry.get_by_sha256(pdf_sha256)

                    if existing_rec and existing_rec.processing_status == "completed" and dup_policy == "pular":
                        skipped_cnt += 1
                        project.checkpoint_mgr.update_item_success(chk, str(pdf_path), current_item_name=f"[PULADO] {pdf_path.name}")
                        tracker.update(processed=idx, current_item=f"[PULADO] {pdf_path.name}", successes=success_cnt, skipped=skipped_cnt, errors=error_cnt)
                        continue

                    tracker.set_step(1, "Remover Referências Bibliográficas")
                    try:
                        doc_metadata, full_text, pages_data = read_pdf_content(pdf_path)
                        clean_pages, ref_removed = remove_references(pages_data)

                        meta = ArticleMetadata(
                            sld_id=article_id,
                            title=doc_metadata.get("title", article_id),
                            authors=doc_metadata.get("authors", []),
                            source_pdf=pdf_path.name,
                            source_path=str(pdf_path),
                            pdf_sha256=pdf_sha256,
                            references_removed=ref_removed,
                            processed_at=datetime.now().isoformat()
                        )

                        tracker.set_step(2, "Gravar Arquivos .MD Persistentes")
                        target_md_path, written = write_markdown_file(
                            metadata=meta,
                            pages_data=clean_pages if isinstance(clean_pages, list) else [{"page": 1, "text": str(clean_pages)}],
                            output_dir=processed_dir,
                            overwrite_policy="overwrite" if dup_policy == "sobrescrever" else "skip"
                        )

                        with open(target_md_path, "r", encoding="utf-8") as md_f:
                            md_content = md_f.read()

                        proc_art = ProcessedArticle(
                            metadata=meta,
                            markdown_content=md_content,
                            markdown_path=str(target_md_path),
                            page_count=len(clean_pages) if isinstance(clean_pages, list) else 1,
                            char_count=len(md_content),
                            status="success",
                            processing_duration=0.0
                        )

                        version_num = 1
                        supersedes_id = None
                        if existing_rec and dup_policy == "versao":
                            version_num = existing_rec.version_number + 1
                            supersedes_id = existing_rec.article_id

                        reg_item = ArticleRegistryRecord(
                            article_id=article_id,
                            source_filename=pdf_path.name,
                            source_path=str(pdf_path),
                            pdf_sha256=pdf_sha256,
                            file_size=pdf_path.stat().st_size if pdf_path.exists() else 0,
                            markdown_path=str(target_md_path),
                            processing_status="completed",
                            extraction_status="success",
                            reference_removal_status="success" if ref_removed else "skipped",
                            page_count=len(clean_pages) if isinstance(clean_pages, list) else 1,
                            character_count=len(md_content),
                            version_number=version_num,
                            supersedes=supersedes_id
                        )
                        project.registry.upsert_record(reg_item)

                        articles_records.append(proc_art)
                        success_cnt += 1
                        project.checkpoint_mgr.update_item_success(chk, str(pdf_path), current_item_name=article_id)

                    except Exception as e:
                        error_cnt += 1
                        err_msg = str(e)
                        reg_item = ArticleRegistryRecord(
                            article_id=article_id,
                            source_filename=pdf_path.name,
                            source_path=str(pdf_path),
                            pdf_sha256=pdf_sha256,
                            processing_status="error",
                            error_message=err_msg
                        )
                        project.registry.upsert_record(reg_item)
                        project.checkpoint_mgr.update_item_error(chk, str(pdf_path), error_msg=err_msg, current_item_name=article_id)

                    tracker.update(processed=idx, current_item=article_id, successes=success_cnt, skipped=skipped_cnt, errors=error_cnt)

                dur = time.time() - start_t
                st.session_state.articles_records = articles_records

                if not tracker.is_cancelled():
                    project.checkpoint_mgr.mark_completed(chk)
                    project.update_status("ready")

                    raw_input_bytes = sum(p.stat().st_size for p in unique_input_pdfs if p.exists())
                    md_output_bytes = sum(p.char_count for p in articles_records)
                    size_reduction_pct = (1.0 - (md_output_bytes / raw_input_bytes)) * 100.0 if raw_input_bytes > 0 else 0.0

                    st.session_state.funnel_counts["1_Ingestao"] = {
                        "n_in": len(unique_input_pdfs),
                        "n_out": len(articles_records),
                        "duration_sec": dur,
                        "n_pdfs": len(unique_input_pdfs),
                        "raw_input_bytes": raw_input_bytes,
                        "md_output_bytes": md_output_bytes,
                        "size_reduction_pct": size_reduction_pct
                    }

                    if 1 not in st.session_state.completed_steps:
                        st.session_state.completed_steps.append(1)

                    save_full_session_state(project, st.session_state)
                    import gc; gc.collect()

                    tracker.complete(message=f"ETL concluído: {success_cnt} convertidos, {skipped_cnt} pulados, {error_cnt} erros.")
                    st.toast(f"✓ ETL concluído com sucesso!")
                    st.balloons()
                    st.rerun()

        if "1_Ingestao" in st.session_state.funnel_counts and st.session_state.articles_records:
            fn = st.session_state.funnel_counts["1_Ingestao"]
            raw_b = fn.get("raw_input_bytes", 0)
            md_b = fn.get("md_output_bytes", 0)
            red_pct = fn.get("size_reduction_pct", 0.0)

            def fmt_b(n_bytes: float) -> str:
                if n_bytes >= 1024**3:
                    return f"{n_bytes / (1024**3):.2f} GB"
                elif n_bytes >= 1024**2:
                    return f"{n_bytes / (1024**2):.2f} MB"
                elif n_bytes >= 1024:
                    return f"{n_bytes / 1024:.2f} KB"
                else:
                    return f"{n_bytes:.0f} B"

            render_pipeline_metrics(
                input_count=fn["n_in"],
                output_count=fn["n_out"],
                duration_sec=fn["duration_sec"],
                n_docs=fn.get("n_pdfs"),
                n_words=sum(len(art.markdown_content.split()) for art in st.session_state.articles_records),
                unit_label="arquivos .md gerados"
            )
            render_completion_panel(
                title="Processo ETL e Carregamento Concluídos",
                metrics={
                    "Documentos PDFs Ingestados": fn["n_in"],
                    "Arquivos .MD Gerados": len(st.session_state.articles_records),
                    "Tamanho Amostra Inicial (PDFs)": fmt_b(raw_b),
                    "Tamanho Amostra Final (Markdown)": fmt_b(md_b),
                    "Redução de Tamanho Armazenado": f"{red_pct:.2f}%",
                    "Tempo Total": f"{fn['duration_sec']:.2f}s"
                }
            )

            # Opção de Download dos Arquivos .MD Individuais da Etapa 1
            md_t1 = f"# Coleção de Arquivos Markdown Individuais — Run `{st.session_state.run_id}`\n\n" + "\n\n---\n\n".join(
                f"## Artigo `{art.metadata.sld_id}` (`{art.metadata.source_pdf}`)\n- **Caminho:** `{art.markdown_path}`\n- **Caracteres:** {art.char_count}\n\n```markdown\n{art.markdown_content[:500]}...\n```"
                for art in st.session_state.articles_records
            )
            df_t1 = pd.DataFrame([{
                "article_id": art.metadata.sld_id,
                "source_pdf": art.metadata.source_pdf,
                "markdown_path": art.markdown_path,
                "char_count": art.char_count,
                "references_removed": art.metadata.references_removed
            } for art in st.session_state.articles_records])
            jsonl_t1 = "\n".join(json.dumps({
                "sld_id": art.metadata.sld_id,
                "source_pdf": art.metadata.source_pdf,
                "markdown_path": art.markdown_path,
                "char_count": art.char_count,
                "content_preview": art.markdown_content[:200]
            }, ensure_ascii=False) for art in st.session_state.articles_records)

            render_export_section("1_ETL_Arquivos_Markdown", st.session_state.run_id, md_t1, df_t1, jsonl_t1, file_prefix="etl_arquivos_md")
        else:
            render_empty_state(
                title="Nenhum Documento Ingestado",
                description="Selecione o diretório contendo arquivos PDF/ZIP e clique em 'Iniciar Ingestão' para converter os artigos em arquivos .md individuais.",
                recommendation="Aguardando execução da Etapa 1."
            )

    # ==========================================
    # ABA 2: SENTENCE EMBEDDINGS VETORIAIS E SEGMENTAÇÃO
    # ==========================================
    with t2:
        render_methodology_header(
            title="2. Segmentação em Parágrafos e Embeddings Vetoriais",
            description=(
                "Esta etapa lê os arquivos .md individuais gerados no ETL (Etapa 1), fragmenta cada artigo em parágrafos conceituais "
                "e calcula os Sentence Embeddings vetoriais por meio de um modelo Transformer (ex: nomic-embed-text)."
            ),
            objective="Fragmentar os artigos em parágrafos e mapear o conteúdo semântico para um espaço vetorial contínuo denso.",
            method="Segmentação Markdown concisa por parágrafos (max 500 caracteres) seguida de codificação vetorial densa com normalização L2.",
            formula_latex=r"P_{i,j} = \text{Segment}(MD_i) \quad \land \quad E(P_{i,j}) \in \mathbb{R}^{d}, \quad \|E(P_{i,j})\|_2 = 1",
            legend_dict={
                "MD_i": "arquivo Markdown individual do i-ésimo artigo",
                "P_{i,j}": "j-ésimo parágrafo extraído do artigo i",
                "E(.)": "função de transformação do Sentence Transformer",
                "d": "dimensionalidade do vetor numérico (ex: 768 no nomic-embed-text ou 384 no all-MiniLM-L6-v2)",
                "R^d": "espaço vetorial real de dimensão d",
                "||.||_2": "norma Euclidiana L2 unitária"
            },
            interpretation="A segmentação isola as unidades fundamentais de análise e a vetorização permite comparar parágrafos por proximidade semântica."
        )

        render_stage_disk_loader(2, "Sentence Embeddings e Vetorização", project, st.session_state)

        articles = st.session_state.articles_records
        processed_dir = st.session_state.run_dirs.get("processed", st.session_state.run_dirs["root"] / "processed")

        if not articles and processed_dir.exists():
            md_files = list(processed_dir.glob("*.md"))
            if md_files:
                articles = []
                for mf in md_files:
                    with open(mf, "r", encoding="utf-8") as f:
                        txt = f.read()
                    articles.append(ProcessedArticle(
                        metadata=ArticleMetadata(sld_id=mf.stem, source_path=str(mf)),
                        markdown_content=txt,
                        markdown_path=str(mf),
                        page_count=1,
                        char_count=len(txt),
                        status="success"
                    ))
                st.session_state.articles_records = articles

        n_articles = len(articles)

        if n_articles == 0:
            render_empty_state(
                title="Nenhum Arquivo Markdown Disponível para Segmentação e Vetorização",
                description="Conclua a etapa 1 (ETL) para gerar os arquivos .md individuais dos artigos antes de calcular os embeddings.",
                recommendation="Aguardando conclusão da Etapa 1 (ETL)."
            )
        else:
            tracker_mgr = EmbeddingsTracker(project.index_dir)
            total_recs, comp_recs = tracker_mgr.get_summary_counts()

            st.info(f"Artigos em Markdown disponíveis: **{n_articles:,}** | Arquivos de embeddings (.npy) já gerados: **{comp_recs:,} de {n_articles:,}**".replace(",", "."))

            if st.button("Segmentar Parágrafos e Calcular Embeddings Vetoriais Por Artigo", type="primary"):
                start_t = time.time()
                emb_service = EmbeddingService(model_name=st.session_state.config["embedding_model"])
                vec_index = VectorIndex(project.index_dir)

                if vec_index.metadata_path.exists():
                    try:
                        with open(vec_index.metadata_path, "r", encoding="utf-8") as f:
                            old_meta = json.load(f)
                        old_model = old_meta.get("embedding_model")
                        if old_model and old_model != emb_service.model_name:
                            st.warning(f"⚠️ **Incompatibilidade de Modelo:** O índice atual usa `{old_model}`, mas você selecionou `{emb_service.model_name}`. O índice será reconstruído.")
                            vec_index.clear()
                    except Exception:
                        pass

                chk_emb = project.checkpoint_mgr.create_checkpoint(
                    operation_type="embeddings",
                    total_items=len(articles),
                    all_item_keys=[art.metadata.sld_id for art in articles],
                    config=st.session_state.config
                )

                tracker = ProgressTracker(
                    title="Vetorizando Parágrafos e Gerando Arquivos .NPY por Artigo",
                    total=len(articles),
                    steps=["Checar Arquivos .NPY Existentes", "Segmentar & Vetorizar Artigo", "Atualizar Registro embeddings_tracker.md"],
                    update_interval=1
                )

                all_corpus_records = []
                all_embeddings_list = []
                all_segments_to_save = []
                generated_files_count = comp_recs
                skipped_files_count = 0

                for idx_a, art in enumerate(articles, start=1):
                    if tracker.is_cancelled():
                        project.checkpoint_mgr.mark_interrupted(chk_emb)
                        st.warning("⚠️ Operação interrompida pelo usuário. Progresso preservado em `embeddings_tracker.md`!")
                        break

                    article_id = art.metadata.sld_id

                    paras = segment_markdown_paragraphs(
                        markdown_content=art.markdown_content,
                        article_id=article_id,
                        doc_id=article_id,
                        max_characters=st.session_state.config.get("max_characters", 500)
                    )
                    for p in paras:
                        p.status = "INGESTED"

                    if tracker_mgr.has_article_embedding(article_id):
                        skipped_files_count += 1
                        art_emb = tracker_mgr.load_article_embedding(article_id)
                        if art_emb is not None and len(art_emb) == len(paras):
                            for idx_p, p in enumerate(paras):
                                p.embedding = art_emb[idx_p]
                            all_embeddings_list.append(art_emb)
                            all_corpus_records.extend(paras)
                            for p in paras:
                                all_segments_to_save.append(Segment.from_dict({
                                    "segment_id": p.paragraph_id, "article_id": p.article_id, "paragraph_id": p.paragraph_id,
                                    "source_pdf": getattr(art.metadata, "source_pdf", f"{article_id}.pdf"),
                                    "markdown_path": art.markdown_path, "text": p.text, "text_sha256": calculate_text_sha256(p.text)
                                }))

                        project.checkpoint_mgr.update_item_success(chk_emb, article_id, current_item_name=f"[PULADO .NPY] {article_id}")
                        tracker.update(
                            processed=idx_a,
                            current_item=f"Gerados: {generated_files_count}/{len(articles)} arquivos .npy | Pulado: {article_id}",
                            successes=generated_files_count,
                            skipped=skipped_files_count
                        )
                        continue

                    tracker.set_step(1, f"Gerar Embeddings Vetoriais ({article_id})")
                    texts = [p.text for p in paras]
                    if texts:
                        art_emb = emb_service.get_embeddings(texts)
                        for idx_p, p in enumerate(paras):
                            p.embedding = art_emb[idx_p]

                        tracker.set_step(2, "Atualizar Registro embeddings_tracker.md")
                        tracker_mgr.save_article_embedding(
                            article_id=article_id,
                            source_pdf=art.metadata.source_pdf,
                            embeddings=art_emb,
                            model_name=emb_service.model_name,
                            paragraph_count=len(paras)
                        )

                        all_embeddings_list.append(art_emb)
                        all_corpus_records.extend(paras)
                        for p in paras:
                            all_segments_to_save.append(Segment.from_dict({
                                "segment_id": p.paragraph_id, "article_id": p.article_id, "paragraph_id": p.paragraph_id,
                                "source_pdf": getattr(art.metadata, "source_pdf", f"{article_id}.pdf"),
                                "markdown_path": art.markdown_path, "text": p.text, "text_sha256": calculate_text_sha256(p.text)
                            }))

                        generated_files_count += 1
                        project.checkpoint_mgr.update_item_success(chk_emb, article_id, current_item_name=article_id)

                    tracker.update(
                        processed=idx_a,
                        current_item=f"Gerados: {generated_files_count}/{len(articles)} arquivos .npy | Concluído: {article_id}",
                        successes=generated_files_count,
                        skipped=skipped_files_count
                    )

                if not tracker.is_cancelled() and all_embeddings_list:
                    master_embeddings = np.vstack(all_embeddings_list)
                    vec_index.build_and_save(
                        embeddings=master_embeddings,
                        segments=all_segments_to_save,
                        model_name=emb_service.model_name,
                        device=emb_service.device,
                        config=st.session_state.config
                    )

                    dur = time.time() - start_t
                    st.session_state.corpus_records = all_corpus_records
                    st.session_state.embeddings_matrix = master_embeddings

                    repo = CorpusRepository(project.index_dir)
                    repo.save_corpus_records(all_corpus_records)

                    meta = project.load_metadata()
                    meta.segment_count = len(all_corpus_records)
                    meta.status = "ready"
                    project.save_metadata(meta)
                    project.checkpoint_mgr.mark_completed(chk_emb)

                    st.session_state.funnel_counts["2_Embeddings"] = {
                        "n_in": len(all_corpus_records),
                        "n_out": master_embeddings.shape[0],
                        "duration_sec": dur,
                        "dimension": master_embeddings.shape[1]
                    }

                    if 2 not in st.session_state.completed_steps:
                        st.session_state.completed_steps.append(2)

                    save_full_session_state(project, st.session_state)
                    import gc; gc.collect()

                    tracker.complete(message=f"Vetorização concluída: {generated_files_count} arquivos .npy gerados e registrados no embeddings_tracker.md.")
                    st.toast(f"✓ Embeddings por artigo (.npy) e arquivo embeddings_tracker.md gerados com sucesso!")
                    st.balloons()
                    st.rerun()

            if st.session_state.embeddings_matrix is not None:
                fn = st.session_state.funnel_counts.get("2_Embeddings", {})
                render_pipeline_metrics(
                    input_count=fn.get("n_in", len(st.session_state.corpus_records)),
                    output_count=fn.get("n_out", st.session_state.embeddings_matrix.shape[0]),
                    duration_sec=fn.get("duration_sec", 0.0),
                    n_docs=len(set(r.article_id for r in st.session_state.corpus_records)),
                    n_words=sum(len(r.text.split()) for r in st.session_state.corpus_records)
                )
                render_completion_panel(
                    title="Segmentação e Vetorização de Embeddings Concluídas",
                    metrics={
                        "Modelo": st.session_state.config["embedding_model"],
                        "Parágrafos Extraídos": len(st.session_state.corpus_records),
                        "Dimensão (d)": st.session_state.embeddings_matrix.shape[1],
                        "Vetores Gerados": f"{st.session_state.embeddings_matrix.shape[0]:,}".replace(",", "."),
                        "Tempo": f"{fn.get('duration_sec', 0.0):.2f}s"
                    }
                )

                # Opção de Download da Etapa 2
                md_t2 = "# Parágrafos Vetorizados (Etapa 2 — Embeddings)\n\n" + "\n\n---\n\n".join(
                    f"## Parágrafo `{r.paragraph_id}` (`{r.article_id}`)\n- **Dimensão Embedding:** `{st.session_state.embeddings_matrix.shape[1]}`\n\n{r.text}" for r in st.session_state.corpus_records
                )
                df_t2 = pd.DataFrame([{"paragraph_id": r.paragraph_id, "article_id": r.article_id, "embedding_dim": st.session_state.embeddings_matrix.shape[1], "text": r.text} for r in st.session_state.corpus_records])
                jsonl_t2 = "\n".join(json.dumps(r.model_dump(), ensure_ascii=False) for r in st.session_state.corpus_records)
                render_export_section("2_Embeddings_Vetoriais", st.session_state.run_id, md_t2, df_t2, jsonl_t2, file_prefix="embeddings")

    # ==========================================
    # ABA 3: ANÁLISE EXPLORATÓRIA E BIBLIOMÉTRICA
    # ==========================================
    with t3:
        render_methodology_header(
            title="3. Análise Exploratória e Bibliométrica do Corpus",
            description=(
                "Esta etapa caracteriza quantitativamente o corpus e permite examinar sua composição documental, distribuição temporal, "
                "frequência lexical e padrões de co-ocorrência antes das etapas de seleção semântica e classificação supervisionada."
            ),
            objective="Descrever quantitativamente e visualmente a estrutura textual e estatística dos artigos científicos sem alterar o corpus.",
            method="Tokenização limpa, contagem de frequências lexicais, estatísticas descritivas e matriz de co-ocorrência de termos.",
            formula_latex=r"C_{ij} = \sum_{d=1}^{D} I(i \in d) I(j \in d)",
            legend_dict={
                "C_{ij}": "frequência de co-ocorrência observada entre os termos i e j",
                "D": "número total de unidades de análise (documento ou parágrafo)",
                "I(.)": "função indicadora que assume 1 se a condição for verdadeira e 0 caso contrário",
                "i, j": "termos lexicais analisados"
            },
            interpretation="Revela os termos centrais da literatura e as interconexões conceituais mais frequentes nos artigos científicos."
        )

        render_stage_disk_loader(3, "Análise Exploratória e Bibliométrica", project, st.session_state)

        if not st.session_state.corpus_records:
            render_empty_state(
                title="Análise Exploratória Aguardando Carregamento do Corpus",
                description="Os parágrafos ainda não estão carregados na memória desta sessão.",
                recommendation="Clique no botão '📥 Carregar Arquivos da Etapa 3' acima para carregar o corpus salvo no diretório de saída."
            )
        else:
            if st.session_state.exploratory_results is None:
                st.info("💡 A análise exploratória calcula descritores estatísticos, nuvem de palavras (WordCloud) e matriz de co-ocorrência sobre o corpus. Clique no botão abaixo para gerar a análise.")
                if st.button("📊 Gerar Análise Exploratória e Bibliométrica", type="primary"):
                    with st.spinner("Processando estatísticas lexicais, nuvem de palavras e matriz de co-ocorrência..."):
                        descriptors = compute_corpus_descriptors(st.session_state.corpus_records)
                        word_counts = [len(r.text.split()) for r in st.session_state.corpus_records]
                        wc_img = generate_wordcloud_image(st.session_state.corpus_records)
                        df_terms = compute_top_terms(st.session_state.corpus_records, top_n=20)
                        matrix_df, G = compute_cooccurrence_matrix(st.session_state.corpus_records, level="paragraph", top_n_terms=20)
                        df_terms_full = compute_top_terms(st.session_state.corpus_records, top_n=50)

                        md_t3 = (
                            f"# Relatório de Análise Exploratória (Etapa 3)\n\n"
                            f"- **Total de Documentos:** {descriptors['n_documents']}\n"
                            f"- **Total de Parágrafos:** {descriptors['n_paragraphs']}\n"
                            f"- **Total de Palavras:** {descriptors['total_words']}\n"
                            f"- **Média de Palavras por Documento:** {descriptors['mean_words_per_doc']:.2f}\n\n"
                            f"## Top 50 Termos Frequentes\n\n" + dataframe_to_markdown(df_terms_full, index=False)
                        )

                        st.session_state.exploratory_results = {
                            "descriptors": descriptors,
                            "word_counts": word_counts,
                            "wc_img": wc_img,
                            "df_terms": df_terms,
                            "matrix_df": matrix_df,
                            "df_terms_full": df_terms_full,
                            "md_t3": md_t3,
                        }

                        if 3 not in st.session_state.completed_steps:
                            st.session_state.completed_steps.append(3)
                        st.rerun()
            else:
                exp_res = st.session_state.exploratory_results
                descriptors = exp_res["descriptors"]

                col_info, col_btn = st.columns([4, 1])
                with col_btn:
                    if st.button("🔄 Recalcular Análise", help="Recalcular análise exploratória sobre os parágrafos atuais."):
                        st.session_state.exploratory_results = None
                        st.rerun()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Documentos", f"{descriptors['n_documents']}")
                c2.metric("Parágrafos", f"{descriptors['n_paragraphs']:,}".replace(",", "."))
                c3.metric("Total Palavras", f"{descriptors['total_words']:,}".replace(",", "."))
                c4.metric("Média Palavras/Doc", f"{descriptors['mean_words_per_doc']:.1f}")

                render_descriptive_statistics(exp_res["word_counts"], title="Estatísticas Descritivas do Comprimento dos Parágrafos (Palavras)")

                col_wc, col_top = st.columns([1, 1])
                with col_wc:
                    st.subheader("Nuvem de Palavras (WordCloud)")
                    if exp_res["wc_img"]:
                        st.image(exp_res["wc_img"], use_container_width=True)
                    render_interpretation_box("Visualização das palavras mais proeminentes no corpus de parágrafos ingestados.")

                with col_top:
                    st.subheader("Top Termos mais Frequentes")
                    fig_terms = px.bar(exp_res["df_terms"], x="Frequência", y="Termo", orientation="h", title="Top 20 Frequência de Termos no Corpus")
                    fig_terms.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig_terms, use_container_width=True)
                    render_interpretation_box("Exibe a contagem bruta dos termos lexicais mais recorrentes na literatura analisada.")

                st.divider()
                st.subheader("Análise da Matriz de Co-ocorrência de Termos (C_ij)")
                level = st.radio("Unidade de Análise da Co-ocorrência", options=["document", "paragraph"], horizontal=True, key="cooccur_level_select")
                col_update, _ = st.columns([2, 3])
                if col_update.button(f"Atualizar Matriz por {level.capitalize()}", key="update_cooccur_btn"):
                    with st.spinner("Atualizando matriz de co-ocorrência..."):
                        m_df, _ = compute_cooccurrence_matrix(st.session_state.corpus_records, level=level, top_n_terms=20)
                        exp_res["matrix_df"] = m_df
                        st.rerun()

                renamed_matrix = exp_res["matrix_df"].copy()
                renamed_matrix.index.name = "Termo A / Termo B"
                st.dataframe(renamed_matrix, use_container_width=True)
                render_interpretation_box(f"Frequência com que cada par de termos aparece simultaneamente na mesma unidade de análise ({level}).")

                render_export_section("3_Analise_Exploratoria", st.session_state.run_id, exp_res["md_t3"], exp_res["df_terms_full"], file_prefix="exploratoria")

    # ==========================================
    # ABA 4: SIMILARIDADE SEMÂNTICA (COSINE SIMILARITY)
    # ==========================================
    with t4:
        render_methodology_header(
            title="4. Recuperação por Similaridade Semântica (Cosine Similarity)",
            description=(
                "Esta etapa realiza a recuperação semântica inicial. Seu objetivo é reduzir o espaço documental, "
                "mantendo parágrafos suficientemente próximos das sentenças-âncora definidas pelo pesquisador."
            ),
            objective="Filtrar parágrafos irrelevantes mantendo alta revocação (High Recall) na seleção candidato.",
            method="Cálculo da similaridade do cosseno entre o vetor das sentenças-âncora e os embeddings dos parágrafos.",
            formula_latex=r"S(P_i, Q) = \frac{E(P_i) \cdot E(Q)}{\|E(P_i)\| \|E(Q)\|}",
            legend_dict={
                "P_i": "parágrafo analisado do corpus",
                "Q": "consulta ou sentença-âncora de referência conceitual",
                "E(P_i)": "embedding vetorial do parágrafo",
                "E(Q)": "embedding vetorial da consulta",
                ".": "produto escalar entre os vetores",
                "||.||": "norma Euclidiana L2",
                "S(P_i, Q)": "score de similaridade do cosseno resultante (S em [-1, 1])"
            },
            interpretation="Valores maiores indicam maior proximidade entre as representações vetoriais. O valor não deve ser interpretado automaticamente como probabilidade de relevância."
        )

        render_formula(
            title="Critério de Seleção por Limiar Semântico (θ_s)",
            latex_formula=r"\text{Selected}_i = I[S(P_i, Q) \ge \theta_s]",
            legend_dict={
                "Selected_i": "indicador binário de seleção do parágrafo i",
                "I[.]": "função indicadora",
                "S(P_i, Q)": "score de similaridade do cosseno calculado",
                "θ_s": "limiar mínimo definido pelo pesquisador para recuperação"
            },
            interpretation="O limiar controla o equilíbrio entre amplitude da recuperação e seletividade. Valores menores tendem a aumentar o recall, enquanto valores maiores tornam a recuperação mais restritiva."
        )

        render_stage_disk_loader(4, "Recuperação por Similaridade Semântica", project, st.session_state)

        st.divider()
        st.subheader("Sentenças-Âncoras da Consulta Semântica")

        with st.form("add_anchors_form", clear_on_submit=True):
            new_anchors_raw = st.text_area(
                "Adicionar frase(s)/definição(ões) âncora (separe múltiplas sentenças por ';'):",
                placeholder="Cole ou digite uma ou mais frases de referência separadas por ponto e vírgula (;)\nExemplo: Vulnerabilidade social e desigualdade populacional; Infraestrutura física frágil e suscetível; Escassez de recursos econômicos e pobreza",
                help="Você pode inserir uma sentença individual ou várias sentenças de uma vez separadas por ponto e vírgula ';'."
            )
            col_sub, col_preset = st.columns([3, 2])
            submit_anchors = col_sub.form_submit_button("➕ Adicionar Sentença(s)-Âncora", type="primary")

        if submit_anchors and new_anchors_raw:
            raw_list = [s.strip() for s in new_anchors_raw.split(";") if s.strip()]
            if raw_list:
                for text in raw_list:
                    st.session_state.reference_set.add_anchor(text)
                st.success(f"{len(raw_list)} sentença(s)-âncora adicionada(s) com sucesso!")
                st.rerun()

        if st.session_state.reference_set.anchors:
            col_h, col_dl, col_clr = st.columns([3, 1.5, 1])
            col_h.markdown(f"**Âncoras Cadastradas ({len(st.session_state.reference_set.anchors)}):**")

            anchors_md = (
                f"# Sentenças-Âncoras da Consulta Semântica\n\n"
                f"- **Data/Hora:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n"
                f"- **Execução ID:** `{st.session_state.run_id}`\n"
                f"- **Total de Âncoras Cadastradas:** `{len(st.session_state.reference_set.anchors)}`\n\n"
                f"---\n\n"
                f"## Lista de Sentenças-Âncoras\n\n" +
                "\n".join(f"- **[{a.id}]**: {a.text}" for a in st.session_state.reference_set.anchors) + "\n"
            )

            col_dl.download_button(
                label="📥 Baixar Âncoras (.md)",
                data=anchors_md,
                file_name=f"sentencas_ancoras_{st.session_state.run_id}.md",
                mime="text/markdown",
                help="Baixar lista de sentenças-âncoras em formato Markdown (.md)."
            )

            if col_clr.button("🗑️ Limpar Todas", key="clear_all_anchors_btn"):
                st.session_state.reference_set.anchors = []
                st.rerun()

            for idx, anc in enumerate(st.session_state.reference_set.anchors):
                col_a, col_btn = st.columns([5, 1])
                col_a.text(f"• [{anc.id}] {anc.text}")
                if col_btn.button("Remover", key=f"del_anc_{idx}"):
                    st.session_state.reference_set.remove_anchor(anc.id)
                    st.rerun()
        else:
            st.info("Nenhuma sentença-âncora cadastrada. Insira uma ou mais frases acima para realizar a busca semântica.")

        st.divider()

        # Cartão de Status da Operação de Busca Semântica
        if "4_Similaridade" in st.session_state.funnel_counts:
            fn_st = st.session_state.funnel_counts["4_Similaridade"]
            render_status_card(
                title="Status da Recuperação Semântica por Cosseno",
                status_state="completed",
                details=f"Concluído: {fn_st['n_out']:,} parágrafos candidatos retidos de {fn_st['n_in']:,} analisados em {fn_st['duration_sec']:.2f}s (Limiar θ_s = {fn_st['threshold']:.2f}).".replace(",", ".")
            )
        else:
            render_status_card(
                title="Status da Recuperação Semântica por Cosseno",
                status_state="idle",
                details="Aguardando execução da busca por cosseno sobre os embeddings do corpus."
            )

        if st.session_state.embeddings_matrix is None:
            render_empty_state(
                title="Busca Semântica Indisponível",
                description="Execute a etapa 2 (Embeddings) para calcular os vetores do corpus antes de realizar a busca por cosseno.",
                recommendation="Aguardando geração de embeddings."
            )
        else:
            col_t4_th, col_t4_btn = st.columns([2, 3])
            with col_t4_th:
                th_input = st.slider(
                    "Limiar de Similaridade Semântica (θ_s):",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.config.get("similarity_threshold", 0.50)),
                    step=0.05,
                    key="t4_threshold_slider",
                    help="Parágrafos com score de similaridade igual ou superior a este limiar serão retidos."
                )
                st.session_state.config["similarity_threshold"] = th_input
            with col_t4_btn:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                run_search_btn = st.button("Executar Busca Semântica por Cosseno", type="primary")

            if run_search_btn:
                start_t = time.time()
                emb_service = EmbeddingService(model_name=st.session_state.config["embedding_model"])
                batch_sz = st.session_state.config.get("semantic_batch_size", DEFAULT_SEMANTIC_BATCH_SIZE)
                th = th_input

                # Prepara VectorIndex temporário em memória para o corpus atual
                v_index = VectorIndex(DEFAULT_INDEX_DIR)
                v_index.embeddings = st.session_state.embeddings_matrix
                v_index.segments = [
                    Segment(
                        segment_id=getattr(r, "segment_id", f"{r.article_id}_{r.paragraph_id}"),
                        article_id=r.article_id,
                        paragraph_id=r.paragraph_id,
                        source_pdf=getattr(r, "source_pdf", "corpus.pdf"),
                        markdown_path=getattr(r, "markdown_path", ""),
                        title=getattr(r, "title", "não identificado"),
                        section=getattr(r, "section", "Geral"),
                        subsection=getattr(r, "subsection", ""),
                        page_start=getattr(r, "page_start", 1),
                        page_end=getattr(r, "page_end", 1),
                        text=r.text,
                        text_sha256=getattr(r, "text_sha256", ""),
                        status="valid_paragraph",
                    )
                    for r in st.session_state.corpus_records
                ]

                tracker = ProgressTracker(
                    title="4. Recuperação por Similaridade Semântica (Cosine Similarity - Batched)",
                    total=len(st.session_state.corpus_records),
                    steps=["Vetorizar Sentenças-Âncoras", "Processar Cosine Similarity em Batches", "Filtrar por Limiar θ_s"],
                    update_interval=1
                )
                tracker.set_step(0, "Vetorizar Sentenças-Âncoras")

                def batch_progress_cb(processed, total, retained, discarded, b_idx, n_batches, b_dur):
                    tracker.set_step(1, f"Processando Batch {b_idx}/{n_batches} ({b_dur:.3f}s/batch)")
                    tracker.update(
                        processed=processed,
                        current_item=f"Lote {b_idx}/{n_batches} | Retidos: {retained:,} | Descartados: {discarded:,}",
                        step_processed=b_idx,
                        step_total=n_batches,
                        successes=retained
                    )

                results, summary = perform_multi_anchor_search(
                    vector_index=v_index,
                    embedding_service=emb_service,
                    reference_set=st.session_state.reference_set,
                    aggregation_strategy=st.session_state.config.get("aggregation_strategy", "maximum"),
                    threshold=th,
                    batch_size=batch_sz,
                    top_k_anchors=st.session_state.config.get("top_k_anchors", 1),
                    progress_callback=batch_progress_cb,
                    return_summary=True,
                    only_retained=False
                )

                res_map = {f"{r.article_id}_{r.paragraph_id}": r.aggregate_score for r in results}
                # Fallback para chaves simples caso necessário
                for r in results:
                    res_map[r.paragraph_id] = r.aggregate_score

                for r in st.session_state.corpus_records:
                    k = f"{r.article_id}_{r.paragraph_id}"
                    r.semantic_score = res_map.get(k, res_map.get(r.paragraph_id, 0.0))

                st.session_state.semantic_scores_map = res_map
                st.session_state.semantic_search_results = results
                candidates = [r for r in st.session_state.corpus_records if (r.semantic_score or 0.0) >= th]
                st.session_state.semantic_candidates = candidates
                dur = time.time() - start_t

                st.session_state.funnel_counts["4_Similaridade"] = {
                    "n_in": len(st.session_state.corpus_records),
                    "n_out": len(candidates),
                    "duration_sec": dur,
                    "threshold": th,
                    "summary": summary.to_dict()
                }

                if 4 not in st.session_state.completed_steps:
                    st.session_state.completed_steps.append(4)

                save_full_session_state(project, st.session_state)
                import gc; gc.collect()

                tracker.complete(message=f"Busca semântica concluída: {len(candidates):,} parágrafos candidatos selecionados com limiar θ_s={th:.2f}.", show_balloons=True)
                st.balloons()
                st.rerun()

            if "4_Similaridade" in st.session_state.funnel_counts:
                fn = st.session_state.funnel_counts["4_Similaridade"]
                candidates = [r for r in st.session_state.corpus_records if (r.semantic_score or 0.0) >= fn["threshold"]]

                render_pipeline_metrics(
                    input_count=fn["n_in"],
                    output_count=fn["n_out"],
                    duration_sec=fn["duration_sec"],
                    n_docs=len(set(r.article_id for r in st.session_state.corpus_records)),
                    unit_label="parágrafos candidatos"
                )
                render_completion_panel(
                    title="Recuperação Semântica Concluída em Batches",
                    metrics={
                        "Entrada": fn["n_in"],
                        "Candidatos Selecionados": fn["n_out"],
                        "Retenção": f"{(fn['n_out']/fn['n_in']*100):.2f}%" if fn["n_in"]>0 else "0.00%",
                        "Threshold (θ_s)": f"{fn['threshold']:.2f}",
                        "Batch Size": st.session_state.config.get("semantic_batch_size", DEFAULT_SEMANTIC_BATCH_SIZE)
                    }
                )
                if "summary" in fn:
                    st.json(fn["summary"])

                scores_np = np.array([r.semantic_score for r in st.session_state.corpus_records if r.semantic_score is not None])
                if len(scores_np) > 0:
                    render_descriptive_statistics(scores_np, title="Estatísticas Descritivas dos Scores de Similaridade Semântica")

                    col_chart1, col_chart2 = st.columns([1, 1])

                    with col_chart1:
                        fig_sim = px.histogram(
                            scores_np,
                            nbins=30,
                            title="Distribuição da Similaridade Semântica (Cosine Similarity)",
                            labels={"value": "Score de Similaridade Cosseno", "count": "Frequência de Parágrafos"}
                        )
                        fig_sim.add_vline(
                            x=st.session_state.config["similarity_threshold"],
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Limiar θ_s"
                        )
                        st.plotly_chart(fig_sim, use_container_width=True)
                        render_interpretation_box("Distribuição dos scores de similaridade obtidos para todos os parágrafos do corpus. A linha vermelha representa o ponto de corte selecionado.")

                    with col_chart2:
                        # Curva de Sensibilidade Limiar θ_s vs. Parágrafos Retidos
                        threshold_grid = np.linspace(0.0, 1.0, 51)
                        retained_counts = [(scores_np >= t).sum() for t in threshold_grid]
                        retained_pcts = [c / len(scores_np) * 100.0 for c in retained_counts]

                        df_thresh_curve = pd.DataFrame({
                            "Limiar (θ_s)": threshold_grid,
                            "Parágrafos Retidos": retained_counts,
                            "Retenção (%)": retained_pcts
                        })

                        fig_curve = px.line(
                            df_thresh_curve,
                            x="Limiar (θ_s)",
                            y="Parágrafos Retidos",
                            hover_data=["Retenção (%)"],
                            title="Sensibilidade: Limiar de Similaridade (θ_s) vs. Parágrafos Retidos",
                            labels={"Limiar (θ_s)": "Limiar de Similaridade (θ_s)", "Parágrafos Retidos": "Parágrafos Retidos"}
                        )
                        curr_th = st.session_state.config["similarity_threshold"]
                        curr_retained = (scores_np >= curr_th).sum()
                        fig_curve.add_vline(
                            x=curr_th,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"θ_s = {curr_th:.2f} ({curr_retained:,} retidos)".replace(",", ".")
                        )
                        fig_curve.update_traces(line_color="#2563eb", line_width=3)
                        st.plotly_chart(fig_curve, use_container_width=True)
                        render_interpretation_box("Curva de sensibilidade demonstrando a variação do número de parágrafos preservados à medida que o limiar θ_s varia de 0.00 a 1.00.")

                # ----------------------------------------------------
                # ESTATÍSTICAS E GRÁFICOS POR SENTENÇA-ÂNCORA
                # ----------------------------------------------------
                sem_res_list = getattr(st.session_state, "semantic_search_results", [])
                if sem_res_list:
                    st.divider()
                    st.markdown("### 🎯 Estatísticas Detalhadas por Sentença-Âncora")
                    st.caption("Desempenho comparativo individual de cada sentença-âncora de referência em relação ao corpus de parágrafos e documentos.")

                    df_anchor_stats = compute_per_anchor_statistics(
                        results=sem_res_list,
                        reference_set=st.session_state.reference_set,
                        threshold=float(fn["threshold"])
                    )
                    st.session_state.df_anchor_stats = df_anchor_stats

                    if not df_anchor_stats.empty:
                        # Gravação física imediata no disco
                        export_anchor_statistics_to_disk(project, df_anchor_stats, st.session_state.run_id)

                        st.dataframe(df_anchor_stats, use_container_width=True)

                        anc_csv_disk = project.semantic_dir / "estatisticas_ancoras.csv"
                        anc_csv_bytes = anc_csv_disk.read_bytes() if anc_csv_disk.exists() else df_anchor_stats.to_csv(index=False, encoding="utf-8").encode("utf-8")

                        anc_md_disk = project.semantic_dir / "estatisticas_ancoras.md"
                        anc_md_str = anc_md_disk.read_text(encoding="utf-8") if anc_md_disk.exists() else df_anchor_stats.to_markdown(index=False)

                        # Opções de Download das Estatísticas por Âncora
                        col_dl_anc1, col_dl_anc2 = st.columns(2)
                        col_dl_anc1.download_button(
                            label="📥 Baixar Estatísticas por Âncora em CSV (.csv)",
                            data=anc_csv_bytes,
                            file_name=f"estatisticas_ancoras_{st.session_state.run_id}.csv",
                            mime="text/csv",
                            key="dl_anc_csv_t4",
                            use_container_width=True
                        )
                        col_dl_anc2.download_button(
                            label="📥 Baixar Tabela de Âncoras em Markdown (.md)",
                            data=anc_md_str,
                            file_name=f"estatisticas_ancoras_{st.session_state.run_id}.md",
                            mime="text/markdown",
                            key="dl_anc_md_t4",
                            use_container_width=True
                        )

                        col_anc_plot1, col_anc_plot2 = st.columns(2)
                        with col_anc_plot1:
                            fig_anc_bars = px.bar(
                                df_anchor_stats,
                                x="Âncora ID",
                                y=["Parágrafos (≥ θ_s)", "Documentos Únicos (≥ θ_s)"],
                                barmode="group",
                                hover_data=["Texto da Âncora", "Score Médio", "Melhor Âncora (A*)"],
                                title="Retenção de Parágrafos e Documentos Únicos por Âncora",
                                labels={"value": "Quantidade", "variable": "Métrica"}
                            )
                            fig_anc_bars.update_layout(legend_title_text="Dimensão")
                            st.plotly_chart(fig_anc_bars, use_container_width=True)

                        with col_anc_plot2:
                            # Boxplot da distribuição de similaridade por âncora
                            anc_sample_data = []
                            for r in sem_res_list[:3000]:
                                for a_id, a_sc in r.anchor_scores.items():
                                    anc_sample_data.append({"Âncora ID": a_id, "Similaridade Cosseno": a_sc})
                            if anc_sample_data:
                                df_anc_box = pd.DataFrame(anc_sample_data)
                                fig_anc_box = px.box(
                                    df_anc_box,
                                    x="Âncora ID",
                                    y="Similaridade Cosseno",
                                    color="Âncora ID",
                                    title="Dispersão dos Scores Cosseno por Sentença-Âncora"
                                )
                                fig_anc_box.add_hline(
                                    y=float(fn["threshold"]),
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text=f"θ_s = {fn['threshold']:.2f}"
                                )
                                st.plotly_chart(fig_anc_box, use_container_width=True)

                if candidates:
                    st.divider()
                    st.subheader(f"Parágrafos Selecionados pela Busca Semântica ({len(candidates):,} Itens Preservados)".replace(",", "."))

                    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
                    with col_f1:
                        filter_sim_doc = st.selectbox(
                            "Filtrar por Documento ID",
                            options=["Todos"] + sorted(list(set(r.article_id for r in candidates))),
                            key="filter_sim_doc_t4"
                        )
                    with col_f2:
                        min_sim_score = st.slider(
                            "Filtrar por Score Mínimo de Cosseno",
                            min_value=float(fn["threshold"]),
                            max_value=1.0,
                            value=float(fn["threshold"]),
                            step=0.05,
                            key="min_sim_score_t4"
                        )
                    with col_f3:
                        n_display = st.selectbox("Exibir na Tela", options=[25, 50, 100, 200, "Todos"], index=1, key="n_disp_t4")

                    view_candidates = [r for r in candidates if (r.semantic_score or 0.0) >= min_sim_score]
                    if filter_sim_doc != "Todos":
                        view_candidates = [r for r in view_candidates if r.article_id == filter_sim_doc]

                    view_candidates = sorted(view_candidates, key=lambda r: r.semantic_score or 0.0, reverse=True)
                    limit = len(view_candidates) if n_display == "Todos" else int(n_display)

                    tab_table, tab_cards = st.tabs(["📋 Visão em Tabela Resumida", "📄 Visão Completa dos Parágrafos Selecionados"])

                    with tab_table:
                        st.markdown("#### Tabela de Parágrafos Selecionados (Ordenados por Score Cosseno)")
                        table_sim_rows = []
                        for r in view_candidates[:limit]:
                            table_sim_rows.append({
                                "ID do Parágrafo": r.paragraph_id,
                                "Documento ID": r.article_id,
                                "Score Cosseno (S)": f"{r.semantic_score or 0.0:.4f}",
                                "Texto (Resumo)": r.text[:140] + "..." if len(r.text) > 140 else r.text
                            })
                        st.dataframe(pd.DataFrame(table_sim_rows), use_container_width=True)

                    with tab_cards:
                        st.markdown(f"#### Exibição dos Parágrafos Selecionados (Mostrando {min(limit, len(view_candidates))} de {len(view_candidates)})")
                        for idx, r in enumerate(view_candidates[:limit], start=1):
                            st.markdown(
                                f"<div style='background-color: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #2563eb; padding: 14px 18px; margin-bottom: 12px; border-radius: 4px;'>"
                                f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>"
                                f"<strong style='color: #0f172a; font-size: 1.0rem;'>#{idx} — Parágrafo <code>{r.paragraph_id}</code> (Doc: <code>{r.article_id}</code>)</strong>"
                                f"<span style='background-color: #dbeafe; color: #1e40af; font-weight: 700; padding: 3px 12px; border-radius: 12px; font-size: 0.88rem;'>"
                                f"Score Cosseno: {r.semantic_score or 0.0:.4f}"
                                f"</span>"
                                f"</div>"
                                f"<p style='color: #334155; font-size: 0.95rem; line-height: 1.6; margin: 0px;'>{r.text}</p>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                    with st.expander("🔍 Inspeção Detalhada com Editor do Parágrafo Selecionado", expanded=False):
                        if view_candidates:
                            selected_cand_id = st.selectbox(
                                "Selecione um Parágrafo para Inspeção:",
                                options=[r.paragraph_id for r in view_candidates],
                                key="sel_cand_id_t4"
                            )
                            target_cand = next((r for r in view_candidates if r.paragraph_id == selected_cand_id), None)
                            if target_cand:
                                st.markdown(f"### Parágrafo `{target_cand.paragraph_id}` (Documento: `{target_cand.article_id}`)")
                                st.caption(f"Score Cosseno: `{target_cand.semantic_score:.4f}` | Limiar θ_s: `{fn['threshold']:.2f}`")
                                st.text_area("Texto Integral do Parágrafo:", value=target_cand.text, height=150, disabled=True)

                # ----------------------------------------------------
                # PERSISTÊNCIA FÍSICA AUTOMÁTICA NO DISCO
                # ----------------------------------------------------
                saved_cand_paths = export_semantic_candidates_to_disk(
                    project=project,
                    candidates=candidates,
                    run_id=st.session_state.run_id,
                    df_anchor_stats=getattr(st.session_state, "df_anchor_stats", None)
                )

                st.success(
                    f"💾 **Candidatos Semânticos e Estatísticas por Âncora Salvos no Disco!**\n\n"
                    f"Todos os dados da busca semântica foram gravados na pasta de saída configurada do projeto:\n\n"
                    f"- 🎯 **Estatísticas por Sentença-Âncora (CSV):** `{project.semantic_dir / 'estatisticas_ancoras.csv'}`\n"
                    f"- 🎯 **Estatísticas por Sentença-Âncora (Markdown):** `{project.semantic_dir / 'estatisticas_ancoras.md'}`\n"
                    f"- 🎯 **Estatísticas por Sentença-Âncora (Parquet):** `{project.semantic_dir / 'estatisticas_ancoras.parquet'}`\n"
                    f"- 📄 **Tabela Geral de Candidatos (CSV):** `{project.semantic_dir / 'candidates.csv'}`\n"
                    f"- ⚡ **Candidatos em Formato Colunar (Parquet):** `{project.semantic_dir / 'candidates.parquet'}`\n"
                    f"- 📦 **Candidatos em JSON Lines (JSONL):** `{project.semantic_dir / 'candidates.jsonl'}`\n"
                    f"- 📝 **Documento dos Candidatos (Markdown):** `{project.semantic_dir / 'candidates.md'}`\n\n"
                    f"📍 **Pasta Local dos Arquivos:** `{project.semantic_dir.resolve()}`"
                )

                col_sync_t4_1, col_sync_t4_2 = st.columns([2, 2])
                with col_sync_t4_1:
                    if st.button("🔄 Sincronizar e Gravar Candidatos e Estatísticas no Disco Novamente", key="btn_sync_t4_disk", use_container_width=True):
                        export_semantic_candidates_to_disk(
                            project=project,
                            candidates=candidates,
                            run_id=st.session_state.run_id,
                            df_anchor_stats=getattr(st.session_state, "df_anchor_stats", None)
                        )
                        st.toast("✓ Candidatos e estatísticas por âncora gravados no disco com sucesso!")
                with col_sync_t4_2:
                    st.caption(f"📁 Localização: `{project.semantic_dir}`")

                # Opção de Download da Etapa 4 (Recuperação por Cosseno)
                csv_cand_disk = project.semantic_dir / "candidates.csv"
                csv_cand_bytes = csv_cand_disk.read_bytes() if csv_cand_disk.exists() else df_t4.to_csv(index=False, encoding="utf-8").encode("utf-8")

                md_t4 = f"# Parágrafos Candidatos — Recuperação por Similaridade Semântica (Cosine Similarity)\n\n" + f"- **Threshold Aplicado (θ_s):** `{fn['threshold']:.2f}`\n- **Total Selecionados:** `{len(candidates)}`\n\n---\n\n" + "\n\n---\n\n".join(
                    f"## Parágrafo `{r.paragraph_id}` (`{r.article_id}`)\n- **Score Cosseno:** `{r.semantic_score:.4f}`\n\n{r.text}" for r in candidates[:1000]
                )
                df_t4 = pd.DataFrame([{"paragraph_id": r.paragraph_id, "article_id": r.article_id, "semantic_score": r.semantic_score, "text": r.text} for r in candidates])
                jsonl_t4 = "\n".join(json.dumps({"paragraph_id": r.paragraph_id, "article_id": r.article_id, "semantic_score": float(r.semantic_score or 0.0), "text": r.text}, ensure_ascii=False) for r in candidates)
                render_export_section("4_Similaridade_Semantica", st.session_state.run_id, md_t4, df_t4, jsonl_t4, file_prefix="similaridade_cosseno")

    # ==========================================
    # ABA 5: TREINAMENTO SUPERVISIONADO E ANOTAÇÃO
    # ==========================================
    with t5:
        render_methodology_header(
            title="5. Treinamento Supervisionado e Anotação Conceitual",
            description=(
                "Esta etapa possibilita a rotulação de trechos recuperados para o conceito investigado pela pesquisa. "
                "Suporta anotação interna no SLD, exportação de conjuntos em Markdown (.md) para trabalho externo ou colaborativo, "
                "importação com validação de integridade por hash e adjudicação de múltiplos anotadores para treinamento supervisionado."
            ),
            objective="Construir um Gold Standard auditável e treinar classificadores multilabel One-vs-Rest sobre Sentence Embeddings.",
            method="Amostragem ativa/estratificada, anotação interna/externa via Markdown (.md), validação estrita da Regra da Classe 0, matriz de confusão e concordância Cohen's Kappa.",
            formula_latex=r"P(y_k = 1 \mid X_i) = \frac{1}{1 + e^{-(w_k^T X_i + b_k)}}, \quad \kappa = \frac{p_o - p_e}{1 - p_e}",
            legend_dict={
                "X_i": "vetor de embedding do parágrafo i",
                "y_k": "indicador binário de pertencimento à classe k",
                "w_k": "vetor de coeficientes da classe k",
                "b_k": "termo de intercepto",
                r"\kappa": "coeficiente de concordância Cohen's Kappa",
                "p_o": "concordância observada entre anotadores",
                "p_e": "concordância esperada ao acaso"
            },
            interpretation="A anotação é totalmente genérica para qualquer conceito de pesquisa acadêmica definido pelo usuário."
        )

        render_stage_disk_loader(5, "Treinamento Supervisionado e Gold Standard", project, st.session_state)

        ann_service = AnnotationService(st.session_state.run_dirs["annotations"])
        st.session_state.gold_annotations = ann_service.load_annotations()

        col_concept, col_ann_id = st.columns([3, 1])
        with col_concept:
            concept_investigated = st.text_input(
                "Conceito Investigado na Pesquisa",
                value=st.session_state.config.get("research_concept", "conceito investigado"),
                help="Nome do conceito acadêmico em análise (ex: resiliência, risco, governança, adaptação)."
            )
            st.session_state.config["research_concept"] = concept_investigated

        with col_ann_id:
            annotator_name_input = st.text_input(
                "Nome do Anotador (Opcional)",
                value="",
                placeholder="Anotador anônimo",
                help="Nome ou identificador do anotador. Deixe em branco para anonimato."
            )
            annotator_id_val = "ANN_001" if not annotator_name_input else f"ANN_{abs(hash(annotator_name_input)) % 1000:03d}"

        sub_t1, sub_t2, sub_t3, sub_t4, sub_t5, sub_t6, sub_t7 = st.tabs([
            "📊 Visão Geral",
            "✍️ Anotar no SLD",
            "📤 Exportar para Anotação",
            "📥 Importar Anotações",
            "🏆 Gold Standard",
            "🧠 Treinar Modelo",
            "📈 Avaliação e Concordância"
        ])

        # ------------------------------------------
        # SUB-ABA 1: VISÃO GERAL
        # ------------------------------------------
        with sub_t1:
            st.subheader("Visão Geral do Acervo de Anotação")

            col_sync_ann1, col_sync_ann2 = st.columns([3, 1])
            with col_sync_ann1:
                st.caption(f"📁 Diretório de Anotações: `{ann_service.annotations_dir}`")
            with col_sync_ann2:
                if st.button("🔄 Recarregar Anotações do Disco", key="btn_reload_annos_sub1", use_container_width=True):
                    st.session_state.gold_annotations = ann_service.load_annotations()
                    st.toast(f"✓ {len(st.session_state.gold_annotations)} anotações recarregadas do disco!")
                    st.rerun()

            all_recs = ann_service.load_annotations()
            n_candidates = len([r for r in st.session_state.corpus_records if (r.semantic_score or 0.0) >= st.session_state.config["similarity_threshold"]])
            n_annotated_valid = len([r for r in all_recs if r.annotation_status == "valid"])
            n_unannotated = max(0, n_candidates - n_annotated_valid)
            n_gold = len(ann_service.get_gold_standard_records())
            unique_annotators = len({r.annotator_id for r in all_recs if r.annotator_id})

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Candidatos", f"{n_candidates:,}".replace(",", "."))
            m2.metric("Anotados Válidos", f"{n_annotated_valid:,}".replace(",", "."))
            m3.metric("Não Anotados", f"{n_unannotated:,}".replace(",", "."))
            m4.metric("Gold Standard", f"{n_gold:,}".replace(",", "."))
            m5.metric("Anotadores", f"{unique_annotators}")

            st.divider()

            st.markdown("#### Legenda Permanente das Classes Conceituais")
            st.code(
                "0 — Não relevante\n"
                "1 — Definição ou conceituação\n"
                "2 — Fator determinante\n"
                "3 — Tipo ou dimensão\n"
                "4 — Relação causal\n"
                "5 — Característica ou propriedade",
                language="text"
            )

            with st.expander("ⓘ Critérios Detalhados de Classificação Conceitual", expanded=False):
                for c_num in range(6):
                    st.markdown(f"**{CONCEPT_LABEL_NAMES[c_num]}:** {CONCEPT_LABEL_DESCRIPTIONS[c_num]}")

            st.divider()
            st.markdown("#### Distribuição de Frequência por Classe no Dataset")
            if all_recs:
                counts = {CONCEPT_LABEL_NAMES[c]: sum(1 for r in all_recs if getattr(r, f"label_{c}")) for c in range(6)}
                df_dist = pd.DataFrame([
                    {"Classe": k, "Frequência (N)": v, "Percentual (%)": f"{(v/len(all_recs)*100):.1f}%"}
                    for k, v in counts.items()
                ])
                st.dataframe(df_dist, use_container_width=True)
                st.caption("A soma das frequências das classes 1–5 pode superar o número total de trechos porque um mesmo trecho pode receber múltiplas categorias.")

        # ------------------------------------------
        # SUB-ABA 2: ANOTAR NO SLD
        # ------------------------------------------
        with sub_t2:
            st.subheader("Anotação Interna Sequencial no SLD")
            st.caption(f"Conceito Ativo: **{concept_investigated}** | Anotador: **{annotator_name_input or 'Anotador Anônimo'}** (`{annotator_id_val}`)")

            c_num, c_strat, c_btn = st.columns([2, 3, 2])
            with c_num:
                n_to_sample = st.number_input("Quantidade de Parágrafos no Lote", min_value=1, max_value=50, value=5, step=1, key="sub2_num")
            with c_strat:
                strategy_choice = st.selectbox(
                    "Critério de Amostragem",
                    options=["random", "stratified", "uncertainty", "top_k"],
                    format_func=lambda x: {
                        "random": "Aleatória Simples",
                        "stratified": "Estratificada por Quantis de Similaridade",
                        "uncertainty": "Active Learning (Maior Incerteza)",
                        "top_k": "Top-K Similaridade Semântica"
                    }[x],
                    key="sub2_strat"
                )
            with c_btn:
                st.write("")
                st.write("")
                if st.button("Gerar Lote Sequencial para Anotação", key="sub2_gen_btn"):
                    if not st.session_state.corpus_records:
                        st.warning("Nenhum parágrafo carregado. Complete a etapa 1 (Ingestão).")
                    else:
                        batch_samples = sample_paragraphs(
                            st.session_state.corpus_records,
                            n_samples=int(n_to_sample),
                            strategy=strategy_choice,
                            random_seed=42
                        )
                        st.session_state.active_sampling_batch = batch_samples
                        st.success(f"Lote sequencial com {len(batch_samples)} parágrafos gerado!")
                        st.rerun()

            st.divider()

            if st.session_state.active_sampling_batch:
                st.markdown(f"### Lote Ativo ({len(st.session_state.active_sampling_batch)} Trechos)")

                with st.form(key="internal_annotation_batch_form", clear_on_submit=False):
                    for item_idx, rec in enumerate(st.session_state.active_sampling_batch, start=1):
                        st.markdown(f"#### Item #{item_idx} — `{rec.paragraph_id}` (`{rec.article_id}`)")
                        st.text_area(f"Texto do Trecho #{item_idx}:", value=rec.text, height=100, disabled=True, key=f"txt_sub2_{item_idx}_{rec.paragraph_id}")

                        st.write("**Selecione as Categorias Conceituais:**")
                        c0, c1, c2, c3, c4, c5 = st.columns(6)
                        c0.checkbox("0 — Não Relevante", key=f"chk0_sub2_{item_idx}_{rec.paragraph_id}")
                        c1.checkbox("1 — Definição", key=f"chk1_sub2_{item_idx}_{rec.paragraph_id}")
                        c2.checkbox("2 — Fator", key=f"chk2_sub2_{item_idx}_{rec.paragraph_id}")
                        c3.checkbox("3 — Dimensão", key=f"chk3_sub2_{item_idx}_{rec.paragraph_id}")
                        c4.checkbox("4 — Causal", key=f"chk4_sub2_{item_idx}_{rec.paragraph_id}")
                        c5.checkbox("5 — Propriedade", key=f"chk5_sub2_{item_idx}_{rec.paragraph_id}")

                        st.text_input("Observação opcional", key=f"note_sub2_{item_idx}_{rec.paragraph_id}")
                        st.divider()

                    save_batch_submitted = st.form_submit_button("💾 Salvar Lote no Dataset de Anotação", type="primary")

                if save_batch_submitted:
                    batch_annotations = []
                    for item_idx, rec in enumerate(st.session_state.active_sampling_batch, start=1):
                        c0_val = st.session_state.get(f"chk0_sub2_{item_idx}_{rec.paragraph_id}", False)
                        c1_val = st.session_state.get(f"chk1_sub2_{item_idx}_{rec.paragraph_id}", False)
                        c2_val = st.session_state.get(f"chk2_sub2_{item_idx}_{rec.paragraph_id}", False)
                        c3_val = st.session_state.get(f"chk3_sub2_{item_idx}_{rec.paragraph_id}", False)
                        c4_val = st.session_state.get(f"chk4_sub2_{item_idx}_{rec.paragraph_id}", False)
                        c5_val = st.session_state.get(f"chk5_sub2_{item_idx}_{rec.paragraph_id}", False)
                        n_val = st.session_state.get(f"note_sub2_{item_idx}_{rec.paragraph_id}", "")

                        is_unannotated = not (c0_val or c1_val or c2_val or c3_val or c4_val or c5_val)
                        anno_status = "unannotated" if is_unannotated else ("valid" if not (c0_val and (c1_val or c2_val or c3_val or c4_val or c5_val)) else "invalid")

                        batch_annotations.append(
                            AnnotationRecord(
                                annotation_id=f"ANN_{rec.paragraph_id}_{annotator_id_val}",
                                dataset_id="ANNOTATION_SET_INTERNAL",
                                run_id=st.session_state.run_id,
                                document_id=rec.article_id,
                                paragraph_id=rec.paragraph_id,
                                annotator_id=annotator_id_val,
                                annotator_name=annotator_name_input if annotator_name_input else None,
                                annotation_source="internal",
                                label_0=c0_val,
                                label_1=c1_val,
                                label_2=c2_val,
                                label_3=c3_val,
                                label_4=c4_val,
                                label_5=c5_val,
                                annotation_status=anno_status,
                                annotation_note=n_val,
                                text_hash=rec.text_sha256,
                                included_in_gold_standard=(anno_status == "valid")
                            )
                        )

                    with st.status("Salvando anotações internas no acervo...", expanded=True) as status_box:
                        ann_service.save_annotations(batch_annotations)
                        st.session_state.gold_annotations = ann_service.load_annotations()
                        st.session_state.active_sampling_batch = []
                        status_box.update(label="✓ Anotações salvas no acervo com sucesso!", state="complete", expanded=False)
                        st.toast(f"✓ {len(batch_annotations)} anotações salvas!")
                        st.balloons()
                        st.rerun()

        # ------------------------------------------
        # SUB-ABA 3: EXPORTAR PARA ANOTAÇÃO
        # ------------------------------------------
        with sub_t3:
            st.subheader("Exportar Conjunto para Anotação Externa")
            st.markdown(
                "<div class='sld-card'>"
                "<strong>Anotação Externa:</strong> Exporte um conjunto de trechos para classificação manual ou colaborativa em formato Markdown (.md) "
                "ou tabular (.csv, .jsonl) e importe posteriormente as anotações para o SLD com total rastreabilidade."
                "</div>",
                unsafe_allow_html=True
            )
            st.write("")

            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                exp_set_id = st.text_input("Identificador do Conjunto (Dataset ID)", value="ANNOTATION_SET_001")
                exp_n_samples = st.number_input("Quantidade de Trechos a Exportar", min_value=1, max_value=500, value=20, step=5)
                exp_strategy = st.selectbox(
                    "Estratégia de Amostragem para Exportação",
                    options=["random", "stratified", "top_k", "unannotated"],
                    format_func=lambda x: {
                        "random": "Aleatória Simples",
                        "stratified": "Estratificada por Faixa de Similaridade",
                        "top_k": "Casos de Maior Similaridade Semântica",
                        "unannotated": "Trechos Ainda Não Anotados"
                    }[x]
                )

            with col_ex2:
                st.markdown("#### Configurações de Anonimização / Modo Cego")
                exp_hide_scores = st.checkbox("Ocultar Score Semântico no Arquivo", value=True, help="Oculta o valor do score semântico para reduzir viés no anotador.")
                exp_blind_mode = st.checkbox("Modo Cego Completo (Ocultar Nome do Documento)", value=False, help="Oculta o document_id para garantir avaliação cega independente.")

            if st.button("Gerar Conjunto para Anotação Externa", type="primary"):
                if not st.session_state.corpus_records:
                    st.warning("Nenhum parágrafo disponível. Complete a etapa 1 (Ingestão).")
                else:
                    exp_paras = sample_paragraphs(
                        st.session_state.corpus_records,
                        n_samples=int(exp_n_samples),
                        strategy=exp_strategy if exp_strategy != "unannotated" else "random",
                        random_seed=42
                    )

                    md_path = ann_service.export_annotation_set(
                        paragraphs=exp_paras,
                        dataset_id=exp_set_id,
                        run_id=st.session_state.run_id,
                        concept=concept_investigated,
                        annotator_name=annotator_name_input,
                        hide_scores=exp_hide_scores,
                        blind_mode=exp_blind_mode
                    )

                    with open(md_path, "r", encoding="utf-8") as f:
                        md_data_str = f.read()

                    df_exp_csv = pd.DataFrame([{
                        "dataset_id": exp_set_id,
                        "paragraph_id": p.paragraph_id,
                        "document_id": p.article_id if not exp_blind_mode else "BLIND",
                        "text": p.text,
                        "text_hash": p.text_sha256
                    } for p in exp_paras])

                    st.success(f"Conjunto `{exp_set_id}` gerado com sucesso com {len(exp_paras)} trechos!")
                    
                    c_dl1, c_dl2, c_dl3 = st.columns(3)
                    c_dl1.download_button(
                        label="Baixar Arquivo Markdown (.md)",
                        data=md_data_str,
                        file_name=f"{exp_set_id}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                    c_dl2.download_button(
                        label="Baixar Tabela CSV (.csv)",
                        data=df_exp_csv.to_csv(index=False, encoding="utf-8"),
                        file_name=f"{exp_set_id}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    c_dl3.download_button(
                        label="Baixar Dados JSONL (.jsonl)",
                        data=df_exp_csv.to_json(orient="records", lines=True, force_ascii=False),
                        file_name=f"{exp_set_id}.jsonl",
                        mime="application/jsonlines",
                        use_container_width=True
                    )
                    st.balloons()

        # ------------------------------------------
        # SUB-ABA 4: IMPORTAR ANOTAÇÕES
        # ------------------------------------------
        with sub_t4:
            st.subheader("Importar Anotações Externas (.md, .csv, .jsonl)")
            uploaded_file = st.file_uploader(
                "Selecione o arquivo de anotação preenchido (.md, .csv, .jsonl)",
                type=["md", "csv", "jsonl"],
                help="Faça o upload do arquivo Markdown (.md), CSV (.csv) ou JSONL (.jsonl, ex: gold_standard.jsonl) anotado manualmente.",
                key="uploader_annotations_sub4"
            )

            if uploaded_file is not None:
                file_bytes = uploaded_file.getvalue()
                try:
                    file_text = file_bytes.decode("utf-8")
                except Exception:
                    file_text = file_bytes.decode("latin-1", errors="ignore")

                with st.status("Validando arquivo de anotação importado...", expanded=True) as status_box:
                    val_res = ann_service.validate_import_file(file_text, uploaded_file.name, st.session_state.corpus_records)
                    status_box.update(label="✓ Validação concluída!", state="complete", expanded=False)

                st.markdown("#### Resultado do Reconhecimento e Validação")
                v_col1, v_col2, v_col3, v_col4, v_col5 = st.columns(5)
                v_col1.metric("Formato", f"v{val_res['format_version']}")
                v_col2.metric("Encontrados", val_res["total_items"])
                v_col3.metric("Válidos", val_res["n_valid"])
                v_col4.metric("Inválidos", val_res["n_invalid"])
                v_col5.metric("Não Anotados", val_res["n_unannotated"])

                if val_res["warnings"]:
                    for w in val_res["warnings"]:
                        render_methodological_alert(w, alert_type="warning")

                st.markdown("#### Pré-visualização das Anotações Reconhecidas")
                df_val_preview = pd.DataFrame(val_res["validated_rows"])[
                    ["paragraph_id", "document_id", "labels_str", "status", "hash_valid", "error"]
                ]
                st.dataframe(df_val_preview, use_container_width=True)

                if st.button("Confirmar Importação de Anotações Validadas", type="primary", key="btn_confirm_import_annos"):
                    with st.status("Incorporando anotações validadas ao acervo...", expanded=True) as status_box:
                        imported_recs = ann_service.import_validated_markdown(
                            val_res,
                            annotator_name_override=annotator_name_input,
                            annotator_id_override=annotator_id_val
                        )
                        st.session_state.gold_annotations = ann_service.load_annotations()
                        status_box.update(label=f"✓ {len(imported_recs)} anotações incorporadas com sucesso!", state="complete", expanded=False)
                        st.toast(f"✓ {len(imported_recs)} anotações incorporadas com sucesso!")
                        st.balloons()
                        st.rerun()

        # ------------------------------------------
        # SUB-ABA 5: GOLD STANDARD E ADJUDICAÇÃO
        # ------------------------------------------
        with sub_t5:
            st.subheader("Gold Standard Consolidado e Adjudicação de Conflitos")
            
            gold_recs = ann_service.get_gold_standard_records()
            st.info(f"Registros Válidos e Aprovados no Gold Standard: **{len(gold_recs)}**")

            # Adjudicação de Conflitos
            all_annos = ann_service.load_annotations()
            p_map: Dict[str, List[AnnotationRecord]] = {}
            for a in all_annos:
                p_map.setdefault(a.paragraph_id, []).append(a)

            conflicts = {p_id: recs for p_id, recs in p_map.items() if len({r.labels_binary_tuple for r in recs if hasattr(r, 'labels_binary_tuple')}) > 1 or len(set(tuple(r.labels_list) for r in recs)) > 1}

            if conflicts:
                st.warning(f"Foram detectados **{len(conflicts)} parágrafos com anotações conflitantes** entre diferentes anotadores.")
                with st.expander("Painel de Adjudicação de Conflitos", expanded=True):
                    for c_p_id, c_recs in list(conflicts.items())[:10]:
                        st.markdown(f"#### Conflito no Parágrafo `{c_p_id}`")
                        for idx_c, r_c in enumerate(c_recs, start=1):
                            st.write(f"• **Anotador {r_c.annotator_name or r_c.annotator_id}:** Classes `{r_c.labels_list}` (Fonte: `{r_c.annotation_source}`)")
                        st.divider()

            if gold_recs:
                st.markdown("#### Tabela do Gold StandardConsolidado")
                df_gold_view = pd.DataFrame([{
                    "Paragraph ID": r.paragraph_id,
                    "Document ID": r.document_id,
                    "Anotador": r.annotator_name or r.annotator_id,
                    "Classes": ", ".join(str(c) for c in r.labels_list),
                    "Origem": r.annotation_source,
                    "Status": r.annotation_status
                } for r in gold_recs])
                st.dataframe(df_gold_view, use_container_width=True)

        # ------------------------------------------
        # SUB-ABA 6: TREINAR MODELO
        # ------------------------------------------
        with sub_t6:
            st.subheader("Treinamento do Classificador Supervisionado")
            
            with st.expander("Parâmetros do Classificador (Regressão Logística Multilabel)", expanded=True):
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    c_val = st.number_input("Parâmetro de Regularização (C)", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
                    class_weight_choice = st.selectbox("Ponderação de Classes (class_weight)", options=["balanced", "none"])
                    seed_clf = st.number_input("Semente de Aleatoriedade (random_state)", value=42, step=1)

                with col_p2:
                    max_iter_val = st.number_input("Máximo de Iterações (max_iter)", min_value=100, max_value=2000, value=500, step=50)

            valid_gold_records = ann_service.get_gold_standard_records()
            st.info(f"Registros válidos disponíveis no Gold Standard para treinamento: **{len(valid_gold_records)}**")

            if st.button("Treinar Regressão Logística Multilabel", type="primary", key="train_model_sub6_btn"):
                if len(valid_gold_records) < 5:
                    st.warning("Adicione pelo menos 5 anotações válidas no Gold Standard antes de treinar.")
                elif st.session_state.embeddings_matrix is None and not st.session_state.corpus_records:
                    st.warning("Execute a etapa de Ingestão e Vetorização de Embeddings primeiro.")
                else:
                    with st.status("Treinando modelo de regressão logística multilabel...", expanded=True) as status_box:
                        start_t = time.time()
                        if st.session_state.embeddings_matrix is None:
                            with st.spinner("Gerando matriz de embeddings para o corpus..."):
                                emb_service = EmbeddingService(st.session_state.config.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
                                texts = [r.text for r in st.session_state.corpus_records]
                                st.session_state.embeddings_matrix = emb_service.get_embeddings(texts)

                        # Mapeia registros válidos para o treinamento
                        id_to_idx = {r.paragraph_id: idx for idx, r in enumerate(st.session_state.corpus_records)}
                        X_tr_list, y_tr_list = [], []
                        for a in valid_gold_records:
                            if a.paragraph_id in id_to_idx:
                                X_tr_list.append(st.session_state.embeddings_matrix[id_to_idx[a.paragraph_id]])
                                y_tr_list.append(a.labels_binary)

                        if not X_tr_list:
                            st.error("Nenhum embedding encontrado para os parágrafos do Gold Standard.")
                        else:
                            X_tr = np.array(X_tr_list)
                            y_tr = np.array(y_tr_list)

                            cw_val = class_weight_choice if class_weight_choice != "none" else None
                            clf = MultilabelLogisticClassifier(C=c_val, class_weight=cw_val, max_iter=int(max_iter_val), random_state=int(seed_clf))
                            clf.fit(X_tr, y_tr)
                            clf.save(st.session_state.run_dirs["models"])
                            st.session_state.logistic_classifier = clf

                            # Cálculo das métricas completas de avaliação supervisionada
                            thresh_dict = getattr(st.session_state, "optimal_thresholds", None) or {
                                "definition": 0.50, "determinant": 0.50, "type_dimension": 0.50, "causal_relation": 0.50, "property": 0.50
                            }
                            y_probs = clf.predict_proba(X_tr)
                            y_pred_bin = clf.predict_with_thresholds(X_tr, thresh_dict)

                            eval_rep = compute_multilabel_evaluation(
                                model_id=f"logistic_regression_{st.session_state.run_id}",
                                classifier_type="LogisticRegression (One-vs-Rest Multilabel)",
                                y_true_binary=y_tr,
                                y_probs=y_probs,
                                y_pred_binary=y_pred_bin,
                                thresholds=thresh_dict,
                                total_articles=len(set(a.document_id for a in valid_gold_records if a.document_id))
                            )
                            st.session_state.evaluation_report = eval_rep
                            st.session_state.eval_report = eval_rep

                            dur = time.time() - start_t
                            if 5 not in st.session_state.completed_steps:
                                st.session_state.completed_steps.append(5)

                            save_full_session_state(project, st.session_state)
                            import gc; gc.collect()

                            status_box.update(label=f"✓ Modelo supervisionado treinado e avaliado com sucesso em {dur:.1f}s!", state="complete", expanded=False)
                            st.toast("✓ Classificador treinado e métricas calculadas!")
                            st.balloons()
                            render_completion_panel(
                                title="Treinamento e Avaliação do Classificador Concluídos",
                                metrics={
                                    "Amostras Gold Standard": len(valid_gold_records),
                                    "Macro-F1": f"{eval_rep.macro_f1:.4f}",
                                    "Micro-F1": f"{eval_rep.micro_f1:.4f}",
                                    "Tempo Total": f"{dur:.2f}s"
                                }
                            )

            # Se o modelo já estiver treinado, renderiza o painel de avaliação supervisionada em Sub-t6
            if getattr(st.session_state, "evaluation_report", None) is not None:
                st.divider()
                st.subheader("Avaliação Detalhada do Modelo Treinado")
                render_supervised_evaluation_panel(
                    eval_report=st.session_state.evaluation_report,
                    project=project,
                    run_id=st.session_state.run_id,
                    key_suffix="sub6"
                )

        # ------------------------------------------
        # SUB-ABA 7: AVALIAÇÃO E CONCORDÂNCIA
        # ------------------------------------------
        with sub_t7:
            st.subheader("Avaliação Quantitativa do Modelo e Concordância entre Anotadores")

            tab_ev1, tab_ev2 = st.tabs([
                "📈 1. Desempenho do Modelo Supervisionado (Macro/Micro F1, Precision, Recall e Matrizes)",
                "👥 2. Concordância Interanotador (Cohen's Kappa)"
            ])

            with tab_ev1:
                eval_rep_to_show = getattr(st.session_state, "evaluation_report", None) or getattr(st.session_state, "eval_report", None)
                if eval_rep_to_show is None and st.session_state.logistic_classifier is not None and len(valid_gold_records) >= 5 and st.session_state.embeddings_matrix is not None:
                    try:
                        id_to_idx = {r.paragraph_id: idx for idx, r in enumerate(st.session_state.corpus_records)}
                        X_ev_list, y_ev_list = [], []
                        for a in valid_gold_records:
                            if a.paragraph_id in id_to_idx:
                                X_ev_list.append(st.session_state.embeddings_matrix[id_to_idx[a.paragraph_id]])
                                y_ev_list.append(a.labels_binary)
                        if X_ev_list:
                            X_ev = np.array(X_ev_list)
                            y_ev = np.array(y_ev_list)
                            clf = st.session_state.logistic_classifier
                            th_d = getattr(st.session_state, "optimal_thresholds", None) or {"definition": 0.5, "determinant": 0.5, "type_dimension": 0.5, "causal_relation": 0.5, "property": 0.5}
                            y_probs = clf.predict_proba(X_ev)
                            y_pred_bin = clf.predict_with_thresholds(X_ev, th_d)
                            eval_rep_to_show = compute_multilabel_evaluation(
                                model_id=f"logistic_regression_{st.session_state.run_id}",
                                classifier_type="LogisticRegression (One-vs-Rest Multilabel)",
                                y_true_binary=y_ev,
                                y_probs=y_probs,
                                y_pred_binary=y_pred_bin,
                                thresholds=th_d,
                                total_articles=len(set(a.document_id for a in valid_gold_records if a.document_id))
                            )
                            st.session_state.evaluation_report = eval_rep_to_show
                    except Exception as e:
                        logger.warning(f"Erro ao computar relatório de avaliação na sub-aba 7: {e}")

                if eval_rep_to_show is not None:
                    render_supervised_evaluation_panel(
                        eval_report=eval_rep_to_show,
                        project=project,
                        run_id=st.session_state.run_id,
                        key_suffix="sub7"
                    )
                else:
                    render_empty_state(
                        title="Avaliação Supervisionada Indisponível",
                        description="Treine o classificador na Sub-aba '6. Treinar Modelo' para gerar métricas completas (Macro-F1, Micro-F1, Precisão, Recall e Matrizes de Confusão Binárias por classe).",
                        recommendation="Aguardando treinamento do modelo."
                    )

            with tab_ev2:
                all_recs = ann_service.load_annotations()
                agreement_res = compute_inter_annotator_agreement(all_recs)

                if agreement_res.get("has_paired_annotations"):
                    st.markdown("#### Concordância entre Anotadores (Cohen's Kappa — κ)")
                    st.latex(r"\kappa = \frac{p_o - p_e}{1 - p_e}")
                    st.markdown(
                        "**Onde:**\n"
                        "- **κ:** Coeficiente Kappa de Cohen\n"
                        "- **p_o:** Proporção de concordância observada entre os dois anotadores\n"
                        "- **p_e:** Proporção de concordância esperada ao acaso"
                    )

                    st.metric("Macro Kappa (Concordância Global)", f"{agreement_res['macro_kappa']:.4f}")
                    st.write("")

                    df_agree = pd.DataFrame([
                        {
                            "Classe Conceitual": k,
                            "Concordância Observada (p_o)": f"{v['p_o']:.4f}",
                            "Concordância Esperada (p_e)": f"{v['p_e']:.4f}",
                            "Cohen's Kappa (κ)": f"{v['kappa']:.4f}",
                            "Amostras Pareadas": v["n_samples"]
                        }
                        for k, v in agreement_res["per_class_agreement"].items()
                    ])
                    st.dataframe(df_agree, use_container_width=True)
                    render_interpretation_box("O coeficiente Kappa avalia a concordância entre anotadores descontando a concordância esperada ao acaso.")
                else:
                    st.info(agreement_res.get("message", "Nenhum dado de concordância disponível."))

    # ==========================================
    # ABA 6: CLASSIFICAÇÃO CONCEITUAL
    # ==========================================
    with t6:
        render_methodology_header(
            title="6. Classificação Conceitual dos Candidatos",
            description=(
                "O classificador treinado é aplicado aos parágrafos recuperados semanticamente, produzindo probabilidades independentes para cada categoria conceitual. "
                "Como a classificação é multilabel, um mesmo parágrafo pode pertencer simultaneamente a mais de uma categoria."
            ),
            objective="Atribuir categorias analíticas aos parágrafos candidatos recuperados.",
            method="Predição de probabilidade por classe e aplicação de limiares de decisão específicos por categoria conceitual.",
            formula_latex=r"\hat{y}_{ik} = I[P(y_k = 1 \mid X_i) \ge \theta_k]",
            legend_dict={
                r"\hat{y}_{ik}": "classificação binária atribuída ao parágrafo i na classe k",
                "P(y_k=1 | X_i)": "probabilidade estimada pelo classificador supervisionado",
                r"\theta_k": "limiar de decisão específico e otimizado da classe k",
                "I[.]": "função indicadora"
            },
            interpretation="Rotula automaticamente cada parágrafo candidato com uma ou mais dimensões conceituais da vulnerabilidade."
        )

        render_stage_disk_loader(6, "Classificação Conceitual dos Candidatos", project, st.session_state)

        # Restauração automática de modelo salvo em disco se session_state estiver vazio
        if st.session_state.logistic_classifier is None:
            model_dir = st.session_state.run_dirs.get("models", Path("./output/models"))
            if (model_dir / "logistic_classifier.joblib").exists():
                try:
                    st.session_state.logistic_classifier = MultilabelLogisticClassifier.load(model_dir)
                except Exception as err:
                    st.warning(f"Não foi possível carregar o modelo treinado em disco: {err}")

        # Restauração automática do vetor de embeddings se session_state estiver vazio
        if st.session_state.embeddings_matrix is None and DEFAULT_INDEX_DIR.exists():
            try:
                v_idx_load = VectorIndex.load(DEFAULT_INDEX_DIR)
                if v_idx_load.embeddings is not None:
                    st.session_state.embeddings_matrix = v_idx_load.embeddings
            except Exception:
                pass

        if st.session_state.logistic_classifier is None:
            render_empty_state(
                title="Classificador Não Treinado",
                description="Treine o classificador na Etapa 5 (Treinamento Supervisionado) antes de executar a classificação conceitual.",
                recommendation="Aguardando treinamento do modelo."
            )
        elif st.session_state.embeddings_matrix is None:
            render_empty_state(
                title="Embeddings Não Disponíveis",
                description="Gere os embeddings na Etapa 2 (Sentence Embeddings) antes de executar a classificação.",
                recommendation="Aguardando geração dos vetores."
            )
        else:
            # Cartão de Status da Classificação
            if "6_Classificacao" in st.session_state.funnel_counts:
                fn_cl = st.session_state.funnel_counts["6_Classificacao"]
                render_status_card(
                    title="Status da Classificação Conceitual",
                    status_state="completed",
                    details=f"Concluído: {fn_cl['n_out']:,} parágrafos rotulados como relevantes de {fn_cl['n_in']:,} analisados em {fn_cl['duration_sec']:.2f}s.".replace(",", ".")
                )
            else:
                render_status_card(
                    title="Status da Classificação Conceitual",
                    status_state="idle",
                    details="Aguardando disparo da classificação conceitual multilabel."
                )

            st.divider()

            # Restauração e sincronização estrita dos scores da Etapa 4 (Cascata Metodológica)
            if st.session_state.semantic_scores_map:
                for r in st.session_state.corpus_records:
                    if r.semantic_score is None:
                        k = f"{r.article_id}_{r.paragraph_id}"
                        r.semantic_score = st.session_state.semantic_scores_map.get(k, st.session_state.semantic_scores_map.get(r.paragraph_id, 0.0))

            sim_th_used = st.session_state.funnel_counts.get("4_Similaridade", {}).get("threshold", st.session_state.config.get("similarity_threshold", 0.50))
            col_t6_th, col_t6_cnt = st.columns([2, 3])
            with col_t6_th:
                th_sim_val = st.number_input(
                    "Limiar Semântico para Candidatos (θ_s):",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(sim_th_used),
                    step=0.05,
                    key="t6_threshold_input",
                    help="Parágrafos com score de similaridade igual ou superior a este limiar na Etapa 4 serão classificados."
                )
            with col_t6_cnt:
                cands_at_th = [r for r in st.session_state.corpus_records if (r.semantic_score is not None and r.semantic_score >= th_sim_val)]
                if not cands_at_th and st.session_state.semantic_candidates:
                    cands_at_th = [r for r in st.session_state.semantic_candidates if (r.semantic_score or 0.0) >= th_sim_val]

                if not cands_at_th:
                    if any(r.semantic_score is not None for r in st.session_state.corpus_records):
                        max_s = max(r.semantic_score for r in st.session_state.corpus_records if r.semantic_score is not None)
                        st.warning(f"⚠️ Nenhum parágrafo atingiu θ_s={th_sim_val:.2f} (o maior score de similaridade calculado na Etapa 4 foi {max_s:.4f}). Ajuste o limiar ou execute novamente a Etapa 4.")
                    else:
                        st.warning("⚠️ Nenhum score de similaridade detectado. Execute a **Etapa 4 (Similaridade Semântica)** antes de classificar.")
                else:
                    st.markdown(f"<div style='margin-top: 28px;'><strong>Candidatos Refinados na Etapa 4:</strong> <code>{len(cands_at_th):,}</code> parágrafos</div>".replace(",", "."), unsafe_allow_html=True)

            if st.button("Executar Classificação Conceitual Multilabel", type="primary", disabled=(len(cands_at_th) == 0)):
                with st.status(f"Classificando {len(cands_at_th):,} candidatos refinados na Etapa 4...", expanded=True) as status_box:
                    try:
                        candidates = cands_at_th

                        if not candidates:
                            status_box.update(label="✕ Nenhum parágrafo candidato selecionado.", state="error", expanded=False)
                            st.error("Nenhum parágrafo candidato selecionado. Execute a Etapa 4 (Similaridade Semântica) para refinar o corpus.")
                        else:
                            start_t = time.time()
                            id_to_idx = {r.paragraph_id: idx for idx, r in enumerate(st.session_state.corpus_records)}

                            batch_sz = st.session_state.config.get("semantic_batch_size", DEFAULT_SEMANTIC_BATCH_SIZE)
                            total_cand = len(candidates)
                            relevant_count = 0

                            for b_start in range(0, total_cand, batch_sz):
                                b_end = min(b_start + batch_sz, total_cand)
                                b_cand = candidates[b_start:b_end]

                                batch_indices = [id_to_idx[r.paragraph_id] for r in b_cand if r.paragraph_id in id_to_idx]
                                if not batch_indices:
                                    continue

                                X_batch = st.session_state.embeddings_matrix[batch_indices]
                                probs_batch = st.session_state.logistic_classifier.predict_proba(X_batch)
                                preds_batch = st.session_state.logistic_classifier.predict(X_batch)

                                for idx, r in enumerate(b_cand):
                                    p_row = probs_batch[idx]
                                    l_row = preds_batch[idx]

                                    r.predicted_probabilities = {CONCEPT_LABEL_SHORT_NAMES[c]: float(p_row[c-1]) for c in MULTILABEL_CLASSES}
                                    r.predicted_labels = [CONCEPT_LABEL_SHORT_NAMES[c] for c in MULTILABEL_CLASSES if l_row[c-1] == 1]

                                    if r.predicted_labels:
                                        r.status = "MODEL_RELEVANT"
                                        relevant_count += 1
                                    else:
                                        r.status = "MODEL_NOT_RELEVANT"

                            dur = time.time() - start_t
                            st.session_state.classified_records = candidates

                            # Sincroniza e reseta registros não classificados nesta rodada para não misturar execuções antigas
                            cand_id_set = {r.paragraph_id for r in candidates}
                            for r in st.session_state.corpus_records:
                                if r.paragraph_id not in cand_id_set and r.status in ["MODEL_RELEVANT", "MODEL_NOT_RELEVANT"]:
                                    r.status = "CANDIDATE" if r.semantic_score is not None else "INGESTED"
                                    r.predicted_labels = []
                                    r.predicted_probabilities = {}

                            # Invalida explicitamente os artefatos do Índice RAG (Etapa 7) para forçar nova construção
                            st.session_state.rag_index_stats = None
                            st.session_state.rag_index_manifest = None
                            st.session_state.rag_index_zip_path = None
                            st.session_state.rag_retriever = None
                            if 7 in st.session_state.completed_steps:
                                st.session_state.completed_steps.remove(7)

                            st.session_state.funnel_counts["6_Classificacao"] = {
                                "n_in": total_cand,
                                "n_out": relevant_count,
                                "duration_sec": dur,
                                "threshold": th_sim_val
                            }

                            if 6 not in st.session_state.completed_steps:
                                st.session_state.completed_steps.append(6)

                            save_full_session_state(project, st.session_state)
                            import gc; gc.collect()

                            status_box.update(label=f"✓ Classificação concluída: {relevant_count:,} parágrafos relevantes de {total_cand:,} candidatos em {dur:.2f}s", state="complete", expanded=False)
                            st.toast(f"✓ Classificação concluída: {relevant_count:,} parágrafos relevantes.")
                            st.balloons()
                            st.rerun()

                    except Exception as err:
                        status_box.update(label=f"✕ Ocorreu um erro durante a classificação: {err}", state="error", expanded=True)
                        st.error(f"Erro detalhado na execução da classificação: {err}")

            classified_records = st.session_state.classified_records or [r for r in st.session_state.corpus_records if r.status in ["MODEL_RELEVANT", "MODEL_NOT_RELEVANT"]]

            if classified_records:
                st.divider()

                if "6_Classificacao" in st.session_state.funnel_counts:
                    fn_cl = st.session_state.funnel_counts["6_Classificacao"]
                    n_in = fn_cl["n_in"]
                    n_out = fn_cl["n_out"]
                    dur = fn_cl["duration_sec"]

                    render_pipeline_metrics(
                        input_count=n_in,
                        output_count=n_out,
                        duration_sec=dur,
                        n_docs=len(set(r.article_id for r in classified_records)),
                        unit_label="parágrafos relevantes"
                    )

                    render_completion_panel(
                        title="Classificação Conceitual Multilabel Concluída",
                        metrics={
                            "Entrada (Candidatos)": f"{n_in:,}".replace(",", "."),
                            "Relevantes (MODEL_RELEVANT)": f"{n_out:,}".replace(",", "."),
                            "Não Relevantes (Filtrados)": f"{max(0, n_in - n_out):,}".replace(",", "."),
                            "Taxa de Retenção": f"{(n_out / n_in * 100):.2f}%" if n_in > 0 else "0.00%",
                            "Tempo de Execução": f"{dur:.2f}s"
                        }
                    )

                # ----------------------------------------------------
                # ESTATÍSTICAS AVANÇADAS: PARÁGRAFOS E DOCUMENTOS
                # ----------------------------------------------------
                total_p = len(classified_records)
                all_docs = sorted(list(set(r.article_id for r in classified_records)))
                total_docs = len(all_docs)

                rel_p = [r for r in classified_records if r.status == "MODEL_RELEVANT"]
                not_rel_p = [r for r in classified_records if r.status == "MODEL_NOT_RELEVANT"]

                rel_docs = set(r.article_id for r in rel_p)
                not_rel_docs = set(all_docs) - rel_docs

                doc_rel_counts = {doc: 0 for doc in all_docs}
                for r in rel_p:
                    doc_rel_counts[r.article_id] += 1

                rel_counts_per_doc = [c for c in doc_rel_counts.values() if c > 0]
                mean_p_per_doc = float(np.mean(rel_counts_per_doc)) if rel_counts_per_doc else 0.0
                med_p_per_doc = float(np.median(rel_counts_per_doc)) if rel_counts_per_doc else 0.0

                st.markdown("### 📊 Métricas Globais de Cobertura (Parágrafos e Documentos)")
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                m_c1.metric(
                    "Parágrafos Relevantes",
                    f"{len(rel_p):,}".replace(",", "."),
                    delta=f"{(len(rel_p)/total_p*100):.1f}% do total analisado" if total_p > 0 else "0.0%"
                )
                m_c2.metric(
                    "Documentos com Evidências",
                    f"{len(rel_docs):,}".replace(",", "."),
                    delta=f"{(len(rel_docs)/total_docs*100):.1f}% dos artigos" if total_docs > 0 else "0.0%"
                )
                m_c3.metric(
                    "Documentos Descartados",
                    f"{len(not_rel_docs):,}".replace(",", "."),
                    delta=f"{(len(not_rel_docs)/total_docs*100):.1f}% sem evidências" if total_docs > 0 else "0.0%",
                    delta_color="inverse"
                )
                m_c4.metric(
                    "Densidade Relevante",
                    f"{mean_p_per_doc:.1f} p/doc",
                    delta=f"Mediana: {med_p_per_doc:.1f}"
                )

                # ----------------------------------------------------
                # TABELA E GRÁFICOS POR CLASSE CONCEITUAL (0 A 5)
                # ----------------------------------------------------
                st.markdown("#### 🏷️ Distribuição por Classe Conceitual (Parágrafos vs. Documentos Únicos)")

                class_stat_rows = []
                p_c0 = len(not_rel_p)
                d_c0 = len(set(r.article_id for r in not_rel_p))
                class_stat_rows.append({
                    "Classe ID": 0,
                    "Dimensão Conceitual": CONCEPT_LABEL_NAMES[0],
                    "Parágrafos (N)": p_c0,
                    "% Parágrafos": f"{(p_c0/total_p*100):.1f}%" if total_p > 0 else "0.0%",
                    "Documentos Únicos (N)": d_c0,
                    "% Documentos": f"{(d_c0/total_docs*100):.1f}%" if total_docs > 0 else "0.0%",
                    "Probabilidade Média": "—",
                    "Probabilidade Mediana": "—",
                    "Probabilidade Máxima": "—"
                })

                probs_per_class = {c: [] for c in MULTILABEL_CLASSES}
                for c in MULTILABEL_CLASSES:
                    c_name = CONCEPT_LABEL_SHORT_NAMES[c]
                    p_matching = [r for r in classified_records if c_name in r.predicted_labels]
                    d_matching = set(r.article_id for r in p_matching)

                    c_probs = [r.predicted_probabilities.get(c_name, 0.0) for r in classified_records if r.predicted_probabilities]
                    probs_per_class[c] = c_probs

                    mean_prob = float(np.mean(c_probs)) if c_probs else 0.0
                    med_prob = float(np.median(c_probs)) if c_probs else 0.0
                    max_prob = float(np.max(c_probs)) if c_probs else 0.0

                    class_stat_rows.append({
                        "Classe ID": c,
                        "Dimensão Conceitual": CONCEPT_LABEL_NAMES[c],
                        "Parágrafos (N)": len(p_matching),
                        "% Parágrafos": f"{(len(p_matching)/total_p*100):.1f}%" if total_p > 0 else "0.0%",
                        "Documentos Únicos (N)": len(d_matching),
                        "% Documentos": f"{(len(d_matching)/total_docs*100):.1f}%" if total_docs > 0 else "0.0%",
                        "Probabilidade Média": f"{mean_prob:.4f}",
                        "Probabilidade Mediana": f"{med_prob:.4f}",
                        "Probabilidade Máxima": f"{max_prob:.4f}"
                    })

                df_class_stats = pd.DataFrame(class_stat_rows)
                st.dataframe(df_class_stats, use_container_width=True)

                col_pl1, col_pl2 = st.columns(2)
                with col_pl1:
                    df_bar_data = df_class_stats[["Dimensão Conceitual", "Parágrafos (N)", "Documentos Únicos (N)"]].melt(
                        id_vars=["Dimensão Conceitual"],
                        value_vars=["Parágrafos (N)", "Documentos Únicos (N)"],
                        var_name="Tipo de Contagem",
                        value_name="Quantidade"
                    )
                    fig_classes_bar = px.bar(
                        df_bar_data,
                        x="Dimensão Conceitual",
                        y="Quantidade",
                        color="Tipo de Contagem",
                        barmode="group",
                        title="Comparação: Parágrafos vs. Documentos Únicos por Classe Conceitual",
                        color_discrete_map={"Parágrafos (N)": "#2563eb", "Documentos Únicos (N)": "#10b981"}
                    )
                    fig_classes_bar.update_layout(xaxis_tickangle=-20)
                    st.plotly_chart(fig_classes_bar, use_container_width=True)

                with col_pl2:
                    box_probs_data = []
                    for c in MULTILABEL_CLASSES:
                        c_label = f"C{c} — {CONCEPT_LABEL_NAMES[c].split('—')[1].strip() if '—' in CONCEPT_LABEL_NAMES[c] else CONCEPT_LABEL_NAMES[c]}"
                        for p_val in probs_per_class[c][:3000]:
                            box_probs_data.append({"Classe": c_label, "Probabilidade Estimada": p_val})

                    if box_probs_data:
                        df_box_probs = pd.DataFrame(box_probs_data)
                        fig_probs_box = px.box(
                            df_box_probs,
                            x="Classe",
                            y="Probabilidade Estimada",
                            color="Classe",
                            title="Dispersão das Probabilidades Preditas por Dimensão Conceitual"
                        )
                        fig_probs_box.update_layout(showlegend=False, xaxis_tickangle=-20)
                        st.plotly_chart(fig_probs_box, use_container_width=True)

                # ----------------------------------------------------
                # DISTRIBUIÇÃO POR PROBABILIDADE MÁXIMA (PARÁGRAFOS E DOCUMENTOS)
                # ----------------------------------------------------
                st.markdown("#### 📈 Distribuição de Trabalhos e Parágrafos por Probabilidade Máxima (P_max)")

                p_max_paras = [max(r.predicted_probabilities.values()) if r.predicted_probabilities else 0.0 for r in classified_records]

                doc_pmax_map = {doc: 0.0 for doc in all_docs}
                for r in classified_records:
                    if r.predicted_probabilities:
                        curr_max = max(r.predicted_probabilities.values())
                        if curr_max > doc_pmax_map[r.article_id]:
                            doc_pmax_map[r.article_id] = curr_max

                doc_pmax_values = list(doc_pmax_map.values())

                col_hist1, col_hist2 = st.columns(2)
                with col_hist1:
                    fig_pmax_para = px.histogram(
                        p_max_paras,
                        nbins=25,
                        title="Distribuição de Parágrafos por Probabilidade Máxima P_max",
                        labels={"value": "Probabilidade Máxima do Parágrafo", "count": "Frequência de Parágrafos"},
                        color_discrete_sequence=["#3b82f6"]
                    )
                    fig_pmax_para.add_vline(x=0.50, line_dash="dash", line_color="orange", annotation_text="P=0.50")
                    fig_pmax_para.add_vline(x=0.75, line_dash="dash", line_color="green", annotation_text="P=0.75")
                    st.plotly_chart(fig_pmax_para, use_container_width=True)
                    render_interpretation_box("Histograma da probabilidade máxima de pertinência a qualquer dimensão conceituada. Parágrafos à direita de 0.50 possuem maior robustez semântica.")

                with col_hist2:
                    fig_pmax_doc = px.histogram(
                        doc_pmax_values,
                        nbins=25,
                        title="Distribuição de Documentos (Trabalhos) por Maior P_max Interno",
                        labels={"value": "Maior Probabilidade no Artigo (P_max_doc)", "count": "Frequência de Artigos"},
                        color_discrete_sequence=["#10b981"]
                    )
                    fig_pmax_doc.add_vline(x=0.50, line_dash="dash", line_color="orange", annotation_text="P=0.50")
                    fig_pmax_doc.add_vline(x=0.75, line_dash="dash", line_color="green", annotation_text="P=0.75 (Forte)")
                    st.plotly_chart(fig_pmax_doc, use_container_width=True)
                    render_interpretation_box("Distribuição dos artigos científicos pela evidência mais forte contida neles. Indica quantos trabalhos trazem evidências conceituais categóricas.")

                st.subheader(f"Parágrafos Classificados ({len(classified_records):,} Itens Processados)".replace(",", "."))

                f_c1, f_c2, f_c3 = st.columns([2, 2, 1])
                with f_c1:
                    filter_class_status = st.selectbox(
                        "Filtrar por Status de Relevância",
                        options=["Todos os Classificados", "Somente Relevantes (MODEL_RELEVANT)", "Não Relevantes (MODEL_NOT_RELEVANT)"],
                        key="filter_class_status_t6"
                    )
                with f_c2:
                    filter_doc_id = st.selectbox(
                        "Filtrar por Documento ID",
                        options=["Todos"] + sorted(list(set(r.article_id for r in classified_records))),
                        key="filter_doc_id_t6"
                    )
                with f_c3:
                    n_display_t6 = st.selectbox("Exibir na Tela", options=[25, 50, 100, 200, 500, "Todos"], index=1, key="n_disp_t6")

                view_records = classified_records
                if filter_class_status == "Somente Relevantes (MODEL_RELEVANT)":
                    view_records = [r for r in view_records if r.status == "MODEL_RELEVANT"]
                elif filter_class_status == "Não Relevantes (MODEL_NOT_RELEVANT)":
                    view_records = [r for r in view_records if r.status == "MODEL_NOT_RELEVANT"]

                if filter_doc_id != "Todos":
                    view_records = [r for r in view_records if r.article_id == filter_doc_id]

                # Conteúdo para Exportação e Download
                md_t6 = "# Corpus Final Classificado — SLD (Etapa 6)\n\n" + f"**Data de Geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n**Total Parágrafos Exibidos:** {len(view_records)}\n\n---\n\n" + "\n\n---\n\n".join(
                    f"## Parágrafo `{r.paragraph_id}` (`{r.article_id}`)\n- **Status:** `{r.status}`\n- **Similaridade Cosseno:** `{r.semantic_score or 0.0:.4f}`\n- **Classes Atribuídas:** `{', '.join(r.predicted_labels) if r.predicted_labels else 'Nenhum'}`\n- **Probabilidades:** `{json.dumps(r.predicted_probabilities)}` \n\n{r.text}"
                    for r in view_records[:1000]
                )
                df_t6 = pd.DataFrame([{
                    "paragraph_id": r.paragraph_id,
                    "article_id": r.article_id,
                    "status": r.status,
                    "semantic_score": r.semantic_score or 0.0,
                    "predicted_labels": ", ".join(r.predicted_labels) if r.predicted_labels else "Nenhum",
                    "predicted_probabilities": json.dumps(r.predicted_probabilities),
                    "text": r.text
                } for r in view_records])
                jsonl_t6 = "\n".join(json.dumps({
                    "paragraph_id": r.paragraph_id,
                    "article_id": r.article_id,
                    "status": r.status,
                    "semantic_score": float(r.semantic_score or 0.0),
                    "predicted_labels": r.predicted_labels or [],
                    "predicted_probabilities": r.predicted_probabilities or {},
                    "text": r.text
                }, ensure_ascii=False) for r in view_records)

                # ----------------------------------------------------
                # PERSISTÊNCIA FÍSICA AUTOMÁTICA NO DISCO
                # ----------------------------------------------------
                saved_class_paths = export_classified_corpus_to_disk(project, classified_records, st.session_state.run_id)
                stats_csv_p = project.classification_dir / "estatisticas_classes.csv"
                try:
                    df_class_stats.to_csv(stats_csv_p, index=False, encoding="utf-8")
                    saved_class_paths["stats_csv"] = stats_csv_p
                except Exception:
                    pass

                # Aviso visual claro com caminhos absolutos e relativos no disco
                st.success(
                    f"💾 **Dados Refinados Salvos Fisicamente no Disco!**\n\n"
                    f"Todos os registros classificados foram gravados automaticamente na pasta de saída configurada. "
                    f"Você pode acessá-los diretamente no seu computador para abrir no Excel, R, Python, SPSS, Stata ou Obsidian:\n\n"
                    f"- 📄 **Tabela Estruturada (CSV):** `{project.classification_dir / 'classified_corpus.csv'}`\n"
                    f"- ⚡ **Formato Colunar (Parquet):** `{project.classification_dir / 'classified_corpus.parquet'}`\n"
                    f"- 📦 **Formato JSON Lines (JSONL):** `{project.classification_dir / 'classified_corpus.jsonl'}`\n"
                    f"- 📝 **Documento Completo (Markdown):** `{project.classification_dir / 'classified_corpus.md'}`\n"
                    f"- 📊 **Estatísticas por Dimensão (CSV):** `{project.classification_dir / 'estatisticas_classes.csv'}`\n\n"
                    f"📍 **Pasta Local dos Arquivos:** `{project.classification_dir.resolve()}`"
                )

                col_sync_d1, col_sync_d2 = st.columns([2, 2])
                with col_sync_d1:
                    if st.button("🔄 Sincronizar e Gravar Arquivos no Disco Novamente", key="btn_sync_t6_disk", use_container_width=True):
                        export_classified_corpus_to_disk(project, classified_records, st.session_state.run_id)
                        df_class_stats.to_csv(stats_csv_p, index=False, encoding="utf-8")
                        st.toast("✓ Todos os arquivos foram sincronizados e gravados no disco com sucesso!")
                with col_sync_d2:
                    st.caption(f"📁 Localização: `{project.classification_dir}`")

                # Conteúdo para Exportação e Download
                csv_disk_file = project.classification_dir / "classified_corpus.csv"
                csv_data_bytes = csv_disk_file.read_bytes() if csv_disk_file.exists() else df_t6.to_csv(index=False, encoding="utf-8").encode("utf-8")

                jsonl_disk_file = project.classification_dir / "classified_corpus.jsonl"
                jsonl_data_bytes = jsonl_disk_file.read_bytes() if jsonl_disk_file.exists() else jsonl_t6.encode("utf-8")

                md_disk_file = project.classification_dir / "classified_corpus.md"
                md_data_str = md_disk_file.read_text(encoding="utf-8") if md_disk_file.exists() else md_t6

                stats_disk_file = project.classification_dir / "estatisticas_classes.csv"
                stats_data_bytes = stats_disk_file.read_bytes() if stats_disk_file.exists() else df_class_stats.to_csv(index=False, encoding="utf-8").encode("utf-8")

                # Botões de Download Diretos e Seção de Exportação
                st.markdown("#### 📥 Opções de Download dos Resultados")
                col_dl_md, col_dl_csv, col_dl_jsonl, col_dl_st = st.columns(4)
                col_dl_md.download_button(
                    label="📥 Baixar em Markdown (.md)",
                    data=md_data_str,
                    file_name=f"corpus_classificado_{st.session_state.run_id}.md",
                    mime="text/markdown",
                    key="btn_dl_md_t6_class",
                    use_container_width=True
                )
                col_dl_csv.download_button(
                    label="📥 Baixar Tabela em CSV (.csv)",
                    data=csv_data_bytes,
                    file_name=f"corpus_classificado_{st.session_state.run_id}.csv",
                    mime="text/csv",
                    key="btn_dl_csv_t6_class",
                    use_container_width=True
                )
                col_dl_jsonl.download_button(
                    label="📥 Baixar Dados em JSONL (.jsonl)",
                    data=jsonl_data_bytes,
                    file_name=f"corpus_classificado_{st.session_state.run_id}.jsonl",
                    mime="application/jsonlines",
                    key="btn_dl_jsonl_t6_class",
                    use_container_width=True
                )
                col_dl_st.download_button(
                    label="📥 Baixar Estatísticas por Classe (.csv)",
                    data=stats_data_bytes,
                    file_name=f"estatisticas_classes_{st.session_state.run_id}.csv",
                    mime="text/csv",
                    key="btn_dl_stats_t6_class",
                    use_container_width=True
                )

                limit_t6 = len(view_records) if n_display_t6 == "Todos" else int(n_display_t6)

                tab_table_t6, tab_cards_t6 = st.tabs(["📋 Visão em Tabela Resumida", "📄 Visão Completa dos Parágrafos Classificados"])

                with tab_table_t6:
                    st.markdown("#### Tabela de Parágrafos Classificados")
                    table_class_rows = []
                    for r in view_records[:limit_t6]:
                        max_p = max(r.predicted_probabilities.values()) if r.predicted_probabilities else 0.0
                        table_class_rows.append({
                            "ID do Parágrafo": r.paragraph_id,
                            "Documento ID": r.article_id,
                            "Status de Relevância": r.status,
                            "Score Similaridade": f"{r.semantic_score or 0.0:.4f}",
                            "Classes Atribuídas": ", ".join(r.predicted_labels) if r.predicted_labels else "Nenhum",
                            "Probabilidade Máxima": f"{max_p:.4f}",
                            "Texto (Resumo)": r.text[:140] + "..." if len(r.text) > 140 else r.text
                        })
                    st.dataframe(pd.DataFrame(table_class_rows), use_container_width=True)

                with tab_cards_t6:
                    st.markdown(f"#### Exibição dos Parágrafos Classificados (Mostrando {min(limit_t6, len(view_records))} de {len(view_records)})")
                    for idx, r in enumerate(view_records[:limit_t6], start=1):
                        status_color = "#166534" if r.status == "MODEL_RELEVANT" else "#991b1b"
                        status_bg = "#dcfce7" if r.status == "MODEL_RELEVANT" else "#fee2e2"
                        labels_str = ", ".join(r.predicted_labels) if r.predicted_labels else "Nenhuma classe atribuída"

                        st.markdown(
                            f"<div style='background-color: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid {'#16a34a' if r.status == 'MODEL_RELEVANT' else '#dc2626'}; padding: 14px 18px; margin-bottom: 12px; border-radius: 4px;'>"
                            f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>"
                            f"<strong style='color: #0f172a; font-size: 1.0rem;'>#{idx} — Parágrafo <code>{r.paragraph_id}</code> (Doc: <code>{r.article_id}</code>)</strong>"
                            f"<div>"
                            f"<span style='background-color: {status_bg}; color: {status_color}; font-weight: 700; padding: 3px 10px; border-radius: 12px; font-size: 0.85rem; margin-right: 6px;'>"
                            f"{r.status}"
                            f"</span>"
                            f"<span style='background-color: #dbeafe; color: #1e40af; font-weight: 700; padding: 3px 10px; border-radius: 12px; font-size: 0.85rem;'>"
                            f"Similaridade Cosseno: {r.semantic_score or 0.0:.4f}"
                            f"</span>"
                            f"</div>"
                            f"</div>"
                            f"<div style='margin-bottom: 8px; font-size: 0.90rem; color: #475569;'>"
                            f"<strong>Categorias Atribuídas:</strong> <code style='color: #0369a1;'>{labels_str}</code>"
                            f"</div>"
                            f"<p style='color: #334155; font-size: 0.95rem; line-height: 1.6; margin: 0px;'>{r.text}</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

    # ==========================================
    # ABA 7: ÍNDICE DE RECUPERAÇÃO DO CORPUS REFINADO
    # ==========================================
    with t7:
        render_methodology_header(
            title="7. Índice de Recuperação do Corpus Refinado",
            description=(
                "Esta etapa constrói, audita e persiste o **Índice Vetorial Local (FAISS)** e a tabela estruturada de metadados (**Parquet**) "
                "a partir exclusivamente do corpus refinado resultante da classificação por regressão logística. "
                "O índice servirá como mecanismo de recuperação de evidências para o sistema RAG conectado a modelos LLM."
            ),
            objective="Indexar de forma determinística e em lote (sem recálculo de vetores) os parágrafos relevantes (Classes 1 a 5), excluindo a Classe 0 (Não Relevante), garantindo correspondência 1:1 entre vetores e metadados.",
            method="Indexação FAISS (IndexHNSWFlat para busca aproximada ultra-rápida ou IndexFlatIP para busca exata) com métrica de produto interno sobre embeddings L2-normalizados, tabela de metadados colunar Parquet, manifesto de reprodutibilidade e empacotamento ZIP.",
            formula_latex=r"N_{\text{vetores}} = N_{\text{relevantes\_únicos}} = \text{len}(\text{metadata.parquet}) \quad \land \quad \text{Sim}(q, P_i) = \langle \mathbf{e}_q, \mathbf{e}_{P_i} \rangle",
            legend_dict={
                "N_{vetores}": "quantidade total de vetores adicionados ao índice FAISS",
                "N_{relevantes_únicos}": "total de parágrafos classificados em pelo menos uma das classes 1 a 5",
                "Sim(q, P_i)": "similaridade do cosseno calculada por produto interno de vetores normalizados L2"
            },
            interpretation="Garante a integridade matemática, rastreabilidade documental (FAISS ID ↔ paragraph_id ↔ Markdown) e reprodutibilidade do mecanismo de recuperação RAG."
        )

        render_stage_disk_loader(7, "Índice de Recuperação do Corpus Refinado", project, st.session_state)

        st.divider()

        # 1. CORPUS DISPONÍVEL PARA INDEXAÇÃO
        st.markdown("### 📊 1. Corpus Disponível para Indexação")
        classified_recs = st.session_state.classified_records or [r for r in st.session_state.corpus_records if r.status in ["MODEL_RELEVANT", "MODEL_NOT_RELEVANT"]]

        if not classified_recs:
            render_empty_state(
                title="Classificação Conceitual Não Executada",
                description="Execute a Etapa 6 (Classificação Conceitual dos Candidatos) para rotular o corpus antes de construir o índice RAG.",
                recommendation="Aguardando classificação supervisionada por regressão logística."
            )
        else:
            sub_rag_1, sub_rag_2, sub_rag_3 = st.tabs([
                "🏗️ 1. Construção, Estatísticas e Pacote RAG",
                "🔎 2. Teste Operacional e Busca Vetorial Top-k",
                "📂 3. Carregamento e Restauração de Índice"
            ])

            with sub_rag_1:
                dist_stats = compute_corpus_distribution_stats(classified_recs)

                c_st1, c_st2, c_st3, c_st4 = st.columns(4)
                c_st1.metric("Parágrafos Classificados", f"{dist_stats.total_classified:,}".replace(",", "."))
                c_st2.metric("Relevantes Únicos (Classes 1–5)", f"{dist_stats.total_unique_relevant:,}".replace(",", "."), delta=f"{dist_stats.pct_relevant:.1f}% do total")
                c_st3.metric("Não Relevantes (Classe 0)", f"{dist_stats.class_0_not_relevant:,}".replace(",", "."), delta=f"-{dist_stats.pct_class_0:.1f}% descartados")
                c_st4.metric("Ocorrências Multilabel", f"{dist_stats.total_multilabel_occurrences:,}".replace(",", "."))

                # Distribuição das Classes
                st.markdown("##### Distribuição de Ocorrências por Classe Conceitual")
                st.caption("ℹ️ *Como a classificação é multilabel, um mesmo parágrafo pode pertencer a várias classes (1 a 5). Por isso, a soma das ocorrências pode ser superior ao número de parágrafos únicos.*")

                col_tb_cls, col_gr_cls = st.columns([1, 1])

                class_table_rows = [
                    {"Código": "Classe 0", "Categoria": "Não Relevante", "Ocorrências": dist_stats.class_0_not_relevant, "Status": "Excluída do Índice", "% do Corpus": f"{dist_stats.pct_class_0:.2f}%"},
                    {"Código": "Classe 1", "Categoria": "Definição ou Conceituação", "Ocorrências": dist_stats.class_1_definition, "Status": "Indexada", "% do Corpus": f"{(dist_stats.class_1_definition / max(1, dist_stats.total_classified) * 100):.2f}%"},
                    {"Código": "Classe 2", "Categoria": "Fator Determinante", "Ocorrências": dist_stats.class_2_determinant, "Status": "Indexada", "% do Corpus": f"{(dist_stats.class_2_determinant / max(1, dist_stats.total_classified) * 100):.2f}%"},
                    {"Código": "Classe 3", "Categoria": "Tipo ou Dimensão", "Ocorrências": dist_stats.class_3_type_dimension, "Status": "Indexada", "% do Corpus": f"{(dist_stats.class_3_type_dimension / max(1, dist_stats.total_classified) * 100):.2f}%"},
                    {"Código": "Classe 4", "Categoria": "Relação Causal", "Ocorrências": dist_stats.class_4_causal_relation, "Status": "Indexada", "% do Corpus": f"{(dist_stats.class_4_causal_relation / max(1, dist_stats.total_classified) * 100):.2f}%"},
                    {"Código": "Classe 5", "Categoria": "Característica ou Propriedade", "Ocorrências": dist_stats.class_5_property, "Status": "Indexada", "% do Corpus": f"{(dist_stats.class_5_property / max(1, dist_stats.total_classified) * 100):.2f}%"},
                ]
                with col_tb_cls:
                    st.dataframe(pd.DataFrame(class_table_rows), use_container_width=True, hide_index=True)

                with col_gr_cls:
                    df_plot_cls = pd.DataFrame([
                        {"Classe": "Classe 1: Definição", "Ocorrências": dist_stats.class_1_definition, "Cor": "#16a34a"},
                        {"Classe": "Classe 2: Fator", "Ocorrências": dist_stats.class_2_determinant, "Cor": "#2563eb"},
                        {"Classe": "Classe 3: Tipo/Dimensão", "Ocorrências": dist_stats.class_3_type_dimension, "Cor": "#9333ea"},
                        {"Classe": "Classe 4: Causal", "Ocorrências": dist_stats.class_4_causal_relation, "Cor": "#ea580c"},
                        {"Classe": "Classe 5: Propriedade", "Ocorrências": dist_stats.class_5_property, "Cor": "#0d9488"},
                        {"Classe": "Classe 0: Não Relevante", "Ocorrências": dist_stats.class_0_not_relevant, "Cor": "#94a3b8"},
                    ])
                    fig_cls = px.bar(
                        df_plot_cls,
                        x="Classe",
                        y="Ocorrências",
                        color="Classe",
                        color_discrete_sequence=["#16a34a", "#2563eb", "#9333ea", "#ea580c", "#0d9488", "#94a3b8"],
                        title="Distribuição das Classes no Corpus",
                    )
                    fig_cls.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20), height=260)
                    st.plotly_chart(fig_cls, use_container_width=True)

                st.divider()

                # 2. CONFIGURAÇÕES DO ÍNDICE FAISS
                st.markdown("### ⚙️ Configurações de Construção do Índice FAISS")
                col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
                with col_cfg1:
                    idx_type = st.radio(
                        "Tipo de Indexação FAISS:",
                        options=["HNSW", "FlatIP"],
                        format_func=lambda x: "IndexHNSWFlat — Busca Aproximada Rápida (Recomendado para grandes volumes)" if x == "HNSW" else "IndexFlatIP — Busca Exata (Benchmark / Validação)",
                        index=0,
                        help="HNSW constrói um grafo de vizinhança hierárquico extremamente rápido. FlatIP realiza força-bruta exata por produto interno.",
                        key="idx_type_t7"
                    )
                with col_cfg2:
                    idx_version = st.text_input("Versão do Índice:", value="v001", help="Identificador de versão para reprodutibilidade.", key="idx_version_t7")
                    batch_size_idx = st.selectbox("Batch Size de Inserção:", options=[2048, 4096, 8192, 16384], index=2, help="Lote de vetores transferidos à memória do FAISS.", key="batch_size_idx_t7")
                with col_cfg3:
                    emb_dim = st.session_state.embeddings_matrix.shape[1] if st.session_state.embeddings_matrix is not None else 384
                    st.text_input("Dimensão dos Embeddings:", value=f"{emb_dim} dimensões", disabled=True, key="emb_dim_t7")
                    st.text_input("Métrica de Similaridade:", value="Produto Interno (Cosine Similarity)", disabled=True, key="emb_metric_t7")

                with st.expander("🛠️ Parâmetros Avançados do Grafo HNSW (Opcional)", expanded=False):
                    col_adv1, col_adv2, col_adv3 = st.columns(3)
                    with col_adv1:
                        hnsw_m = st.number_input("Conexões por Vértice (M):", min_value=8, max_value=128, value=32, step=8, help="Número de links bidirecionais por nó no grafo HNSW.", key="hnsw_m_t7")
                    with col_adv2:
                        hnsw_ef_c = st.number_input("efConstruction:", min_value=16, max_value=256, value=64, step=16, help="Tamanho da fila durante a construção do grafo.", key="hnsw_ef_c_t7")
                    with col_adv3:
                        hnsw_ef_s = st.number_input("efSearch:", min_value=16, max_value=256, value=64, step=16, help="Tamanho da fila durante as buscas de vizinhos.", key="hnsw_ef_s_t7")

                # 3. CONSTRUÇÃO DO ÍNDICE
                if st.button("🚀 Construir / Atualizar Índice de Recuperação do Corpus Refinado", type="primary", key="btn_build_rag_index_t7"):
                    if dist_stats.total_unique_relevant == 0:
                        st.error("Não há parágrafos classificados como relevantes (Classes 1 a 5) para indexação.")
                    elif st.session_state.embeddings_matrix is None:
                        st.error("Matriz de embeddings não encontrada na sessão.")
                    else:
                        target_out_dir = Path(st.session_state.get("selected_output_dir", DEFAULT_OUTPUT_DIR))
                        rag_cfg = RAGIndexConfig(
                            index_type=idx_type,
                            dimension=emb_dim,
                            M=hnsw_m if idx_type == "HNSW" else 32,
                            efConstruction=hnsw_ef_c if idx_type == "HNSW" else 64,
                            efSearch=hnsw_ef_s if idx_type == "HNSW" else 64,
                            index_batch_size=batch_size_idx,
                            normalize_embeddings=True
                        )
                        builder = RAGIndexBuilder(output_dir=target_out_dir, config=rag_cfg)

                        tracker = ProgressTracker(
                            title="Construção do Índice de Recuperação FAISS + Parquet",
                            total=dist_stats.total_unique_relevant,
                            steps=["Filtrar e Deduplicar Corpus Refinado", "Inserir Vetores no FAISS em Lote", "Gerar Metadados Parquet", "Validar Integridade e Empacotar ZIP"],
                            update_interval=1
                        )
                        tracker.set_step(0, "Filtrando parágrafos relevantes...")

                        def idx_progress_cb(processed, total, msg):
                            tracker.set_step(1, msg)
                            tracker.update(processed=processed, current_item=msg, step_processed=processed, step_total=total)

                        try:
                            tracker.set_step(1, "Inserindo vetores no índice FAISS...")
                            faiss_p, parquet_p, manifest_p, zip_p, idx_stats, manifest_obj = builder.build(
                                corpus_records=classified_recs,
                                embeddings_matrix=st.session_state.embeddings_matrix,
                                all_corpus_records=st.session_state.corpus_records,
                                total_original_articles=len(set(r.article_id for r in st.session_state.corpus_records)),
                                embedding_model_name=st.session_state.config.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
                                index_version=idx_version,
                                progress_callback=idx_progress_cb
                            )

                            st.session_state.rag_index_stats = idx_stats
                            st.session_state.rag_index_manifest = manifest_obj
                            st.session_state.rag_index_zip_path = str(zip_p)
                            st.session_state.rag_retriever = None

                            if 7 not in st.session_state.completed_steps:
                                st.session_state.completed_steps.append(7)

                            save_full_session_state(project, st.session_state)
                            import gc; gc.collect()

                            tracker.complete(message=f"✓ Índice RAG {idx_version} construído com sucesso! {idx_stats.total_vectors:,} vetores indexados.", show_balloons=True)
                            st.toast("✓ Índice de Recuperação do Corpus Refinado construído com sucesso!")
                            st.balloons()
                            st.rerun()

                        except Exception as err:
                            st.error(f"Erro na construção do índice RAG: {err}")

                # 4. EXIBIÇÃO DE ESTATÍSTICAS E PAINEL PÓS-CONSTRUÇÃO
                if st.session_state.rag_index_stats is not None:
                    st.divider()
                    st.markdown("### 📈 Estatísticas do Índice de Recuperação")
                    stats_obj = st.session_state.rag_index_stats
                    manifest_obj = st.session_state.rag_index_manifest

                    render_completion_panel(
                        title=f"Índice de Recuperação RAG ({manifest_obj.index_version if manifest_obj else 'v001'}) Operacional",
                        metrics={
                            "Total de Vetores Indexados": f"{stats_obj.total_vectors:,}".replace(",", "."),
                            "Parágrafos Únicos": f"{stats_obj.unique_paragraphs:,}".replace(",", "."),
                            "Documentos Representados": f"{stats_obj.represented_documents:,}".replace(",", "."),
                            "Dimensão dos Embeddings": f"{stats_obj.embedding_dimension}D",
                            "Tipo de Índice FAISS": stats_obj.index_type,
                            "Métrica Utilizada": stats_obj.metric,
                            "Tamanho FAISS em Disco": f"{stats_obj.faiss_file_size_bytes / (1024*1024):.2f} MB",
                            "Tamanho Parquet": f"{stats_obj.parquet_file_size_bytes / (1024*1024):.2f} MB",
                            "Tamanho Pacote ZIP": f"{stats_obj.zip_file_size_bytes / (1024*1024):.2f} MB",
                            "Tempo de Construção": f"{stats_obj.build_duration_sec:.2f}s"
                        }
                    )

                    if stats_obj.coverage:
                        cov = stats_obj.coverage
                        st.markdown("#### 📚 Cobertura Documental e Densidade")
                        col_cv1, col_cv2, col_cv3, col_cv4, col_cv5 = st.columns(5)
                        col_cv1.metric("Artigos no Corpus", f"{cov.total_original_articles}")
                        col_cv2.metric("Artigos no Índice", f"{cov.indexed_articles}", delta=f"{cov.pct_articles_represented:.1f}% representados")
                        col_cv3.metric("Média Parágrafos/Artigo", f"{cov.mean_paragraphs_per_article:.2f}")
                        col_cv4.metric("Mediana Parágrafos/Artigo", f"{cov.median_paragraphs_per_article:.1f}")
                        col_cv5.metric("Min / Max por Artigo", f"{cov.min_paragraphs_per_article} / {cov.max_paragraphs_per_article}")

                    # 5. INTEGRIDADE, AUDITORIA E DOWNLOAD
                    st.divider()
                    st.markdown("### 🔒 Integridade do Pacote e Opções de Download")

                    col_chk_b1, col_chk_b2, col_chk_b3, col_chk_b4 = st.columns(4)
                    col_chk_b1.success("✓ Índice FAISS Validado (ntotal)")
                    col_chk_b2.success("✓ Metadados Parquet Validados")
                    col_chk_b3.success("✓ Mapeamento de IDs Validado")
                    col_chk_b4.success("✓ Hashes SHA-256 Registrados")

                    if manifest_obj and manifest_obj.checksums:
                        with st.expander("🔍 Visualizar Assinaturas Criptográficas SHA-256", expanded=False):
                            for f_name, f_hash in manifest_obj.checksums.items():
                                st.code(f"{f_hash}  {f_name}", language="text")

                    zip_path_str = st.session_state.rag_index_zip_path
                    target_out_dir = Path(st.session_state.get("selected_output_dir", DEFAULT_OUTPUT_DIR)) / "rag_index"
                    
                    st.info(f"📁 **Diretório dos Artefatos em Disco:** `{target_out_dir.resolve()}`")

                    if zip_path_str and Path(zip_path_str).exists():
                        zip_file_p = Path(zip_path_str)
                        zip_size_mb = zip_file_p.stat().st_size / (1024 * 1024)

                        col_dl_main_1, col_dl_main_2 = st.columns([3, 1])
                        with col_dl_main_1:
                            st.caption(f"Pacote consolidado: **{zip_file_p.name}** ({zip_size_mb:.2f} MB)")
                            if zip_size_mb < 50.0:
                                with open(zip_file_p, "rb") as zf:
                                    st.download_button(
                                        label=f"📦 Baixar Pacote Completo (.zip) — {zip_size_mb:.1f} MB",
                                        data=zf.read(),
                                        file_name=zip_file_p.name,
                                        mime="application/zip",
                                        type="primary",
                                        key="btn_dl_rag_zip_main",
                                        use_container_width=True,
                                        help="Contém o índice corpus_refinado.faiss, metadata.parquet, manifest.json, index_report.md e README.md."
                                    )
                            else:
                                if st.checkbox("📥 Carregar Pacote ZIP para Download no Navegador", key="chk_load_rag_zip"):
                                    with open(zip_file_p, "rb") as zf:
                                        st.download_button(
                                            label=f"📦 Clique para Salvar {zip_file_p.name} ({zip_size_mb:.1f} MB)",
                                            data=zf.read(),
                                            file_name=zip_file_p.name,
                                            mime="application/zip",
                                            type="primary",
                                            key="btn_dl_rag_zip_main_loaded",
                                            use_container_width=True
                                        )

                    if target_out_dir.exists():
                        with st.expander("📥 Arquivos Individuais do Índice", expanded=False):
                            col_dl_f, col_dl_p, col_dl_m, col_dl_r = st.columns(4)

                            f_faiss = target_out_dir / "corpus_refinado.faiss"
                            f_pq = target_out_dir / "metadata.parquet"
                            f_mani = target_out_dir / "manifest.json"
                            f_rep = target_out_dir / "index_report.md"

                            if f_mani.exists():
                                with open(f_mani, "r", encoding="utf-8") as f:
                                    col_dl_m.download_button("📥 Baixar manifest.json", data=f.read(), file_name="manifest.json", mime="application/json", key="dl_f_mani", use_container_width=True)
                            if f_rep.exists():
                                with open(f_rep, "r", encoding="utf-8") as f:
                                    col_dl_r.download_button("📥 Baixar index_report.md", data=f.read(), file_name="index_report.md", mime="text/markdown", key="dl_f_rep", use_container_width=True)

                            if f_faiss.exists():
                                sz_faiss_mb = f_faiss.stat().st_size / (1024 * 1024)
                                if sz_faiss_mb < 30.0:
                                    with open(f_faiss, "rb") as f:
                                        col_dl_f.download_button(f"📥 Baixar .faiss ({sz_faiss_mb:.1f}MB)", data=f.read(), file_name="corpus_refinado.faiss", key="dl_f_faiss", use_container_width=True)
                                else:
                                    col_dl_f.caption(f"💾 `corpus_refinado.faiss` ({sz_faiss_mb:.1f} MB) salvo no disco")

                            if f_pq.exists():
                                sz_pq_mb = f_pq.stat().st_size / (1024 * 1024)
                                if sz_pq_mb < 30.0:
                                    with open(f_pq, "rb") as f:
                                        col_dl_p.download_button(f"📥 Baixar .parquet ({sz_pq_mb:.1f}MB)", data=f.read(), file_name="metadata.parquet", key="dl_f_pq", use_container_width=True)
                                else:
                                    col_dl_p.caption(f"💾 `metadata.parquet` ({sz_pq_mb:.1f} MB) salvo no disco")

            # ----------------------------------------------------
            # SUB-ABA 2: TESTE OPERACIONAL E BUSCA VETORIAL TOP-K
            # ----------------------------------------------------
            with sub_rag_2:
                st.subheader("🔎 Teste Operacional do Índice RAG (Busca Vetorial Top-k)")
                st.caption("Execute consultas em linguagem natural diretamente no índice FAISS sobre os metadados para avaliar a proximidade semântica antes de acionar a LLM.")

                target_out_dir = Path(st.session_state.get("selected_output_dir", DEFAULT_OUTPUT_DIR)) / "rag_index"

                col_q_txt, col_q_cls = st.columns([3, 2])
                with col_q_txt:
                    test_query_str = st.text_input("Consulta Textual de Teste:", value="definição de governança e mecanismos adaptativos", help="Texto de entrada a ser vetorizado pelo mesmo modelo de embeddings.", key="test_query_str_t7")
                with col_q_cls:
                    filter_classes_req = st.multiselect(
                        "Filtrar por Classes Específicas (Opcional):",
                        options=["1", "2", "3", "4", "5"],
                        format_func=lambda x: {
                            "1": "Classe 1 — Definição",
                            "2": "Classe 2 — Fator Determinante",
                            "3": "Classe 3 — Tipo/Dimensão",
                            "4": "Classe 4 — Relação Causal",
                            "5": "Classe 5 — Propriedade",
                        }[x],
                        help="Exibe apenas resultados que contenham as classes conceituais selecionadas.",
                        key="filter_classes_req_t7"
                    )

                col_sim_range, col_q_k = st.columns([3, 1])
                with col_sim_range:
                    sim_range = st.slider(
                        "Faixa de Similaridade Cosseno [Limiar Mínimo (θ_min), Limiar Máximo (θ_max)]:",
                        min_value=0.0,
                        max_value=1.0,
                        value=(0.00, 1.00),
                        step=0.01,
                        format="%.2f",
                        help="Filtra os resultados recuperados pela proximidade vetorial. Apenas parágrafos com score dentro de [θ_min, θ_max] serão retornados.",
                        key="rag_sim_range_t7"
                    )
                    min_sim_val, max_sim_val = sim_range
                with col_q_k:
                    test_top_k = st.selectbox("Top-k Resultados:", options=[5, 10, 20, 50, 100, 200], index=0, key="test_top_k_t7")

                if st.button("🔍 Executar Consulta Vetorial de Teste", type="primary", key="btn_run_query_rag_t7"):
                    if not test_query_str.strip():
                        st.warning("Digite uma consulta textual para testar.")
                    else:
                        emb_srv = EmbeddingService(model_name=st.session_state.config.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
                        if st.session_state.rag_retriever is None and target_out_dir.exists():
                            st.session_state.rag_retriever = RAGIndexRetriever()
                            st.session_state.rag_retriever.load_from_dir(target_out_dir)

                        if st.session_state.rag_retriever is None or st.session_state.rag_retriever.faiss_index is None:
                            st.error("O índice RAG ainda não foi construído ou carregado na sessão. Construa o índice na Sub-aba 1.")
                        else:
                            q_results = st.session_state.rag_retriever.query(
                                query_text=test_query_str,
                                embedding_service=emb_srv,
                                top_k=test_top_k,
                                required_classes=[f"class_{c}" for c in filter_classes_req] if filter_classes_req else None,
                                min_score=min_sim_val,
                                max_score=max_sim_val
                            )
                            st.session_state.rag_query_results = q_results

                if getattr(st.session_state, "rag_query_results", None):
                    q_res = st.session_state.rag_query_results
                    st.markdown(f"#### Resultados da Busca Vetorial ({len(q_res)} parágrafos recuperados)")
                    st.caption(f"Filtro aplicado: Similaridade Cosseno entre **{min_sim_val:.2f}** e **{max_sim_val:.2f}** | Top-{test_top_k} max.")

                    # Exportações em CSV e Markdown
                    df_q_export = pd.DataFrame([{
                        "rank": r.rank,
                        "score": round(r.score, 4),
                        "paragraph_id": r.paragraph_id,
                        "article_id": r.article_id,
                        "classes": ", ".join(r.classes),
                        "faiss_id": r.faiss_id,
                        "text": r.text
                    } for r in q_res])

                    md_q_export = (
                        f"# Resultados da Busca Vetorial no Índice RAG\n\n"
                        f"- **Consulta:** `{test_query_str}`\n"
                        f"- **Data/Hora:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n"
                        f"- **Limiar Mínimo de Similaridade:** `{min_sim_val:.2f}`\n"
                        f"- **Limiar Máximo de Similaridade:** `{max_sim_val:.2f}`\n"
                        f"- **Total de Resultados (Top-K):** `{len(q_res)}`\n"
                        f"- **Filtros de Classe:** `{', '.join(filter_classes_req) if filter_classes_req else 'Nenhum'}`\n\n---\n\n"
                        + "\n\n---\n\n".join(
                            f"### #{r.rank} — Parágrafo `{r.paragraph_id}` (`{r.article_id}`)\n"
                            f"- **Similaridade Cosseno:** `{r.score:.4f}`\n"
                            f"- **Classes:** `{', '.join(r.classes)}`\n"
                            f"- **FAISS ID:** `{r.faiss_id}`\n\n{r.text}"
                            for r in q_res
                        )
                    )

                    col_dl_q1, col_dl_q2 = st.columns(2)
                    col_dl_q1.download_button(
                        label="📥 Baixar Resultados da Busca em CSV (.csv)",
                        data=df_q_export.to_csv(index=False, encoding="utf-8").encode("utf-8"),
                        file_name=f"busca_rag_{st.session_state.run_id}.csv",
                        mime="text/csv",
                        key="btn_dl_rag_search_csv",
                        use_container_width=True
                    )
                    col_dl_q2.download_button(
                        label="📥 Baixar Resultados da Busca em Markdown (.md)",
                        data=md_q_export,
                        file_name=f"busca_rag_{st.session_state.run_id}.md",
                        mime="text/markdown",
                        key="btn_dl_rag_search_md",
                        use_container_width=True
                    )

                    st.divider()
                    for r in q_res:
                        classes_badges = " ".join([f"<span style='background-color: #dbeafe; color: #1e40af; font-size: 0.8rem; font-weight: 600; padding: 2px 8px; border-radius: 8px; margin-right: 4px;'>{c}</span>" for c in r.classes])
                        st.markdown(
                            f"<div style='background-color: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #2563eb; padding: 12px 16px; margin-bottom: 10px; border-radius: 4px;'>"
                            f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;'>"
                            f"<strong>#{r.rank} — Parágrafo <code>{r.paragraph_id}</code> (Doc: <code>{r.article_id}</code> | FAISS ID: <code>{r.faiss_id}</code>)</strong>"
                            f"<span style='background-color: #dcfce7; color: #166534; font-weight: 700; padding: 2px 8px; border-radius: 8px; font-size: 0.85rem;'>Similaridade: {r.score:.4f}</span>"
                            f"</div>"
                            f"<div style='margin-bottom: 6px;'><strong>Classes:</strong> {classes_badges if classes_badges else '<em>Nenhuma</em>'}</div>"
                            f"<p style='color: #334155; font-size: 0.95rem; line-height: 1.5; margin: 0px;'>{r.text}</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

            # ----------------------------------------------------
            # SUB-ABA 3: CARREGAMENTO DE ÍNDICE EXISTENTE
            # ----------------------------------------------------
            with sub_rag_3:
                st.subheader("📂 Carregar Índice RAG Existente do Disco ou Arquivo ZIP")
                col_load_dir, col_load_zip = st.columns(2)
                with col_load_dir:
                    st.markdown("##### Carregar do Diretório do Projeto")
                    if st.button("📁 Carregar de output/rag_index/", key="btn_load_rag_dir_t7"):
                        target_dir = Path(st.session_state.get("selected_output_dir", DEFAULT_OUTPUT_DIR)) / "rag_index"
                        if not (target_dir / "corpus_refinado.faiss").exists():
                            st.error(f"Nenhum índice encontrado em {target_dir}")
                        else:
                            try:
                                retr = RAGIndexRetriever()
                                retr.load_from_dir(target_dir)
                                st.session_state.rag_retriever = retr
                                if retr.manifest:
                                    st.session_state.rag_index_manifest = retr.manifest
                                st.success(f"✓ Índice FAISS ({retr.faiss_index.ntotal:,} vetores) carregado com sucesso!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Erro ao carregar: {e}")

                with col_load_zip:
                    st.markdown("##### Carregar a partir de arquivo ZIP")
                    uploaded_zip = st.file_uploader("Enviar pacote rag_index_*.zip", type=["zip"], key="upl_rag_zip_t7")
                    if uploaded_zip is not None:
                        if st.button("📦 Descompactar e Carregar Índice", key="btn_unpack_rag_zip_t7"):
                            try:
                                retr = RAGIndexRetriever()
                                target_dir = Path(st.session_state.get("selected_output_dir", DEFAULT_OUTPUT_DIR)) / "rag_index"
                                retr.load_from_zip(uploaded_zip, extract_to_dir=target_dir)
                                st.session_state.rag_retriever = retr
                                if retr.manifest:
                                    st.session_state.rag_index_manifest = retr.manifest
                                st.success(f"✓ Índice FAISS ({retr.faiss_index.ntotal:,} vetores) descompactado e carregado com sucesso!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Erro ao descompactar ZIP: {e}")

    # ==========================================
    # ABA 8: CORPUS FINAL E ANÁLISE LLM
    # ==========================================
    with t8:
        render_methodology_header(
            title="8. Corpus Final e Análise LLM",
            description=(
                "Esta etapa aplica um modelo de linguagem local ao corpus previamente selecionado pelo pipeline semântico e supervisionado. "
                "O modelo é utilizado para extração, normalização e refinamento estruturado do conteúdo conceitual identificado nos parágrafos."
            ),
            objective="Extração estruturada e normalização de conceitos aplicada exclusivamente sobre o Corpus Final.",
            method="Inspeção individualizada por parágrafo (1 paragraph_id = 1 chamada LLM) com controle de carga por requisições por minuto (RPM), saída JSON validada por schema e verificação de evidência textual.",
            formula_latex=r"C_{\text{final}} = \{ P_i \mid \text{Relevant}(P_i) = 1 \} \quad \land \quad \Delta t_{\min} = \frac{60}{\text{RPM}}",
            legend_dict={
                "C_{final}": "corpus final enviado ao modelo de linguagem",
                "P_i": "parágrafo i do corpus",
                "Relevant(P_i)": "resultado da etapa supervisionada de relevância",
                "RPM": "requisições por minuto configuradas",
                r"\Delta t_{min}": "intervalo mínimo entre requisições em segundos"
            },
            interpretation="O modelo LLM local atua exclusivamente como extrator conceitual sobre o corpus filtrado, com 0% de contaminação por parágrafos irrelevantes."
        )

        render_stage_disk_loader(8, "Extração Conceitual e Validação de Evidências (LLM)", project, st.session_state)

        st.divider()
        ollama_url = st.text_input("Endpoint do Ollama", value=st.session_state.config.get("ollama_url", DEFAULT_OLLAMA_URL), help="URL base da API do servidor Ollama local.")
        st.session_state.config["ollama_url"] = ollama_url

        provider = OllamaProvider(
            model_name=st.session_state.config.get("llm_model", "qwen2.5:7b"),
            base_url=ollama_url
        )
        is_ollama_online = provider.check_connection()
        installed_models = provider.list_models() if is_ollama_online else []

        if not is_ollama_online:
            render_methodological_alert(f"Servidor Ollama inacessível em `{ollama_url}`. Certifique-se de que o servidor está em execução no computador.", alert_type="error")
        elif not installed_models:
            render_methodological_alert(
                "Nenhum modelo Ollama disponível. Execute no terminal: `ollama list` e `ollama pull <modelo>`",
                alert_type="warning"
            )

        with st.expander("Parâmetros do Modelo de Linguagem e Controle de Carga", expanded=True):
            col_lm1, col_lm2 = st.columns(2)
            with col_lm1:
                st.markdown("#### Configurações Metodológicas")
                current_model = st.session_state.config.get("llm_model", installed_models[0] if installed_models else "qwen2.5:7b")
                model_idx = installed_models.index(current_model) if (installed_models and current_model in installed_models) else 0

                llm_model_choice = st.selectbox(
                    "Modelo LLM Selecionado",
                    options=installed_models if installed_models else [current_model],
                    index=model_idx,
                    help="Modelo de linguagem de grande porte baixado localmente no Ollama."
                )
                st.session_state.config["llm_model"] = llm_model_choice

                temp_choice = st.slider("Temperatura", 0.0, 1.0, 0.0, 0.05, help="Valores baixos (0.0) favorecem respostas mais determinísticas.")
                st.session_state.config["temperature"] = temp_choice

                seed_choice = st.number_input("Semente (Seed)", value=42, step=1, help="Semente determinística para reprodutibilidade.")
                st.session_state.config["seed"] = seed_choice

            with col_lm2:
                st.markdown("#### Configurações de Desempenho / Rate Limit")
                rpm_choice_str = st.selectbox(
                    "Requisições por Minuto (RPM)",
                    options=["5", "10", "15", "20", "30", "60", "Sem limite"],
                    index=1,
                    help="Define o número máximo de parágrafos enviados ao modelo por minuto para evitar sobrecarga de hardware."
                )
                rpm_val = 0.0 if rpm_choice_str == "Sem limite" else float(rpm_choice_str)
                st.session_state.config["rpm_limit"] = rpm_val

                if rpm_val > 0:
                    st.caption(f"Intervalo mínimo entre requisições: `Δt_min = {60.0/rpm_val:.1f}s`")
                else:
                    st.warning("⚠️ Execução sem limitação de carga pode aumentar o consumo de CPU/GPU e memória.")

                num_ctx_choice = st.selectbox("Janela de Contexto (num_ctx)", options=[1024, 2048, 4096], index=1)
                st.session_state.config["num_ctx"] = num_ctx_choice

                num_predict_choice = st.number_input("Max Output Tokens (num_predict)", min_value=64, max_value=512, value=256, step=16)
                st.session_state.config["num_predict"] = num_predict_choice

        # Dashboard do Corpus Final
        st.divider()
        n_init = len(st.session_state.corpus_records)
        final_corpus = [
            r for r in st.session_state.corpus_records
            if r.status in ["FINAL_CORPUS", "MODEL_RELEVANT"] or (r.predicted_labels and "0 — Não relevante" not in r.predicted_labels and "not_relevant" not in r.predicted_labels)
        ]
        n_final = len(final_corpus)
        red_pct = (1.0 - (n_final / n_init)) * 100.0 if n_init > 0 else 0.0

        extraction_service = LLMExtractionService(
            llm_provider=provider if is_ollama_online else MockLLMProvider(),
            llm_dir=st.session_state.run_dirs["llm"],
            run_id=st.session_state.run_id,
            rpm_limit=st.session_state.config.get("rpm_limit", 10.0)
        )

        all_llm_res = list(extraction_service.disk_cache.values())
        refined_corpus = extraction_service.get_refined_corpus(require_valid_evidence=True)

        n_proc = len(all_llm_res)
        n_valid_resp = sum(1 for r in all_llm_res if r.schema_valid)
        total_evs = sum(len(r.llm_output.evidence) for r in all_llm_res)
        valid_evs = sum(1 for r in all_llm_res if r.evidence_valid and r.llm_output.evidence)
        evr_rate = (valid_evs / total_evs * 100.0) if total_evs > 0 else 0.0
        pvr_rate = (valid_evs / n_proc * 100.0) if n_proc > 0 else 0.0
        n_failed = sum(1 for r in all_llm_res if r.status == "failed")
        n_pending = max(0, n_final - n_proc)

        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Corpus Final", f"{n_final:,}".replace(",", "."))
        c2.metric("Processados", f"{n_proc:,}".replace(",", "."))
        c3.metric("Válidos", f"{n_valid_resp:,}".replace(",", "."))
        c4.metric("Evidências Válidas", f"{valid_evs:,}".replace(",", "."))
        c5.metric("EVR (% Evidências)", f"{evr_rate:.1f}%")
        c6.metric("PVR (% Parágrafos)", f"{pvr_rate:.1f}%")
        c7.metric("Pendentes", f"{n_pending:,}".replace(",", "."))

        st.divider()
        col_run, col_stop = st.columns([3, 1])
        with col_run:
            start_btn = st.button("Iniciar / Continuar Extração no Corpus Final", type="primary", disabled=not is_ollama_online or n_final == 0)
        with col_stop:
            stop_btn = st.button("Parar após o item atual")

        if stop_btn:
            extraction_service.request_stop()
            st.warning("Solicitação de interrupção enviada! O processamento será concluído com segurança após o parágrafo atual.")

        if start_btn:
            tracker = ProgressTracker(title="Processando Fila do Corpus Final via LLM Local", total=n_final, update_interval=5)
            llm_opts = {
                "model_name": st.session_state.config["llm_model"],
                "temperature": st.session_state.config["temperature"],
                "num_ctx": st.session_state.config["num_ctx"],
                "num_predict": st.session_state.config["num_predict"],
                "seed": st.session_state.config["seed"],
            }

            def on_progress(idx: int, total: int, result_item: LLMParagraphResult, meta: Dict[str, Any]):
                tracker.update(processed=idx, current_item=result_item.paragraph_id)

            llm_results = extraction_service.process_corpus(
                paragraphs=final_corpus,
                options=llm_opts,
                progress_callback=on_progress
            )

            st.session_state.llm_results = llm_results
            if 8 not in st.session_state.completed_steps:
                st.session_state.completed_steps.append(8)

            save_full_session_state(project, st.session_state)
            import gc; gc.collect()

            tracker.complete(message=f"Extração conceitual concluída: {len(llm_results):,} parágrafos analisados.")
            st.toast(f"✓ Extração conceitual concluída: {len(llm_results):,} parágrafos analisados.")
            st.balloons()
            st.rerun()

        if all_llm_res:
            st.divider()
            st.subheader(f"Corpus Refinado ({len(refined_corpus):,} Registros Estruturados Válidos)".replace(",", "."))

            # Filtros Rápidos
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                filter_doc = st.selectbox("Filtrar por Documento ID", options=["Todos"] + sorted(list(set(r.article_id for r in refined_corpus))), key="filter_doc_t8")
            with f_col2:
                filter_ev = st.selectbox("Filtrar por Evidência Válida", options=["Todas", "Somente Evidência Válida (EVR)", "Sem Evidência Válida"], key="filter_ev_t8")
            with f_col3:
                filter_model = st.selectbox("Filtrar por Modelo Executado", options=["Todos"] + sorted(list(set(r.llm_model for r in all_llm_res))), key="filter_model_t8")

            view_refined = refined_corpus
            if filter_doc != "Todos":
                view_refined = [r for r in view_refined if r.article_id == filter_doc]
            if filter_ev == "Somente Evidência Válida (EVR)":
                view_refined = [r for r in view_refined if r.evidence_valid]
            elif filter_ev == "Sem Evidência Válida":
                view_refined = [r for r in view_refined if not r.evidence_valid]
            if filter_model != "Todos":
                view_refined = [r for r in view_refined if r.llm_model == filter_model]

            st.markdown("#### Tabela do Corpus Refinado")
            refined_table_data = []
            for r in view_refined[:100]:
                out = r.llm_output
                refined_table_data.append({
                    "Paragraph ID": r.paragraph_id,
                    "Documento": r.article_id,
                    "Conceitos Extraídos": "; ".join(out.concepts) if out.concepts else "-",
                    "Definições": "; ".join(out.definitions) if out.definitions else "-",
                    "Fatores Determinantes": "; ".join(out.determinants) if out.determinants else "-",
                    "Tipos / Dimensões": "; ".join(out.dimensions) if out.dimensions else "-",
                    "Relações Causais": "; ".join(out.causal_relations) if out.causal_relations else "-",
                    "Características": "; ".join(out.properties) if out.properties else "-",
                    "Evidência Textual": "; ".join(out.evidence) if out.evidence else "-",
                    "Evidência Válida": "✓ Sim" if r.evidence_valid else "✕ Não",
                    "Modelo": r.llm_model
                })
            st.dataframe(pd.DataFrame(refined_table_data), use_container_width=True)

            with st.expander("🔍 Detalhes Individuais do Parágrafo Selecionado", expanded=False):
                selected_p_id = st.selectbox("Selecione um Parágrafo para Inspeção", options=[r.paragraph_id for r in view_refined], key="selected_p_id_t8")
                target_r = next((r for r in view_refined if r.paragraph_id == selected_p_id), None)
                if target_r:
                    st.markdown(f"### Parágrafo `{target_r.paragraph_id}` (`{target_r.article_id}`)")
                    st.caption(f"Modelo: `{target_r.llm_model}` | Evidência Válida: `{'✓ SIM' if target_r.evidence_valid else '✕ NÃO'}` | Tempo: `{target_r.processing_time:.2f}s`")
                    st.text_area("Texto Original do Parágrafo:", value=target_r.text, height=120, disabled=True)

                    st.markdown("#### Conteúdo Conceitual Extraído:")
                    out_t = target_r.llm_output
                    st.markdown(f"- **Conceitos:** {'; '.join(out_t.concepts) if out_t.concepts else 'Nenhum'}")
                    st.markdown(f"- **Definições:** {'; '.join(out_t.definitions) if out_t.definitions else 'Nenhuma'}")
                    st.markdown(f"- **Fatores Determinantes:** {'; '.join(out_t.determinants) if out_t.determinants else 'Nenhum'}")
                    st.markdown(f"- **Tipos / Dimensões:** {'; '.join(out_t.dimensions) if out_t.dimensions else 'Nenhuma'}")
                    st.markdown(f"- **Relações Causais:** {'; '.join(out_t.causal_relations) if out_t.causal_relations else 'Nenhuma'}")
                    st.markdown(f"- **Características:** {'; '.join(out_t.properties) if out_t.properties else 'Nenhuma'}")
                    st.markdown(f"- **Citações de Evidência:** {'; '.join(out_t.evidence) if out_t.evidence else 'Nenhuma'}")

            # Botões de Exportação do Corpus Refinado
            st.divider()
            st.markdown("#### Exportar Resultados do Corpus Refinado")

            df_export_refined = pd.DataFrame([{
                "run_id": r.run_id,
                "document_id": r.article_id,
                "paragraph_id": r.paragraph_id,
                "text": r.text,
                "concepts": "; ".join(r.llm_output.concepts),
                "definitions": "; ".join(r.llm_output.definitions),
                "determinants": "; ".join(r.llm_output.determinants),
                "dimensions": "; ".join(r.llm_output.dimensions),
                "causal_relations": "; ".join(r.llm_output.causal_relations),
                "properties": "; ".join(r.llm_output.properties),
                "evidence": "; ".join(r.llm_output.evidence),
                "evidence_valid": r.evidence_valid,
                "llm_model": r.llm_model,
                "processing_time_sec": r.processing_time
            } for r in view_refined])

            md_refined_lines = [
                f"# SLD — Corpus Refinado por Extração LLM (`{st.session_state.run_id}`)",
                f"**Data de Geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Total de Registros Válidos Refinados:** {len(view_refined)}",
                "", "---", ""
            ]
            for r in view_refined:
                o_ref = r.llm_output
                md_refined_lines.extend([
                    f"## Parágrafo `{r.paragraph_id}` (`{r.article_id}`)",
                    f"- **Conceitos:** `{'; '.join(o_ref.concepts)}`",
                    f"- **Fatores Determinantes:** `{'; '.join(o_ref.determinants)}`",
                    f"- **Dimensões:** `{'; '.join(o_ref.dimensions)}`",
                    f"- **Relações Causais:** `{'; '.join(o_ref.causal_relations)}`",
                    f"- **Evidências:** `{'; '.join(o_ref.evidence)}`",
                    f"- **Evidência Válida:** `{'PASS' if r.evidence_valid else 'FAIL'}`",
                    "", "```text", r.text, "```", "", "---", ""
                ])
            md_refined_str = "\n".join(md_refined_lines)
            jsonl_refined_str = "\n".join(r.model_dump_json() for r in view_refined)

            render_export_section("8_Corpus_Refinado_LLM", st.session_state.run_id, md_refined_str, df_export_refined, jsonl_refined_str, file_prefix="corpus_refinado_llm")

    # ==========================================
    # ABA 9: RELATÓRIO METODOLÓGICO
    # ==========================================
    with t9:
        render_methodology_header(
            title="9. Relatório Metodológico e Auditoria de Execução",
            description=(
                "Esta etapa reúne toda a documentação metodológica, equações, contadores, parâmetros e estatísticas da execução "
                "para garantir rastreabilidade e reprodutibilidade científica total em teses, dissertações e artigos."
            ),
            objective="Gerar automaticamente a documentação metodológica completa da pesquisa para garantias de reprodutibilidade.",
            method="Compilação determinística do funil de retenção, parâmetros, equações e métricas quantitativas de desempenho.",
            formula_latex=r"\text{Retention}_i = \frac{N_{\text{out},i}}{N_{\text{in},i}} \times 100",
            legend_dict={
                "N_{in,i}": "número de unidades recebidas pela etapa i",
                "N_{out,i}": "número de unidades preservadas ao final da etapa i",
                "Retention": "percentual de retenção de unidades mantidas"
            },
            interpretation="Provê rastreabilidade total para dissertações, teses e artigos acadêmicos."
        )

        render_stage_disk_loader(9, "Relatório Metodológico e Auditoria", project, st.session_state)

        if st.button("Gerar e Exportar Relatório Metodológico Completo", type="primary"):
            with st.status("Gerando e compilando relatório metodológico completo (16 seções)...", expanded=True) as status_box:
                rep_files = generate_methodology_report(
                    run_id=st.session_state.run_id,
                    run_dirs=st.session_state.run_dirs,
                    config=st.session_state.config,
                    funnel_counts=st.session_state.funnel_counts,
                    eval_report=st.session_state.eval_report,
                    llm_stats=st.session_state.llm_stats
                )

                if 9 not in st.session_state.completed_steps:
                    st.session_state.completed_steps.append(9)

                status_box.update(label="✓ Relatório metodológico compilado e exportado com sucesso!", state="complete", expanded=False)
                st.toast("✓ Relatório metodológico gerado com sucesso!")
                st.balloons()

            render_completion_panel(
                title="Relatório Metodológico Exportado",
                metrics={
                    "Run ID": st.session_state.run_id,
                    "Seções Metodológicas": 16,
                    "Arquivos Gerados": len(rep_files)
                }
            )

            if rep_files["md"].exists():
                with open(rep_files["md"], "r", encoding="utf-8") as f:
                    st.markdown(f.read())


if __name__ == "__main__":
    main()
