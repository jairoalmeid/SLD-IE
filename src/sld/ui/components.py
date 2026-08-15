"""
Componentes de interface reutilizáveis para a aplicação científica SLD.
Padronização visual e metodológica para pesquisas acadêmicas e auditoria científica.
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from src.sld.ui.styles import inject_custom_styles


def render_institutional_header(logo_path: Optional[Path] = None):
    """
    Renderiza o cabeçalho institucional padronizado do SLD com o logo na barra lateral.
    """
    inject_custom_styles()
    if logo_path and logo_path.exists():
        st.sidebar.image(str(logo_path), use_container_width=True)
        st.sidebar.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    st.markdown(
        "<h1 style='margin-bottom: 0px; font-weight: 700; color: #0f172a;'>SLD — Scientific Literature Decoder</h1>"
        "<p style='margin-top: 0px; color: #475569; font-size: 1.05rem; font-weight: 500;'>"
        "Análise semântica, classificação supervisionada e extração conceitual de literatura científica"
        "</p>",
        unsafe_allow_html=True
    )
    st.divider()


def render_pipeline_stepper(current_step: int = 1, completed_steps: Optional[List[int]] = None):
    """
    Renderiza uma barra horizontal compacta do pipeline metodológico com status visual das etapas.
    """
    completed = completed_steps or []
    steps = [
        "1. Ingestão",
        "2. Embeddings",
        "3. Exploratória",
        "4. Similaridade",
        "5. Treino",
        "6. Classificação",
        "7. Gemma 3",
        "8. Relatório"
    ]

    html_items = []
    for idx, step_name in enumerate(steps, start=1):
        if idx in completed:
            cls = "sld-stepper-item completed"
            icon = "✓ "
        elif idx == current_step:
            cls = "sld-stepper-item active"
            icon = "● "
        else:
            cls = "sld-stepper-item"
            icon = "○ "
        html_items.append(f"<span class='{cls}'>{icon}{step_name}</span>")

    stepper_html = f"<div class='sld-stepper'>{' <span style=\"color:#cbd5e1;\">→</span> '.join(html_items)}</div>"
    st.markdown(stepper_html, unsafe_allow_html=True)


def render_methodology_header(
    title: str,
    description: str,
    objective: str = "",
    method: str = "",
    formula_latex: Optional[str] = None,
    legend_dict: Optional[Dict[str, str]] = None,
    interpretation: str = ""
):
    """
    Renderiza o cabeçalho padronizado da etapa com descrição metodológica de 2 a 4 linhas
    e expander limpo para detalhes matemáticos e conceituais.
    """
    st.header(title)
    st.markdown(f"<p style='color: #334155; font-size: 1.0rem; line-height: 1.6;'>{description}</p>", unsafe_allow_html=True)

    with st.expander("Detalhes metodológicos e representação matemática", expanded=False):
        if objective:
            st.markdown(f"**Objetivo:** {objective}")
        if method:
            st.markdown(f"**Método:** {method}")

        if formula_latex:
            st.markdown("**Representação matemática:**")
            st.latex(formula_latex)

            if legend_dict:
                st.markdown("**Legenda dos Símbolos:**")
                legend_lines = [f"- **{symbol}:** {desc}" for symbol, desc in legend_dict.items()]
                st.markdown("\n".join(legend_lines))

        if interpretation:
            st.markdown(f"**Interpretação Científica:** {interpretation}")


def render_pipeline_metrics(
    input_count: int,
    output_count: int,
    duration_sec: float = 0.0,
    n_docs: Optional[int] = None,
    n_words: Optional[int] = None,
    unit_label: str = "parágrafos"
):
    """
    Renderiza os indicadores quantitativos da etapa com transparência do denominador e estilo minimalista.
    """
    n_removed = max(0, input_count - output_count)
    retention_pct = (output_count / input_count * 100.0) if input_count > 0 else 100.0
    reduction_pct = 100.0 - retention_pct

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"<div class='sld-card'>"
            f"<div class='sld-card-label'>Entrada (N_in)</div>"
            f"<div class='sld-card-value'>{input_count:,}</div>"
            f"<div class='sld-card-sub'>{unit_label}</div>"
            f"</div>".replace(",", "."),
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<div class='sld-card'>"
            f"<div class='sld-card-label'>Saída (N_out)</div>"
            f"<div class='sld-card-value'>{output_count:,}</div>"
            f"<div class='sld-card-sub'>preservados</div>"
            f"</div>".replace(",", "."),
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"<div class='sld-card'>"
            f"<div class='sld-card-label'>Removidos</div>"
            f"<div class='sld-card-value'>{n_removed:,}</div>"
            f"<div class='sld-card-sub'>-{reduction_pct:.1f}% filtrados</div>"
            f"</div>".replace(",", "."),
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"<div class='sld-card'>"
            f"<div class='sld-card-label'>Taxa de Retenção</div>"
            f"<div class='sld-card-value'>{retention_pct:.2f}%</div>"
            f"<div class='sld-card-sub'>{output_count:,} de {input_count:,}</div>"
            f"</div>".replace(",", "."),
            unsafe_allow_html=True
        )
    with c5:
        st.markdown(
            f"<div class='sld-card'>"
            f"<div class='sld-card-label'>Tempo</div>"
            f"<div class='sld-card-value'>{duration_sec:.2f}s</div>"
            f"<div class='sld-card-sub'>duração total</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    if n_docs is not None or n_words is not None:
        c_extra1, c_extra2 = st.columns(2)
        if n_docs is not None:
            c_extra1.caption(f"**Documentos Ativos:** {n_docs:,}".replace(",", "."))
        if n_words is not None:
            c_extra2.caption(f"**Total de Palavras:** {n_words:,}".replace(",", "."))

    st.divider()


def render_completion_panel(title: str, metrics: Dict[str, Any]):
    """
    Renderiza um painel de conclusão elegante ao final da etapa com dados resumidos.
    """
    st.markdown(
        f"<div style='background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 16px 20px; margin-top: 15px; margin-bottom: 20px;'>"
        f"<h4 style='color: #166534; margin: 0px 0px 8px 0px; font-weight: 700;'>✓ {title}</h4>"
        f"<p style='color: #15803d; font-size: 0.95rem; margin: 0px;'>Resultados processados e salvos com sucesso no repositório da execução.</p>"
        f"</div>",
        unsafe_allow_html=True
    )
    cols = st.columns(len(metrics))
    for idx, (k, v) in enumerate(metrics.items()):
        cols[idx].metric(k, str(v))


def render_export_section(
    stage_name: str,
    run_id: str,
    md_content: str,
    df_data: pd.DataFrame,
    jsonl_content: Optional[str] = None,
    file_prefix: str = "export"
):
    """
    Renderiza uma seção padronizada de exportação para a etapa corrente,
    oferecendo botões de download para Markdown (.md), CSV (.csv) e JSONL (.jsonl).
    """
    with st.expander(f"Exportar Dados da Etapa: {stage_name}", expanded=False):
        st.markdown(f"**Download do subconjunto de parágrafos refinados na etapa `{stage_name}`:**")
        c1, c2, c3 = st.columns(3)

        c1.download_button(
            label="Baixar em Markdown (.md)",
            data=md_content,
            file_name=f"{file_prefix}_{stage_name.lower().replace(' ', '_')}_{run_id}.md",
            mime="text/markdown",
            use_container_width=True
        )

        c2.download_button(
            label="Baixar Tabela em CSV (.csv)",
            data=df_data.to_csv(index=False, encoding="utf-8"),
            file_name=f"{file_prefix}_{stage_name.lower().replace(' ', '_')}_{run_id}.csv",
            mime="text/csv",
            use_container_width=True
        )

        if jsonl_content:
            c3.download_button(
                label="Baixar Dados em JSONL (.jsonl)",
                data=jsonl_content,
                file_name=f"{file_prefix}_{stage_name.lower().replace(' ', '_')}_{run_id}.jsonl",
                mime="application/jsonlines",
                use_container_width=True
            )


def render_empty_state(title: str, description: str, recommendation: str = ""):
    """
    Renderiza um cartão discreto de estado vazio quando uma etapa ainda não foi executada.
    """
    st.markdown(
        f"<div class='sld-card' style='text-align: center; padding: 32px 20px; background-color: #f8fafc;'>"
        f"<h4 style='color: #475569; margin-bottom: 8px;'>{title}</h4>"
        f"<p style='color: #64748b; font-size: 0.95rem; margin-bottom: 12px;'>{description}</p>"
        f"{f'<p style=\"color: #2563eb; font-size: 0.85rem; font-weight: 600;\">{recommendation}</p>' if recommendation else ''}"
        f"</div>",
        unsafe_allow_html=True
    )


def render_status_card(title: str, status_state: str, details: str = ""):
    """
    Renderiza um cartão de status visual com badge e texto explicativo.
    """
    badge_map = {
        "completed": ("✓ Concluído", "sld-badge-success"),
        "running": ("● Em Execução", "sld-badge-info"),
        "ready": ("● Pronto", "sld-badge-info"),
        "completed_with_warnings": ("⚠ Concluído com Alertas", "sld-badge-warning"),
        "failed": ("✕ Falha", "sld-badge-error"),
        "idle": ("○ Não Iniciado", "sld-badge-neutral")
    }
    badge_text, badge_cls = badge_map.get(status_state, ("○ Não Iniciado", "sld-badge-neutral"))

    st.markdown(
        f"<div class='sld-card'>"
        f"<div style='display: flex; justify-content: space-between; align-items: center;'>"
        f"<span style='font-weight: 700; color: #0f172a;'>{title}</span>"
        f"<span class='sld-badge {badge_cls}'>{badge_text}</span>"
        f"</div>"
        f"{f'<div class=\"sld-card-sub\" style=\"margin-top: 6px;\">{details}</div>' if details else ''}"
        f"</div>",
        unsafe_allow_html=True
    )


def render_formula(
    title: str,
    latex_formula: str,
    legend_dict: Dict[str, str],
    interpretation: str,
    note: Optional[str] = None
):
    """
    Renderiza um bloco isolado de representação matemática com fórmula LaTeX,
    legenda obrigatória dos símbolos e interpretação acadêmica.
    """
    st.markdown(f"### {title}")
    st.latex(latex_formula)

    st.markdown("**Onde:**")
    for symbol, desc in legend_dict.items():
        st.markdown(f"- **{symbol}:** {desc}")

    st.markdown(f"**Interpretação:** {interpretation}")
    if note:
        st.caption(f"Nota metodológica: {note}")


def render_descriptive_statistics(values: Union[List[float], np.ndarray, pd.Series], title: str = "Estatísticas Descritivas"):
    """
    Calcula e exibe a tabela completa de estatísticas descritivas para uma variável numérica:
    N, Média, Mediana, Desvio Padrão, Mínimo, Máximo, P25, P75.
    """
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        st.caption("Sem dados suficientes para calcular estatísticas descritivas.")
        return

    n = len(arr)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    p25_val = float(np.percentile(arr, 25))
    p75_val = float(np.percentile(arr, 75))

    st.markdown(f"#### {title}")
    df_stats = pd.DataFrame([{
        "N (Amostras)": f"{n:,}".replace(",", "."),
        "Média (x̄)": f"{mean_val:.4f}",
        "Mediana": f"{median_val:.4f}",
        "Desvio Padrão (s)": f"{std_val:.4f}",
        "Mínimo": f"{min_val:.4f}",
        "P25": f"{p25_val:.4f}",
        "P75": f"{p75_val:.4f}",
        "Máximo": f"{max_val:.4f}"
    }])
    st.dataframe(df_stats, use_container_width=True)


def render_interpretation_box(text: str, title: str = "Como interpretar"):
    """
    Exibe uma caixa discreta de orientação metodológica para interpretação de gráficos e tabelas.
    """
    st.markdown(
        f"<div style='background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 12px 16px; margin-top: 10px; margin-bottom: 15px; border-radius: 0px 6px 6px 0px;'>"
        f"<strong style='color: #1e3a8a;'>{title}:</strong> "
        f"<span style='color: #334155; font-size: 0.95rem;'>{text}</span>"
        f"</div>",
        unsafe_allow_html=True
    )


def render_methodological_alert(message: str, alert_type: str = "warning"):
    """
    Renderiza um alerta metodológico acadêmico com cor e texto explicativo.
    """
    prefix = {
        "warning": "Aviso Metodológico",
        "info": "Nota Informativa",
        "error": "Invalidação Técnica",
        "success": "Confirmação de Validação"
    }.get(alert_type, "Alerta")

    full_msg = f"**{prefix}:** {message}"
    if alert_type == "warning":
        st.warning(full_msg)
    elif alert_type == "error":
        st.error(full_msg)
    elif alert_type == "success":
        st.success(full_msg)
    else:
        st.info(full_msg)


def dataframe_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """
    Converte um DataFrame do pandas para formato de tabela Markdown,
    com fallback seguro em Python puro caso o pacote 'tabulate' não esteja disponível.
    """
    if df.empty:
        return ""
    try:
        return df.to_markdown(index=index)
    except Exception:
        target_df = df.reset_index() if index else df
        headers = [str(col) for col in target_df.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |"
        ]
        for _, row in target_df.iterrows():
            row_str = " | ".join(str(val).replace("\n", " ") for val in row.values)
            lines.append(f"| {row_str} |")
        return "\n".join(lines)


# Compatibilidade retroativa para funções de nomes legados
st_funnel_card = render_pipeline_metrics
st_stage_header = render_methodology_header


def render_stage_disk_loader(
    stage_num: int,
    stage_title: str,
    project: Any,
    session_state: Any,
    on_load_callback: Optional[Any] = None,
    custom_help: str = ""
):
    """
    Componente visual padronizado que inspeciona o disco na pasta de saída
    e exibe um botão explícito para carregar/sincronizar os dados salvos da etapa atual.
    """
    from src.sld.corpus.session_manager import inspect_stage_files, restore_full_session_state

    inspection = inspect_stage_files(project)
    stage_key = f"stage_{stage_num}"
    stage_info = inspection.get(stage_key, {})
    has_files = stage_info.get("has_files", False)
    desc = stage_info.get("description", "Nenhum arquivo gravado no disco para esta etapa.")

    with st.container():
        col_inf, col_btn = st.columns([3, 1])
        with col_inf:
            if has_files:
                st.markdown(
                    f"<div style='background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 7px 12px; margin-bottom: 8px;'>"
                    f"<span style='color: #166534; font-weight: 600;'>📁 Arquivos Salvos Detectados no Disco:</span> "
                    f"<span style='color: #15803d;'>{desc}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 7px 12px; margin-bottom: 8px;'>"
                    f"<span style='color: #64748b; font-weight: 500;'>📂 Disco:</span> "
                    f"<span style='color: #94a3b8;'>{desc}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        with col_btn:
            btn_label = f"📥 Carregar Arquivos da Etapa {stage_num}"
            if st.button(
                btn_label,
                key=f"btn_disk_loader_stage_{stage_num}",
                type="primary" if has_files else "secondary",
                use_container_width=True,
                help=custom_help or f"Carrega os arquivos salvos no diretório de saída correspondentes à etapa {stage_num}."
            ):
                with st.spinner(f"Carregando e sincronizando arquivos da Etapa {stage_num} do disco..."):
                    res = restore_full_session_state(project, session_state)
                    if on_load_callback:
                        try:
                            on_load_callback()
                        except Exception:
                            pass
                    st.toast(f"✓ Dados da Etapa {stage_num} carregados com sucesso! ({res.get('restored_count', 0)} itens)")
                    st.rerun()
