"""
Gerador de Relatório Metodológico automático para a execução (run_id).
Sistemática completa com 16 seções acadêmicas, equações, legendas e glossário.
"""

import os
import json
import sys
import platform
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
from src.sld.models.classification import EvaluationReport
from src.sld.utils.files import ensure_directory


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Converte um DataFrame pandas para uma tabela Markdown sem depender exclusivamente do pacote tabulate."""
    if df.empty:
        return ""
    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = [str(c) for c in df.columns]
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        body_lines = []
        for _, row in df.iterrows():
            body_lines.append("| " + " | ".join(str(val) for val in row) + " |")
        return "\n".join([header_line, sep_line] + body_lines)


def generate_methodology_report(
    run_id: str,
    run_dirs: Dict[str, Path],
    config: Dict[str, Any],
    funnel_counts: Dict[str, Dict[str, Any]],
    eval_report: Optional[EvaluationReport] = None,
    llm_stats: Optional[Dict[str, Any]] = None
) -> Dict[str, Path]:
    """
    Gera o relatório metodológico completo e salva em Markdown, JSON e CSV sob output/<run_id>/reports/.
    """
    reports_dir = run_dirs["reports"]
    ensure_directory(reports_dir)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Tabela do Funil de Dados
    funnel_rows = []
    for step_name, data in funnel_counts.items():
        n_in = data.get("n_in", 0)
        n_out = data.get("n_out", 0)
        n_rem = n_in - n_out if n_in >= n_out else 0
        ret_pct = (n_out / n_in * 100.0) if n_in > 0 else 100.0
        funnel_rows.append({
            "Etapa": step_name,
            "Entrada (N_in)": f"{n_in:,}".replace(",", "."),
            "Saída (N_out)": f"{n_out:,}".replace(",", "."),
            "Removidos": f"{n_rem:,}".replace(",", "."),
            "Retenção (%)": f"{ret_pct:.2f}%",
            "Duração (s)": f"{data.get('duration_sec', 0.0):.2f}"
        })

    df_funnel = pd.DataFrame(funnel_rows)
    df_funnel.to_csv(reports_dir / "pipeline_counts.csv", index=False, encoding="utf-8")

    # 2. Salva estatísticas gerais
    stats_json = {
        "run_id": run_id,
        "timestamp": now_str,
        "funnel": funnel_counts,
        "config": config,
    }
    with open(reports_dir / "statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats_json, f, indent=2, ensure_ascii=False)

    # 3. Salva run_config.json
    with open(run_dirs["config"] / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 4. Conteúdo do Relatório Metodológico Acadêmico em 16 Seções
    md_lines = [
        f"# Relatório Metodológico de Execução Científica — {run_id}",
        "",
        "## 1. Identificação da Execução e Ambiente Computacional",
        f"- **Run ID de Execução:** `{run_id}`",
        f"- **Data e Hora do Registro:** `{now_str}`",
        f"- **Diretório Raiz de Saída:** `{run_dirs['root']}`",
        f"- **Versão do Python:** `{sys.version.split()[0]}`",
        f"- **Sistema Operacional:** `{platform.system()} {platform.release()} ({platform.machine()})`",
        f"- **Processador / CPU:** `{platform.processor()}`",
        "",
        "## 2. Caracterização Geral do Corpus Ingestado",
        "O pipeline processou o lote documental fornecido no diretório de entrada, realizando sanitização de metadados e conversão para Markdown limpo.",
        "",
        "## 3. ETL — Extração, Transformação e Carregamento Persistente",
        "O pipeline trata a pasta de saída como uma **unidade de análise persistente** vinculada a `analysis.json`, mantendo rastreabilidade total de cada artigo via hash **SHA-256** do PDF fonte em `manifests/article_registry.json`.",
        "A ingestão realiza a conversão de PDFs em **arquivos .md individuais por artigo**, aplicando controle atômico de duplicidades por hash SHA-256 com políticas configuráveis (`pular`, `sobrescrever`, `versao`) para evitar reprocessamentos desnecessários.",
        "$$T_{\\text{sucesso}} = \\frac{N_{\\text{processados}}}{N_{\\text{válidos}}} \\times 100 \\quad \\land \\quad \\text{Redução}_{\\text{Tamanho}}(\\%) = \\left(1 - \\frac{\\text{Tamanho}_{MD}}{\\text{Tamanho}_{Inicial}}\\right) \\times 100$$",
        "- **N_processados:** número de artigos convertidos em arquivos .md individuais;",
        "- **N_válidos:** quantidade total de PDFs válidos fornecidos na entrada;",
        "- **Tamanho_Inicial:** tamanho acumulado dos PDFs/ZIPs de origem (bytes);",
        "- **Tamanho_MD:** tamanho total dos arquivos .md individuais gerados (bytes).",
        "",
        "## 4. Segmentação em Parágrafos e Representação Semântica (Sentence Embeddings)",
        f"Leitura dos arquivos .md individuais do ETL, fragmentação em parágrafos concisos (`segment_markdown_paragraphs`) e mapeamento vetorial denso via modelo `{config.get('embedding_model', 'nomic-embed-text')}`.",
        "$$P_{i,j} = \\text{Segment}(MD_i) \\quad \\land \\quad E(P_{i,j}) \\in \\mathbb{R}^{d}, \\quad \\|E(P_{i,j})\\|_2 = 1$$",
        "- **MD_i:** arquivo Markdown individual do i-ésimo artigo;",
        "- **P_{i,j}:** j-ésimo parágrafo extraído do artigo i;",
        "- **E(.):** função de transformação do Sentence Transformer;",
        "- **d:** dimensionalidade do vetor numérico;",
        "- **R^d:** espaço vetorial real d-dimensional.",
        "",
        "## 5. Recuperação por Similaridade Semântica (Cosine Similarity)",
        "$$\\text{S}(P_i, Q) = \\frac{E(P_i) \\cdot E(Q)}{\\|E(P_i)\\| \\|E(Q)\\|}$$",
        "- **P_i:** parágrafo analisado;",
        "- **Q:** consulta ou sentença-âncora de referência;",
        "- **E(P_i):** embedding do parágrafo;",
        "- **E(Q):** embedding da consulta;",
        "- **S(P_i, Q):** grau de similaridade do cosseno.",
        "",
        "$$\\text{Selected}_i = I[S(P_i, Q) \\ge \\theta_s]$$",
        f"- **θ_s:** limiar semântico configurado (`{config.get('similarity_threshold', 0.50):.2f}`).",
        "",
        "## 6. Construção do Gold Standard e Validação Auditável",
        "A construção do conjunto supervisionado foi realizada mediante classificação multilabel de trechos previamente recuperados na etapa de busca semântica.",
        "As categorias analíticas representaram diferentes funções conceituais do conteúdo: definição ou conceituação, fatores determinantes, tipos ou dimensões, relações causais e características ou propriedades.",
        "Trechos sem conteúdo pertinente foram classificados como não relevantes.",
        "As anotações puderam ser realizadas diretamente na aplicação ou por meio de conjuntos exportados em formato Markdown (.md) e posteriormente reimportados, preservando identificadores únicos e rastreabilidade entre os registros anotados e o corpus original.",
        "",
        "## 7. Regressão Logística Multilabel (One-vs-Rest)",
        "$$P(y_k = 1 \\mid X_i) = \\frac{1}{1 + e^{-(w_k^T X_i + b_k)}}$$",
        "- **X_i:** vetor de embedding do parágrafo i;",
        "- **y_k:** pertencimento binário à classe k;",
        "- **w_k:** vetor de pesos estimados da classe k;",
        "- **b_k:** termo de intercepto.",
        "",
        "## 8. Avaliação do Classificador Supervisionado (Conjunto de Teste)",
    ]

    if eval_report:
        md_lines.extend([
            f"- **Macro F1:** `{eval_report.macro_f1:.4f}`",
            f"- **Micro F1:** `{eval_report.micro_f1:.4f}`",
            f"- **Weighted F1:** `{eval_report.weighted_f1:.4f}`",
            f"- **Hamming Loss:** `{eval_report.hamming_loss:.4f}`",
            f"- **Subset Accuracy:** `{eval_report.subset_accuracy:.4f}`",
            "",
            "### Desempenho por Classe Conceitual:",
            ""
        ])
        c_rows = []
        for c_name, m in eval_report.per_class_metrics.items():
            c_rows.append({
                "Classe Conceitual": c_name,
                "Threshold (θ)": f"{m.threshold:.2f}",
                "Precision": f"{m.precision:.4f}",
                "Recall": f"{m.recall:.4f}",
                "F1-Score": f"{m.f1:.4f}",
                "Suporte": m.support
            })
        df_c = pd.DataFrame(c_rows)
        df_c.to_csv(reports_dir / "classification_metrics.csv", index=False, encoding="utf-8")
        md_lines.append(df_to_markdown_table(df_c))
        md_lines.append("")

    md_lines.extend([
        "## 9. Classificação Conceitual dos Candidatos",
        "$$\\hat{y}_{ik} = I[P(y_k = 1 \\mid X_i) \\ge \\theta_k]$$",
        "- **ŷ_ik:** classificação binária atribuída ao parágrafo i na classe k;",
        "- **θ_k:** limiar de decisão específico e otimizado da classe k.",
        "",
        "## 10. Constituição do Corpus Final",
        "$$C_{\\text{final}} = \\{ P_i \\mid \\text{Relevant}(P_i) = 1 \\}$$",
        "- **C_final:** subconjunto de parágrafos considerados conceitualmente relevantes para envio ao LLM.",
        "",
        "## 11. Extração Estruturada por Modelo de Linguagem Local (Ollama)",
        "$$\\text{LLM}(P_i) \\rightarrow \\{C, F, D, M, R, E\\}$$",
        "- **C:** conceitos principais;",
        "- **F:** fatores determinantes;",
        "- **D:** dimensões analíticas do conceito;",
        "- **M:** mecanismos de atuação;",
        "- **R:** relações conceituais;",
        "- **E:** evidências textuais literais.",
        "",
        "$$\\text{Reduction}_{LLM}(\\%) = \\left(1 - \\frac{N_{\\text{out}}}{N_{\\text{in}}}\\right) \\times 100$$",
    ])

    if llm_stats:
        model_str = llm_stats.get("llm_model", config.get("llm_model", "qwen2.5:7b"))
        n_proc = llm_stats.get("processed_paragraphs", 0)
        red_pct = llm_stats.get("reduction_percentage", 0.0)
        elapsed = llm_stats.get("total_processing_time_sec", llm_stats.get("total_elapsed_sec", 0.0))
        total_wait = llm_stats.get("total_wait_time_sec", 0.0)
        rpm_val = llm_stats.get("rpm_configured", 10.0)
        evr_val = llm_stats.get("evidence_validation_rate_evr", 0.0)
        pvr_val = llm_stats.get("paragraph_validation_rate_pvr", 0.0)

        md_lines.extend([
            f"- **Modelo de Linguagem Executado:** `{model_str}`",
            f"- **Backend de Inferência:** `Ollama Local ({config.get('ollama_url', 'http://localhost:11434')})`",
            f"- **Janela de Contexto (num_ctx):** `{config.get('num_ctx', 2048)} tokens`",
            f"- **Temperatura:** `{config.get('temperature', 0.0)}`",
            f"- **Semente (Seed):** `{config.get('seed', 42)}`",
            f"- **Parágrafos Processados pelo LLM:** `{n_proc}`",
            f"- **Limite de Carga Configurado:** `{rpm_val} RPM`",
            f"- **Tempo Total de Espera por Rate Limit:** `{total_wait:.2f}s`",
            f"- **Tempo Efetivo de Inferência:** `{elapsed:.2f}s`",
            f"- **Taxa de Validação de Evidências (EVR):** `{evr_val:.2f}%`",
            f"- **Taxa de Parágrafos com Evidência Válida (PVR):** `{pvr_val:.2f}%`",
            f"- **Taxa de Redução de Chamadas ao LLM:** `{red_pct:.2f}%`",
            ""
        ])

    md_lines.extend([
        "## 12. Funil Completo de Processamento de Dados",
        df_to_markdown_table(df_funnel),
        "",
        "## 13. Parâmetros Metodológicos e de Desempenho Utilizados",
        f"- **Sentence Transformer Model:** `{config.get('embedding_model', 'nomic-embed-text')}`",
        f"- **Limiar de Similaridade Cosseno:** `{config.get('similarity_threshold', 0.50)}`",
        f"- **Max Characters por Chunk:** `{config.get('max_characters', 500)}`",
        f"- **Modelo LLM Local:** `{config.get('llm_model', 'qwen2.5:7b')}`",
        f"- **LLM Temperature:** `{config.get('temperature', 0.0)}`",
        f"- **LLM Context Window:** `{config.get('num_ctx', 2048)}`",
        "",
        "## 14. Software e Ambiente Computacional",
        f"- **Python:** `{sys.version.split()[0]}` | **OS:** `{platform.system()} {platform.release()}`",
        "- **Bibliotecas Core:** PyMuPDF, SentenceTransformers, Scikit-Learn, Pandas, PyYAML, Pydantic v2, Streamlit.",
        "",
        "## 15. Síntese das Equações Metodológicas",
        "Todas as equações do pipeline foram executadas em código determinístico, garantindo rastreabilidade por run_id.",
        "",
        "## 16. Glossário Metodológico Científico",
        "| Termo | Definição Metodológica |",
        "| --- | --- |",
        "| **Embedding** | Representação vetorial densa d-dimensional de um texto em um espaço contínuo. |",
        "| **Sentence Transformer** | Arquitetura de rede neural profunda utilizada para gerar embeddings de sentenças e parágrafos. |",
        "| **Cosine Similarity** | Medida do cosseno do ângulo entre dois vetores vetoriais no espaço d-dimensional. |",
        "| **Threshold (θ)** | Valor limiar numérico mínimo utilizado para decisões de corte e seleção. |",
        "| **Multilabel** | Classificação na qual um mesmo parágrafo pode receber múltiplas categorias simultaneamente. |",
        "| **Precision** | Proporção das classificações positivas produzidas pelo modelo que estavam corretas. |",
        "| **Recall** | Capacidade do modelo de recuperar os casos positivos existentes no conjunto avaliado. |",
        "| **F1-Score** | Média harmônica entre a precisão e o recall. |",
        "| **Gold Standard** | Conjunto de dados rotulado manualmente por pesquisador humano utilizado como referência. |",
        "| **LLM** | Modelo de linguagem de grande porte utilizado localmente para extração conceitual estruturada. |",
        ""
    ])

    report_md_path = reports_dir / "methodology_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return {
        "md": report_md_path,
        "json": reports_dir / "statistics.json",
        "csv_funnel": reports_dir / "pipeline_counts.csv"
    }
