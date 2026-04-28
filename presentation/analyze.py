import json
import re
from pathlib import Path

import streamlit as st

from service.extraction import extract_pdf_text
from service.transformation import analyze_text, list_ollama_models, save_result

QUESTION_TYPES = ["Sim/Não", "Texto livre", "Lista", "Múltipla escolha", "Numérico"]
SCHEMA_PATH = Path(__file__).resolve().parents[1] / ".schema.json"


def _label_to_field(label: str) -> str:
    field = label.lower().strip()
    for src, dst in [("áàãâä", "a"), ("éèêë", "e"), ("íìîï", "i"), ("óòõôö", "o"), ("úùûü", "u"), ("ç", "c")]:
        for ch in src:
            field = field.replace(ch, dst)
    field = re.sub(r"[^a-z0-9\s]", "", field)
    return re.sub(r"\s+", "_", field.strip()) or "campo"


def _load_schema() -> list[dict]:
    if SCHEMA_PATH.exists():
        try:
            return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save_schema(questions: list[dict]) -> None:
    SCHEMA_PATH.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")


def _render_upload() -> list:
    st.subheader("1. Enviar artigos em PDF")
    files = st.file_uploader(
        "Selecione um ou mais arquivos PDF",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if files:
        st.caption(f"{len(files)} arquivo(s) selecionado(s)")
    return files or []


def _render_parameters() -> list[dict]:
    st.subheader("2. Parâmetros de análise")
    st.caption("Título, autores, ano, periódico e DOI são extraídos automaticamente de todos os artigos.")

    if "questions" not in st.session_state:
        st.session_state.questions = _load_schema()

    questions: list[dict] = st.session_state.questions

    if questions:
        for i, q in enumerate(questions):
            col_info, col_type, col_del = st.columns([4, 2, 1])
            with col_info:
                st.markdown(f"**{q['label']}**")
                st.caption(f"campo YAML: `{q['field']}`")
            with col_type:
                st.markdown(f"`{q['type']}`")
                if q["type"] == "Múltipla escolha" and q.get("options"):
                    st.caption(" · ".join(q["options"]))
            with col_del:
                if st.button("✕", key=f"del_{i}", help="Remover"):
                    questions.pop(i)
                    st.rerun()

        col_save, col_clear = st.columns(2)
        with col_save:
            if st.button("💾 Salvar parâmetros", use_container_width=True):
                _save_schema(questions)
                st.toast("Parâmetros salvos!", icon="✅")
        with col_clear:
            if st.button("Limpar tudo", use_container_width=True):
                st.session_state.questions = []
                _save_schema([])
                st.rerun()
    else:
        st.info("Nenhum parâmetro definido. Adicione abaixo.")

    with st.expander("➕ Adicionar parâmetro"):
        with st.form("add_question", clear_on_submit=True):
            label = st.text_input(
                "Pergunta / Rótulo",
                placeholder="Ex: Qual tipo de desastre o artigo aborda?",
            )
            col_field, col_type = st.columns(2)
            with col_field:
                custom_field = st.text_input(
                    "Nome do campo YAML",
                    placeholder="Gerado automaticamente se vazio",
                )
            with col_type:
                qtype = st.selectbox("Tipo de resposta", QUESTION_TYPES)

            options_raw = ""
            if qtype == "Múltipla escolha":
                options_raw = st.text_input(
                    "Opções (separadas por vírgula)",
                    placeholder="Ex: Análise, Estudo empírico, Revisão sistemática",
                )

            if st.form_submit_button("Adicionar") and label.strip():
                field = custom_field.strip() or _label_to_field(label)
                options = [o.strip() for o in options_raw.split(",") if o.strip()]
                questions.append({"label": label.strip(), "field": field, "type": qtype, "options": options})
                st.rerun()

    return questions


def _render_model() -> str:
    st.subheader("3. Modelo Ollama")

    models = list_ollama_models()
    if models:
        preferred = ["llama3.1:8b", "llama3.2", "llama3", "mistral"]
        default_idx = next(
            (i for pref in preferred for i, m in enumerate(models) if pref in m),
            0,
        )
        return st.selectbox(
            "Modelo disponível",
            options=models,
            index=default_idx,
            help="Recomendado: llama3.1:8b ou llama3.2. Modelos menores que 7b tendem a dar respostas incompletas em textos acadêmicos.",
        )
    else:
        st.warning("Ollama não encontrado em `localhost:11434`. Certifique-se de que o Ollama está em execução.")
        return st.text_input("Modelo (digitar manualmente)", value="llama3.2")


def _run_analysis(uploaded_files: list, questions: list[dict], model: str) -> None:
    st.subheader("4. Executar análise")

    if not uploaded_files:
        st.info("Envie ao menos um PDF na etapa 1.")
        return
    if not questions:
        st.info("Defina ao menos um parâmetro na etapa 2.")
        return

    st.write(
        f"**{len(uploaded_files)}** arquivo(s) · "
        f"**{len(questions)}** parâmetro(s) · "
        f"Modelo: `{model}`"
    )

    if not st.button("🔍 Executar análise", type="primary", use_container_width=True):
        return

    total = len(uploaded_files)
    progress = st.progress(0, text="Iniciando…")
    results: list[Path] = []
    errors: list[tuple[str, str]] = []

    for i, f in enumerate(uploaded_files, start=1):
        progress.progress((i - 1) / total, text=f"Processando {f.name} ({i}/{total})…")
        try:
            text, pages = extract_pdf_text(f.getvalue())
            answers = analyze_text(text, questions, model)
            output_path = save_result(f.name, questions, answers)
            results.append(output_path)
        except Exception as exc:
            errors.append((f.name, str(exc)))

    progress.progress(1.0, text="Concluído.")

    for name, msg in errors:
        st.error(f"Erro em **{name}**: {msg}")

    if not results:
        return

    st.success(f"{len(results)} arquivo(s) analisado(s) e salvos em `data/collections/`.")

    for output_path in results:
        content = output_path.read_text(encoding="utf-8")
        with st.expander(f"📄 {output_path.name}"):
            col_preview, col_download = st.columns([4, 1])
            with col_preview:
                st.markdown(content)
            with col_download:
                st.download_button(
                    label="⬇️ Baixar",
                    data=content,
                    file_name=output_path.name,
                    mime="text/markdown",
                    key=f"dl_{output_path.stem}",
                )


def show():
    st.header("Análise de Artigos Científicos")
    st.write(
        "Envie os artigos em PDF, defina os parâmetros que deseja extrair "
        "e o modelo local irá analisar cada trabalho diretamente. "
        "O resultado é salvo como Markdown com frontmatter YAML compatível com Obsidian."
    )

    uploaded_files = _render_upload()
    st.divider()
    questions = _render_parameters()
    st.divider()
    model = _render_model()
    st.divider()
    _run_analysis(uploaded_files, questions, model)
