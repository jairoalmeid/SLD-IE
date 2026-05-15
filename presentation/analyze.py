import json
import re
from pathlib import Path

import streamlit as st

from service.extraction import extract_pdf_text
from service.transformation import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    analyze_text,
    list_ollama_models,
    save_result,
    test_model,
)

QUESTION_TYPES = ["Sim/Não", "Texto livre", "Lista", "Múltipla escolha", "Numérico"]
SCHEMA_PATH = Path(__file__).resolve().parents[1] / ".schema.json"

_QUEUE    = "_analysis_queue"
_RESULTS  = "_analysis_results"
_ERRORS   = "_analysis_errors"
_RUNNING  = "_analysis_running"
_PARAMS   = "_analysis_params"


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


def _render_upload() -> tuple[list, str]:
    st.subheader("1. Enviar artigos em PDF")
    collection_name = st.text_input(
        "Nome da coleção",
        placeholder="Ex: desastres_urbanos_2024",
        help="Os resultados serão salvos em data/collections/{nome}/",
    )
    files = st.file_uploader(
        "Selecione um ou mais arquivos PDF",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if files:
        st.caption(f"{len(files)} arquivo(s) selecionado(s)")
    return files or [], collection_name.strip()


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
            label = st.text_input("Pergunta / Rótulo", placeholder="Ex: Qual tipo de desastre o artigo aborda?")
            col_field, col_type = st.columns(2)
            with col_field:
                custom_field = st.text_input("Nome do campo YAML", placeholder="Gerado automaticamente se vazio")
            with col_type:
                qtype = st.selectbox("Tipo de resposta", QUESTION_TYPES)

            options_raw = st.text_input(
                "Opções (separadas por vírgula)",
                placeholder="Ex: Análise, Estudo empírico, Revisão sistemática",
            ) if qtype == "Múltipla escolha" else ""

            if st.form_submit_button("Adicionar") and label.strip():
                field = custom_field.strip() or _label_to_field(label)
                options = [o.strip() for o in options_raw.split(",") if o.strip()]
                questions.append({"label": label.strip(), "field": field, "type": qtype, "options": options})
                st.rerun()

    return questions


def _render_model() -> tuple[str, str, int]:
    st.subheader("3. Modelo Ollama")
    ollama_url = st.text_input(
        "Host Ollama",
        value=DEFAULT_OLLAMA_URL,
        help="Use o IP da máquina remota ou 'http://localhost:11434' para uso local.",
    )
    models = list_ollama_models(ollama_url)
    if models:
        default_idx = next((i for i, m in enumerate(models) if "gemma4" in m or "gemma" in m), 0)
        model = st.selectbox("Modelo disponível", options=models, index=default_idx)
    else:
        st.warning(f"Ollama não encontrado em `{ollama_url}`. Verifique se o serviço está em execução.")
        model = st.text_input("Modelo (digitar manualmente)", value=DEFAULT_MODEL)

    timeout = st.slider(
        "Tempo limite por artigo (s)",
        min_value=60, max_value=600, value=300, step=30,
        help="Aumente para modelos lentos ou artigos longos.",
    )

    if st.button("🔌 Testar modelo", use_container_width=True):
        with st.spinner(f"Testando `{model}`…"):
            try:
                reply = test_model(ollama_url, model)
                st.success(f"Modelo respondeu: `{reply}`")
            except Exception as e:
                st.error(f"Falha: {e}")

    return ollama_url, model, timeout


def _render_analysis(uploaded_files: list, collection_name: str, questions: list[dict], ollama_url: str, model: str, timeout: int) -> None:
    st.subheader("4. Executar análise")

    if not uploaded_files:
        st.info("Envie ao menos um PDF na etapa 1.")
        return
    if not collection_name:
        st.warning("Defina o nome da coleção na etapa 1.")
        return
    if not questions:
        st.info("Defina ao menos um parâmetro na etapa 2.")
        return

    running = st.session_state.get(_RUNNING, False)

    # ── idle: show start button ───────────────────────────────────────────────
    if not running:
        st.write(
            f"**{len(uploaded_files)}** arquivo(s) · "
            f"**{len(questions)}** parâmetro(s) · "
            f"Coleção: `{collection_name}` · Modelo: `{model}`"
        )
        if st.button("🔍 Executar análise", type="primary", use_container_width=True):
            st.session_state[_QUEUE]   = [(f.name, f.getvalue()) for f in uploaded_files]
            st.session_state[_RESULTS] = []
            st.session_state[_ERRORS]  = []
            st.session_state[_RUNNING] = True
            st.session_state[_PARAMS]  = {
                "collection": collection_name,
                "questions": questions,
                "model": model,
                "url": ollama_url,
                "timeout": timeout,
            }
            st.rerun()
        return

    # ── running: process one file per rerun ──────────────────────────────────
    params  = st.session_state[_PARAMS]
    queue   = st.session_state[_QUEUE]
    results = st.session_state[_RESULTS]
    errors  = st.session_state[_ERRORS]
    total   = len(queue) + len(results) + len(errors)
    done    = len(results) + len(errors)

    if queue:
        # Show already-completed items
        for r, tok in results:
            st.success(f"✅ {r.name} · {tok:,} tokens")
        for err_name, err_msg in errors:
            st.error(f"❌ {err_name}: {err_msg}")

        col_prog, col_cancel = st.columns([5, 1])
        with col_prog:
            st.progress(done / total, text=f"{done}/{total} concluído(s) — processando arquivo {done + 1}…")
        with col_cancel:
            if st.button("✕ Cancelar", type="secondary", use_container_width=True):
                st.session_state[_RUNNING] = False
                st.session_state[_QUEUE]   = []
                st.rerun()

        name, file_bytes = queue.pop(0)
        with st.status(f"⏳ {name}", expanded=True) as status:
            try:
                st.write("Extraindo texto do PDF…")
                text, _ = extract_pdf_text(file_bytes)
                st.write(f"Enviando para `{params['model']}`… (pode demorar)")
                answers, tokens = analyze_text(text, params["questions"], params["model"], params["url"], params["timeout"])
                total_tokens = tokens["prompt"] + tokens["response"]
                st.caption(f"Tokens: {tokens['prompt']:,} entrada · {tokens['response']:,} saída · {total_tokens:,} total")
                st.write("Salvando resultado…")
                out_path = save_result(name, params["questions"], answers, params["collection"])
                results.append((out_path, total_tokens))
                status.update(label=f"✅ {name}", state="complete", expanded=False)
            except Exception as exc:
                errors.append((name, str(exc)))
                status.update(label=f"❌ {name}: {exc}", state="error", expanded=False)

        st.rerun()
        return

    # ── done ─────────────────────────────────────────────────────────────────
    st.session_state[_RUNNING] = False
    st.progress(1.0, text="Concluído.")

    for name, msg in errors:
        st.error(f"Erro em **{name}**: {msg}")

    if not results:
        return

    total_tok = sum(t for _, t in results)
    st.success(
        f"{len(results)} arquivo(s) analisado(s) e salvos em "
        f"`data/collections/{params['collection']}/` · {total_tok:,} tokens no total."
    )

    for out_path, tok in results:
        content = out_path.read_text(encoding="utf-8")
        with st.expander(f"📄 {out_path.name} · {tok:,} tokens"):
            col_preview, col_dl = st.columns([4, 1])
            with col_preview:
                st.markdown(content)
            with col_dl:
                st.download_button(
                    label="⬇️ Baixar",
                    data=content,
                    file_name=out_path.name,
                    mime="text/markdown",
                    key=f"dl_{out_path.stem}",
                )


def show():
    st.header("Análise de Artigos Científicos")

    uploaded_files, collection_name = _render_upload()
    st.divider()
    questions = _render_parameters()
    st.divider()
    ollama_url, model, timeout = _render_model()
    st.divider()
    _render_analysis(uploaded_files, collection_name, questions, ollama_url, model, timeout)
