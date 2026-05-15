import streamlit as st

from service.extraction import extract_pdf_text
from service.rapid import RAPID_QUESTIONS, analyze_rapid, save_rapid_result
from service.transformation import DEFAULT_MODEL, DEFAULT_OLLAMA_URL, list_ollama_models, test_model

_QUEUE   = "_rapid_queue"
_RESULTS = "_rapid_results"
_ERRORS  = "_rapid_errors"
_RUNNING = "_rapid_running"
_PARAMS  = "_rapid_params"


def _render_upload() -> tuple[list, str]:
    st.subheader("1. Enviar artigos em PDF")
    collection_name = st.text_input(
        "Nome da coleção",
        placeholder="Ex: vulnerabilidade_2024",
        help="Os resultados serão salvos em data/collections/{nome}/",
        key="rapid_collection",
    )
    files = st.file_uploader(
        "Selecione um ou mais arquivos PDF",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="rapid_uploader",
    )
    if files:
        st.caption(f"{len(files)} arquivo(s) selecionado(s)")
    return files or [], collection_name.strip()


def _render_model() -> tuple[str, str, int]:
    st.subheader("2. Modelo Ollama")
    ollama_url = st.text_input(
        "Host Ollama",
        value=DEFAULT_OLLAMA_URL,
        help="Use o IP da máquina remota ou 'http://localhost:11434' para uso local.",
        key="rapid_ollama_url",
    )
    models = list_ollama_models(ollama_url)
    if models:
        default_idx = next((i for i, m in enumerate(models) if "gemma4" in m or "gemma" in m), 0)
        model = st.selectbox("Modelo disponível", options=models, index=default_idx, key="rapid_model")
    else:
        st.warning(f"Ollama não encontrado em `{ollama_url}`. Verifique se o serviço está em execução.")
        model = st.text_input("Modelo (digitar manualmente)", value=DEFAULT_MODEL, key="rapid_model_manual")

    timeout = st.slider(
        "Tempo limite por artigo (s)",
        min_value=60, max_value=600, value=300, step=30,
        help="Aumente para modelos lentos ou artigos longos.",
        key="rapid_timeout",
    )

    if st.button("🔌 Testar modelo", use_container_width=True, key="rapid_test"):
        with st.spinner(f"Testando `{model}`…"):
            try:
                reply = test_model(ollama_url, model)
                st.success(f"Modelo respondeu: `{reply}`")
            except Exception as e:
                st.error(f"Falha: {e}")

    return ollama_url, model, timeout


def _render_analysis(uploaded_files: list, collection_name: str, ollama_url: str, model: str, timeout: int) -> None:
    st.subheader("3. Executar extração")

    if not uploaded_files:
        st.info("Envie ao menos um PDF na etapa 1.")
        return
    if not collection_name:
        st.warning("Defina o nome da coleção na etapa 1.")
        return

    running = st.session_state.get(_RUNNING, False)

    # ── idle ─────────────────────────────────────────────────────────────────
    if not running:
        st.write(
            f"**{len(uploaded_files)}** arquivo(s) · "
            f"Modelo: `{model}` · Coleção: `{collection_name}`"
        )
        st.caption(
            f"Schema fixo: {len(RAPID_QUESTIONS)} campos de vulnerabilidade extraídos automaticamente."
        )
        if st.button("⚡ Executar extração rápida", type="primary", use_container_width=True):
            st.session_state[_QUEUE]   = [(f.name, f.getvalue()) for f in uploaded_files]
            st.session_state[_RESULTS] = []
            st.session_state[_ERRORS]  = []
            st.session_state[_RUNNING] = True
            st.session_state[_PARAMS]  = {
                "collection": collection_name,
                "model": model,
                "url": ollama_url,
                "timeout": timeout,
                "total": len(uploaded_files),
            }
            st.rerun()
        return

    # ── running ───────────────────────────────────────────────────────────────
    params  = st.session_state[_PARAMS]
    queue   = st.session_state[_QUEUE]
    results = st.session_state[_RESULTS]
    errors  = st.session_state[_ERRORS]
    total   = params["total"]
    done    = len(results) + len(errors)

    if queue:
        for r, tok in results:
            st.success(f"✅ {r.name} · {tok:,} tokens")
        for err_name, err_msg in errors:
            st.error(f"❌ {err_name}: {err_msg}")

        col_prog, col_cancel = st.columns([5, 1])
        with col_prog:
            st.progress(done / total, text=f"{done}/{total} concluído(s) — processando artigo {done + 1}…")
        with col_cancel:
            if st.button("✕ Cancelar", type="secondary", use_container_width=True, key="rapid_cancel"):
                st.session_state[_RUNNING] = False
                st.session_state[_QUEUE]   = []
                st.rerun()

        name, file_bytes = queue.pop(0)
        with st.status(f"⏳ {name}", expanded=True) as status:
            try:
                st.write("Extraindo texto do PDF…")
                text, _ = extract_pdf_text(file_bytes)
                st.write(f"Enviando para `{params['model']}`… (pode demorar)")
                answers, tokens = analyze_rapid(text, params["model"], params["url"], params["timeout"])
                tok_total = tokens["prompt"] + tokens["response"]
                st.caption(
                    f"Tokens: {tokens['prompt']:,} entrada · "
                    f"{tokens['response']:,} saída · {tok_total:,} total"
                )
                st.write("Salvando resultado…")
                out_path = save_rapid_result(
                    name, answers, params["collection"],
                    article_index=done + 1,
                    total_articles=total,
                )
                results.append((out_path, tok_total))
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
        f"{len(results)} artigo(s) extraído(s) e salvos em "
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
                    key=f"rapid_dl_{out_path.stem}",
                )


def show():
    st.header("Extração Rápida de Vulnerabilidade")
    st.caption(
        "Schema fixo otimizado para extração de conceitos de vulnerabilidade. "
        "Sem configuração de parâmetros — envie os PDFs e execute."
    )

    uploaded_files, collection_name = _render_upload()
    st.divider()
    ollama_url, model, timeout = _render_model()
    st.divider()
    _render_analysis(uploaded_files, collection_name, ollama_url, model, timeout)
