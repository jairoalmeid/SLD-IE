"""
Gerenciador de Progresso em Lotes, Dois Níveis de Progresso, Checklist Visual e Cancelamento para o SLD.
"""

import time
import streamlit as st
from typing import List, Dict, Optional, Tuple


def format_duration(seconds: float) -> str:
    """Formata segundos em uma string HH:MM:SS ou MM:SS."""
    if seconds < 0:
        seconds = 0
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class ProgressTracker:
    """
    Rastreia o progresso de operações iterativas com 2 níveis (geral + etapa atual),
    checklist visual de etapas, cálculo de ETA por taxa média, contagem de sucessos/pulados/erros
    e suporte a cancelamento controlado.
    """

    def __init__(
        self,
        title: str,
        total: int,
        steps: Optional[List[str]] = None,
        update_interval: int = 1,
        cancel_key: Optional[str] = None
    ):
        self.title = title
        self.total = max(1, total)
        self.update_interval = update_interval
        self.start_time = time.time()
        self.cancel_key = cancel_key or f"cancel_btn_{id(self)}"

        self.steps = steps or ["Ingestão PDF", "Limpeza & Referências", "Vetorização & Índice"]
        self.current_step_idx = 0

        self.success_count = 0
        self.skipped_count = 0
        self.error_count = 0

        self.status_container = st.status(f"Processando: {title}...", expanded=True)
        with self.status_container:
            self.overall_label = st.markdown(f"**Progresso Geral:** 0 de {self.total:,}".replace(",", "."))
            self.overall_bar = st.progress(0.0)

            self.step_label = st.markdown(f"**Etapa Atual:** {self.steps[0]}")
            self.step_bar = st.progress(0.0)

            self.stepper_box = st.empty()
            self._render_stepper()

            self.detail_text = st.empty()

            if st.button("⏹️ Cancelar Operação", key=self.cancel_key, help="Interrompe o lote com segurança, salvando o progresso acumulado até o momento."):
                st.session_state[f"cancelled_{self.cancel_key}"] = True

    def _render_stepper(self):
        """Renderiza a lista visual das etapas com marcas de progresso."""
        stepper_items = []
        for idx, step_name in enumerate(self.steps):
            if idx < self.current_step_idx:
                stepper_items.append(f"✓ {step_name}")
            elif idx == self.current_step_idx:
                stepper_items.append(f"→ **{step_name}** (em andamento)")
            else:
                stepper_items.append(f"○ {step_name}")

        self.stepper_box.caption(" | ".join(stepper_items))

    def set_step(self, step_idx: int, step_name: Optional[str] = None):
        """Avança para uma etapa específica no checklist visual."""
        if 0 <= step_idx < len(self.steps):
            self.current_step_idx = step_idx
            if step_name:
                self.steps[step_idx] = step_name
            self.step_label.markdown(f"**Etapa Atual:** {self.steps[self.current_step_idx]}")
            self.step_bar.progress(0.0)
            self._render_stepper()

    def is_cancelled(self) -> bool:
        """Verifica se o usuário clicou no botão de cancelamento."""
        return st.session_state.get(f"cancelled_{self.cancel_key}", False)

    def update(
        self,
        processed: int,
        current_item: str = "",
        step_processed: Optional[int] = None,
        step_total: Optional[int] = None,
        successes: Optional[int] = None,
        skipped: Optional[int] = None,
        errors: Optional[int] = None,
        force: bool = False
    ):
        """
        Atualiza as barras de progresso geral e da etapa, métricas de estatística e ETA.
        """
        if successes is not None:
            self.success_count = successes
        if skipped is not None:
            self.skipped_count = skipped
        if errors is not None:
            self.error_count = errors

        if not force and processed % self.update_interval != 0 and processed < self.total:
            return

        now = time.time()
        elapsed = now - self.start_time
        pct_overall = min(1.0, processed / self.total)
        self.overall_bar.progress(pct_overall)

        item_str = f" | Item: `{current_item}`" if current_item else ""
        self.overall_label.markdown(
            f"**Processamento Geral:** {processed:,} de {self.total:,} ({pct_overall*100:.1f}%){item_str}".replace(",", ".")
        )

        if step_processed is not None and step_total is not None and step_total > 0:
            pct_step = min(1.0, step_processed / step_total)
            self.step_bar.progress(pct_step)
            self.step_label.markdown(
                f"**Etapa Atual ({self.steps[self.current_step_idx]}):** {step_processed:,} de {step_total:,} ({pct_step*100:.1f}%)".replace(",", ".")
            )

        # Cálculo da estimativa de tempo (ETA) por taxa média após os primeiros itens
        items_done = max(1, processed)
        rate = items_done / elapsed if elapsed > 0.5 else 0.0
        remaining_items = max(0, self.total - processed)
        eta_sec = (remaining_items / rate) if (rate > 0 and processed >= 2) else 0.0

        eta_str = format_duration(eta_sec) if (rate > 0 and processed >= 2) else "calculando..."
        elapsed_str = format_duration(elapsed)

        self.detail_text.caption(
            f"**Decorrido:** {elapsed_str} | **Restante est.:** {eta_str} | "
            f"**Sucessos:** {self.success_count} | **Pulados:** {self.skipped_count} | **Erros:** {self.error_count}"
        )

        self.status_container.update(
            label=f"Processando: {self.title} ({processed}/{self.total} — {pct_overall*100:.0f}%)",
            state="running"
        )

    def complete(self, message: str = "Operação concluída com sucesso.", show_balloons: bool = True):
        """Finaliza o rastreador alterando o status para concluído."""
        self.overall_bar.progress(1.0)
        self.step_bar.progress(1.0)
        self.current_step_idx = len(self.steps) - 1
        self._render_stepper()

        elapsed = time.time() - self.start_time
        elapsed_str = format_duration(elapsed)

        self.overall_label.markdown(
            f"**{self.title} — Concluído** ({self.total:,} itens em {elapsed_str})".replace(",", ".")
        )
        self.detail_text.caption(
            f"**Tempo total:** {elapsed_str} | **Sucessos:** {self.success_count} | "
            f"**Pulados:** {self.skipped_count} | **Erros:** {self.error_count}"
        )
        self.status_container.update(
            label=f"✓ {self.title} — Concluído ({elapsed_str})",
            state="complete",
            expanded=False
        )
        st.toast(f"✓ {message}")
        if show_balloons:
            st.balloons()
