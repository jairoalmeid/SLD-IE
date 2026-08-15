"""
Serviço de Controle de Carga e Taxa de Requisições por Minuto (Rate Limiter).
Evita sobrecarga computacional durante processamentos locais intensivos no Ollama.
"""

import time
from typing import Optional, Tuple, Callable


class RateLimiter:
    """Gerencia a frequência de requisições enviadas ao LLM com base no parâmetro RPM."""

    def __init__(self, rpm: Optional[float] = 10.0):
        """
        rpm: Limite máximo de requisições por minuto.
             Se None ou <= 0, funciona no modo 'Sem Limite'.
        """
        self.rpm = float(rpm) if (rpm is not None and rpm > 0) else 0.0
        self.last_request_time: float = 0.0
        self.total_wait_time: float = 0.0

    @property
    def is_enabled(self) -> bool:
        return self.rpm > 0.0

    @property
    def min_interval_seconds(self) -> float:
        """Calcula Δt_min = 60 / RPM em segundos."""
        if not self.is_enabled:
            return 0.0
        return 60.0 / self.rpm

    def get_theoretical_remaining_seconds(self, n_remaining: int) -> float:
        """Calcula T_min = (N_remaining / RPM) * 60 em segundos."""
        if not self.is_enabled or n_remaining <= 0:
            return 0.0
        return (n_remaining / self.rpm) * 60.0

    def wait_if_needed(self, stop_checker: Optional[Callable[[], bool]] = None) -> float:
        """
        Aguarda o tempo necessário para respeitar o limite de RPM.
        Retorna o tempo efetivo de espera em segundos.
        """
        if not self.is_enabled or self.last_request_time == 0.0:
            self.last_request_time = time.time()
            return 0.0

        elapsed_since_last = time.time() - self.last_request_time
        required_wait = self.min_interval_seconds - elapsed_since_last

        if required_wait <= 0.0:
            self.last_request_time = time.time()
            return 0.0

        waited = 0.0
        step = 0.2
        while waited < required_wait:
            if stop_checker and stop_checker():
                break
            sleep_chunk = min(step, required_wait - waited)
            time.sleep(sleep_chunk)
            waited += sleep_chunk

        self.total_wait_time += waited
        self.last_request_time = time.time()
        return waited
