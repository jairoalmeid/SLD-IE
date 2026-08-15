"""
Configuração centralizada de registros de execução (logging).
"""

import logging
from pathlib import Path
from typing import Optional
from config.settings import DEFAULT_LOGS_DIR


def setup_logger(log_dir: Optional[Path] = None, log_level: int = logging.INFO) -> logging.Logger:
    """
    Configura o logger principal da aplicação SLD para saída em arquivo e console.
    """
    logger = logging.getLogger("sld")
    logger.setLevel(log_level)

    # Evita adicionar múltiplos handlers se já estiver configurado
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Stream Handler (console)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    target_log_dir = log_dir or DEFAULT_LOGS_DIR
    try:
        target_log_dir.mkdir(parents=True, exist_ok=True)
        log_file = target_log_dir / "sld.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        console_handler.emit(logging.LogRecord(
            "sld", logging.WARNING, __file__, 0,
            f"Não foi possível criar o arquivo de log em {target_log_dir}: {e}", None, None
        ))

    return logger
