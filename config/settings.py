"""
Configurações centralizadas e estrutura de persistência por run_id da aplicação SLD — Scientific Literature Decoder.
"""

from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Diretório base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

# Diretórios padrão de saída
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_MARKDOWN_DIR = DEFAULT_OUTPUT_DIR / "markdown"
DEFAULT_INDEX_DIR = DEFAULT_OUTPUT_DIR / "index"
DEFAULT_MANIFESTS_DIR = DEFAULT_OUTPUT_DIR / "manifests"
DEFAULT_EXPORTS_DIR = DEFAULT_OUTPUT_DIR / "exports"
DEFAULT_EXPERIMENTS_DIR = DEFAULT_OUTPUT_DIR / "experiments"
DEFAULT_LOGS_DIR = DEFAULT_OUTPUT_DIR / "logs"
DEFAULT_GOLD_STANDARD_PATH = DEFAULT_OUTPUT_DIR / "gold_standard.jsonl"

# Modelo de Sentence Embeddings Padrão (Ollama Local)
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
BATCH_SIZE = 32

# Modelo de LLM Padrão (Gemma 3 1B via Ollama Local)
DEFAULT_LLM_MODEL = "gemma3:1b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LLM_TEMPERATURE = 0.0
DEFAULT_LLM_NUM_CTX = 2048
DEFAULT_LLM_NUM_PREDICT = 192
DEFAULT_LLM_SEED = 42
DEFAULT_LLM_KEEP_ALIVE = "30m"
SYSTEM_PROMPT_VERSION = "llm_extraction_prompt_v1"

# Configurações Metodológicas de Segmentação
DEFAULT_MIN_WORDS = 8
DEFAULT_MIN_CHARACTERS = 40
DEFAULT_MAX_CHARACTERS = 500
DEFAULT_LONG_TEXT_STRATEGY = "chunk"  # "chunk", "truncate", "skip"
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_CHUNK_AGGREGATION = "maximum"  # "maximum" ou "mean"

# Configurações de Remoção de Referências
DEFAULT_REF_CONFIDENCE_THRESHOLD = 0.60

# Configurações de Busca Semântica & Agregação Multi-Âncora
DEFAULT_TOP_K = 20
DEFAULT_SIMILARITY_THRESHOLD = 0.50
DEFAULT_AGGREGATION_STRATEGY = "maximum"  # "maximum", "mean", "weighted_mean", "centroid"
DEFAULT_MINIMUM_RECALL_TARGET = 0.90
DEFAULT_RANDOM_SEED = 42
DEFAULT_SEMANTIC_BATCH_SIZE = 8192
DEFAULT_TOP_K_ANCHORS = 1
DEFAULT_SEMANTIC_DTYPE = "float32"

# Políticas de Sobrescrita de Arquivos
OVERWRITE_POLICIES = ["skip", "replace", "timestamp"]
DEFAULT_OVERWRITE_POLICY = "skip"

# Versão do Motor de Extração e da Aplicação
APP_NAME = "SLD — Scientific Literature Decoder"
APP_VERSION = "3.0.0"
EXTRACTION_ENGINE = "PyMuPDF"


def create_run_id() -> str:
    """Gera um identificador único de execução sequencial/temporal."""
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    return f"run_{timestamp}"


def get_run_dir_structure(base_output_dir: Path, run_id: str) -> Dict[str, Path]:
    """
    Cria e retorna a estrutura de diretórios padronizada sob output/<run_id>/.
    """
    run_dir = Path(base_output_dir).expanduser().resolve() / run_id
    subdirs = {
        "root": run_dir,
        "config": run_dir / "config",
        "markdown": run_dir / "markdown",
        "metadata": run_dir / "metadata",
        "paragraphs": run_dir / "paragraphs",
        "interim": run_dir / "interim",
        "processed": run_dir / "paragraphs",
        "embeddings": run_dir / "embeddings",
        "semantic_search": run_dir / "semantic_search",
        "annotations": run_dir / "annotations",
        "models": run_dir / "models",
        "classifications": run_dir / "classifications",
        "corpus_final": run_dir / "corpus_final",
        "llm": run_dir / "llm",
        "reports": run_dir / "reports",
        "logs": run_dir / "logs",
    }
    for d in subdirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return subdirs


def get_default_config() -> Dict[str, Any]:
    """Retorna um dicionário com todas as configurações padrão."""
    return {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "batch_size": BATCH_SIZE,
        "min_words": DEFAULT_MIN_WORDS,
        "min_characters": DEFAULT_MIN_CHARACTERS,
        "max_characters": DEFAULT_MAX_CHARACTERS,
        "long_text_strategy": DEFAULT_LONG_TEXT_STRATEGY,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "chunk_aggregation": DEFAULT_CHUNK_AGGREGATION,
        "ref_confidence_threshold": DEFAULT_REF_CONFIDENCE_THRESHOLD,
        "top_k": DEFAULT_TOP_K,
        "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
        "aggregation_strategy": DEFAULT_AGGREGATION_STRATEGY,
        "minimum_recall_target": DEFAULT_MINIMUM_RECALL_TARGET,
        "random_seed": DEFAULT_RANDOM_SEED,
        "overwrite_policy": DEFAULT_OVERWRITE_POLICY,
        "semantic_batch_size": DEFAULT_SEMANTIC_BATCH_SIZE,
        "top_k_anchors": DEFAULT_TOP_K_ANCHORS,
        "semantic_dtype": DEFAULT_SEMANTIC_DTYPE,
    }
