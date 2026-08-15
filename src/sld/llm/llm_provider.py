"""
Camada abstrata de LLM para extração conceitual estruturada sobre o Corpus Final via Ollama.
Desacoplada de modelos específicos, suportando listagem dinâmica de modelos instalados.
"""

import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import requests
from config.settings import (
    DEFAULT_LLM_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_NUM_CTX,
    DEFAULT_LLM_NUM_PREDICT,
    DEFAULT_LLM_SEED,
    DEFAULT_LLM_KEEP_ALIVE,
    SYSTEM_PROMPT_VERSION,
)
from src.sld.models.classification import ParagraphRecord
from src.sld.models.llm_extraction import ExtractionOutput


SYSTEM_EXTRACTION_PROMPT = """You are performing structured information extraction from scientific literature for an academic research project.

Analyze only the supplied paragraph.

Use exclusively information explicitly supported by the paragraph text.

Do not add external knowledge. Do not speculate. Do not expand concepts beyond what is supported by the source text.

The paragraph has already been selected by a semantic retrieval and supervised classification pipeline.

Your task is to extract, categorize, and normalize the conceptual information contained in it into the canonical fields:
- concepts: main research concepts discussed
- definitions: explicit definitions or conceptualizations
- determinants: determinant factors, drivers, or conditioning elements
- dimensions: categories, types, or analytical dimensions
- causal_relations: cause-and-effect or influential relations
- properties: attributes or properties
- mechanisms: processes or action mechanisms
- consequences: observed impacts or outcomes
- evidence: literal text quotes extracted directly from the paragraph that support each item

Return only valid JSON matching the schema format.
If information for a field is absent, return an empty array []."""


class LLMProvider(ABC):
    """Interface abstrata desacoplada para provedores de LLM."""

    @abstractmethod
    def analyze(self, paragraphs: List[ParagraphRecord], task_prompt: str) -> str:
        """Sintetiza uma análise em lote sobre o corpus final."""
        pass

    @abstractmethod
    def extract_paragraph(
        self,
        record: ParagraphRecord,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Extrai informação conceitual estruturada de um parágrafo individual."""
        pass

    @abstractmethod
    def check_connection(self) -> bool:
        """Verifica se o serviço LLM está acessível."""
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """Lista modelos efetivamente instalados localmente."""
        pass

    @abstractmethod
    def is_model_installed(self, model_name: str) -> bool:
        """Verifica se o modelo especificado está baixado e disponível."""
        pass


class OllamaProvider(LLMProvider):
    """Provedor dinâmico para modelos locais executados via API Ollama."""

    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def analyze(self, paragraphs: List[ParagraphRecord], task_prompt: str) -> str:
        """Método de síntese legado para compatibilidade com o pipeline."""
        if not paragraphs:
            return "Nenhum parágrafo no corpus final para análise por LLM."
        return f"[Ollama {self.model_name}] Análise de {len(paragraphs)} parágrafos concluída."

    def check_connection(self) -> bool:
        """Verifica se o servidor Ollama está ativo em /api/tags."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Lista os nomes de modelos efetivamente instalados no Ollama local."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception:
            pass
        return []

    def is_model_installed(self, model_name: Optional[str] = None) -> bool:
        """Verifica se o modelo especificado está instalado localmente."""
        target = model_name or self.model_name
        installed = self.list_models()
        target_clean = target.lower().strip()
        return any(target_clean in m.lower() for m in installed)

    def extract_paragraph(
        self,
        record: ParagraphRecord,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Executa a extração estruturada para um único parágrafo do Corpus Final via Ollama.
        Retorna (raw_json_string, metrics_dict).
        """
        opts = {
            "temperature": DEFAULT_LLM_TEMPERATURE,
            "num_ctx": DEFAULT_LLM_NUM_CTX,
            "num_predict": DEFAULT_LLM_NUM_PREDICT,
            "seed": DEFAULT_LLM_SEED,
        }
        if options:
            opts.update(options)

        model_to_use = options.get("model_name", self.model_name) if options else self.model_name

        prob_str = ", ".join(f"{k}: {v:.2f}" for k, v in record.predicted_probabilities.items()) if record.predicted_probabilities else "N/A"
        labels_str = ", ".join(record.predicted_labels) if record.predicted_labels else "N/A"

        user_prompt = (
            f"paragraph_id: {record.paragraph_id}\n"
            f"article_id: {record.article_id}\n"
            f"Classes identificadas pela Regressão Logística: [{labels_str}]\n"
            f"Probabilidades: [{prob_str}]\n"
            f"Similaridade Cosseno: {record.semantic_score or 0.0:.4f}\n\n"
            f"Texto do Parágrafo:\n\"\"\"{record.text}\"\"\"\n\n"
            f"Extraia e responda estritamente no formato JSON correspondente ao schema especificado:"
        )

        payload = {
            "model": model_to_use,
            "system": SYSTEM_EXTRACTION_PROMPT,
            "prompt": user_prompt,
            "stream": False,
            "format": "json",
            "keep_alive": DEFAULT_LLM_KEEP_ALIVE,
            "options": opts,
        }

        try:
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            raw_response = data.get("response", "{}")

            metrics = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
                "prompt_eval_duration": data.get("prompt_eval_duration", 0),
                "eval_duration": data.get("eval_duration", 0),
                "total_duration": data.get("total_duration", 0),
            }
            return raw_response, metrics
        except Exception as e:
            raise RuntimeError(f"Erro ao comunicar com Ollama ({model_to_use}): {str(e)}")


class MockLLMProvider(LLMProvider):
    """Provedor sintético determinístico para testes unitários sem Ollama ativo."""

    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name

    def analyze(self, paragraphs: List[ParagraphRecord], task_prompt: str) -> str:
        return f"[Mock LLM Analysis] Analisados {len(paragraphs)} parágrafos para a tarefa: {task_prompt}."

    def check_connection(self) -> bool:
        return True

    def list_models(self) -> List[str]:
        return [self.model_name, "qwen2.5:7b", "llama3.1:8b"]

    def is_model_installed(self, model_name: str) -> bool:
        return True

    def extract_paragraph(
        self,
        record: ParagraphRecord,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:

        sample_words = record.text.split()
        evidence_snippet = " ".join(sample_words[:min(5, len(sample_words))]) if sample_words else record.text

        mock_data = ExtractionOutput(
            concepts=["resiliência", "governança"],
            definitions=["capacidade de recuperação e adaptação sistêmica"],
            determinants=["investimento em infraestrutura"],
            dimensions=["social", "econômica"],
            causal_relations=["boa governança aumenta a resiliência"],
            properties=["dinâmica e multifatorial"],
            mechanisms=["adaptação institucional"],
            consequences=["redução de riscos"],
            evidence=[evidence_snippet],
            confidence=0.95
        )

        metrics = {
            "prompt_tokens": 150,
            "output_tokens": 80,
            "total_duration": 120000000,
        }
        return mock_data.model_dump_json(), metrics
