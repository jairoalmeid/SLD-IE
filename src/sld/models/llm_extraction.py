"""
Modelos Pydantic v2 para estruturação de saída do LLM (Ollama), auditoria e persistência.
Define o Schema Canônico para extração conceitual de literatura científica.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator


class ExtractedItem(BaseModel):
    """Representa um elemento conceitual extraído associado à sua evidência textual."""
    type: str = Field(..., description="Tipo do conteúdo: concept, definition, determinant, dimension, causal_relation, property, mechanism, consequence")
    value: str = Field(..., description="Valor do conteúdo extraído (ex: 'low income')")
    normalized_value: Optional[str] = ""
    evidence: Optional[str] = ""
    evidence_valid: bool = False


class ExtractionOutput(BaseModel):
    """Schema JSON Canônico e Genérico para extração de informação em literatura científica."""
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    concepts: List[str] = Field(default_factory=list, description="Conceitos principais identificados no parágrafo")
    definitions: List[str] = Field(default_factory=list, description="Definições ou conceituações explícitas")
    determinants: List[str] = Field(default_factory=list, description="Fatores determinantes ou condicionantes")
    dimensions: List[str] = Field(default_factory=list, description="Tipos, dimensões ou categorias analíticas")
    causal_relations: List[str] = Field(default_factory=list, description="Relações de causa, efeito ou influência")
    properties: List[str] = Field(default_factory=list, description="Características ou propriedades associadas")
    mechanisms: List[str] = Field(default_factory=list, description="Mecanismos de atuação ou processos")
    consequences: List[str] = Field(default_factory=list, description="Impactos ou consequências observadas")
    evidence: List[str] = Field(default_factory=list, description="Citações textuais literais extraídas")
    structured_items: List[ExtractedItem] = Field(default_factory=list, description="Itens estruturados vinculando conteúdo à evidência")
    confidence: float = Field(default=1.0, description="Nível estimado de confiança (0.0 a 1.0)")

    @model_validator(mode="before")
    @classmethod
    def alias_legacy_and_alternative_fields(cls, data: Any) -> Any:
        """MAPEAMENTO CANÔNICO: Resolve a causa raiz das colunas vazias mapeando aliases legados."""
        if not isinstance(data, dict):
            return data

        d = dict(data)

        # Mapeia conceitos
        if "concepts" not in d:
            if "main_concept" in d and d["main_concept"]:
                val = d["main_concept"]
                d["concepts"] = [val] if isinstance(val, str) else list(val)
            elif "normalized_concept" in d and d["normalized_concept"]:
                val = d["normalized_concept"]
                d["concepts"] = [val] if isinstance(val, str) else list(val)

        # Mapeia dimensões
        if "dimensions" not in d:
            if "vulnerability_dimensions" in d:
                d["dimensions"] = d.get("vulnerability_dimensions") or []
            elif "types_dimensions" in d:
                d["dimensions"] = d.get("types_dimensions") or []

        # Mapeia determinantes / fatores
        if "determinants" not in d:
            if "factors" in d:
                d["determinants"] = d.get("factors") or []
            elif "driver_factors" in d:
                d["determinants"] = d.get("driver_factors") or []

        # Mapeia relações causais / causas
        if "causal_relations" not in d:
            if "causes" in d:
                d["causal_relations"] = d.get("causes") or []
            elif "relations" in d:
                d["causal_relations"] = d.get("relations") or []

        return d


class LLMParagraphResult(BaseModel):
    """Resultado auditável completo de extração conceitual por parágrafo no Corpus Final."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    run_id: str
    article_id: str
    paragraph_id: str
    text: str
    cosine_similarity: float = 0.0
    class_probabilities: Dict[str, float] = Field(default_factory=dict)
    class_labels: List[str] = Field(default_factory=list)

    # Parâmetros de Inferência LLM
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    llm_model_tag: str = "qwen2.5:7b"
    endpoint: str = "http://localhost:11434"
    prompt_version: str = "llm_extraction_prompt_v2"
    schema_version: str = "v2_canonical"
    temperature: float = 0.0
    num_ctx: int = 2048
    num_predict: int = 256
    seed: int = 42

    # Controle de Carga (Rate Limiting)
    requests_per_minute: float = 10.0
    rate_limit_wait_time: float = 0.0

    # Resultados e Validações
    llm_output: ExtractionOutput = Field(default_factory=ExtractionOutput)
    raw_response: str = ""
    json_valid: bool = False
    schema_valid: bool = False
    evidence_valid: bool = False
    cache_hit: bool = False

    # Métricas de Desempenho e Auditoria
    prompt_tokens: int = 0
    output_tokens: int = 0
    processing_time: float = 0.0
    processed_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Estado e Erros
    status: str = "completed"  # "completed", "failed", "skipped"
    error_message: Optional[str] = None
    retry_count: int = 0
