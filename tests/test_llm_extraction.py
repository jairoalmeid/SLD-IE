"""
Testes unitários para o módulo de extração conceitual por LLM (Ollama),
RateLimiter (RPM), Schema Canônico, Validação de Evidência e Corpus Refinado.
"""

import json
import pytest
from pathlib import Path
from src.sld.models.classification import ParagraphRecord
from src.sld.models.llm_extraction import ExtractionOutput, LLMParagraphResult, ExtractedItem
from src.sld.llm.llm_provider import MockLLMProvider, OllamaProvider
from src.sld.llm.rate_limiter import RateLimiter
from src.sld.llm.extraction_service import (
    LLMExtractionService,
    compute_cache_key,
    validate_textual_evidence,
)


def test_mock_llm_provider_connection_and_extraction():
    provider = MockLLMProvider()
    assert provider.check_connection() is True
    assert provider.is_model_installed("qwen2.5:7b") is True
    assert len(provider.list_models()) > 0

    record = ParagraphRecord(
        paragraph_id="DOC001_P000001",
        article_id="DOC001",
        text="Investimentos em infraestrutura e gestão são fatores determinantes para a resiliência.",
        predicted_labels=["2 — Fator determinante"],
        predicted_probabilities={"determinant": 0.91}
    )

    raw_json, metrics = provider.extract_paragraph(record)
    assert raw_json is not None
    data = json.loads(raw_json)
    output = ExtractionOutput(**data)
    assert "resiliência" in output.concepts
    assert "investimento em infraestrutura" in output.determinants
    assert metrics["prompt_tokens"] > 0


def test_canonical_schema_alias_mapping():
    """Testa a resolução da causa raiz das colunas vazias via alias de campos legados."""
    legacy_json = {
        "main_concept": "resiliência sistêmica",
        "vulnerability_dimensions": ["social", "econômica"],
        "factors": ["pobreza", "isolamento"],
        "causes": ["mudanças climáticas"],
        "evidence": ["resiliência sistêmica"]
    }

    output = ExtractionOutput(**legacy_json)
    assert output.concepts == ["resiliência sistêmica"]
    assert output.dimensions == ["social", "econômica"]
    assert output.determinants == ["pobreza", "isolamento"]
    assert output.causal_relations == ["mudanças climáticas"]


def test_validate_textual_evidence_positive_and_negative():
    original_text = "Household low income and social isolation increase severe disaster risk in urban areas."

    valid_snippets = ["low income", "social isolation", "disaster risk"]
    invalid_snippets = ["climate change", "high temperature"]

    is_valid_pos, pos_details = validate_textual_evidence(original_text, valid_snippets)
    is_valid_neg, neg_details = validate_textual_evidence(original_text, invalid_snippets)

    assert is_valid_pos is True
    assert is_valid_neg is False
    assert all(d["valid"] for d in pos_details)
    assert not any(d["valid"] for d in neg_details)


def test_rate_limiter_interval_math():
    limiter = RateLimiter(rpm=10.0)
    assert limiter.is_enabled is True
    assert limiter.min_interval_seconds == 6.0
    assert limiter.get_theoretical_remaining_seconds(10) == 60.0

    disabled_limiter = RateLimiter(rpm=0.0)
    assert disabled_limiter.is_enabled is False
    assert disabled_limiter.min_interval_seconds == 0.0


def test_llm_extraction_service_checkpoint_and_refined_corpus(tmp_path: Path):
    llm_dir = tmp_path / "llm"
    mock_provider = MockLLMProvider()
    service = LLMExtractionService(llm_provider=mock_provider, llm_dir=llm_dir, run_id="run_test_01", rpm_limit=0.0)

    records = [
        ParagraphRecord(
            paragraph_id="DOC001_P000001",
            article_id="DOC001",
            text="Trecho relevante sobre resiliência e infraestrutura.",
            status="MODEL_RELEVANT",
            predicted_labels=["1 — Definição"]
        ),
        ParagraphRecord(
            paragraph_id="DOC002_P000001",
            article_id="DOC002",
            text="Trecho relevante 2 sobre fatores determinantes.",
            status="MODEL_RELEVANT",
            predicted_labels=["2 — Fator determinante"]
        ),
    ]

    results = service.process_corpus(records)

    assert len(results) == 2
    assert (llm_dir / "llm_results.jsonl").exists()

    refined = service.get_refined_corpus(require_valid_evidence=True)
    assert len(refined) == 2
