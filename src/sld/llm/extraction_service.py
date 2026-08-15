"""
Gerenciador do pipeline de extração por LLM (Ollama) sobre o Corpus Final.
Suporta checkpointing incremental, cache por SHA-256, retomada (resume),
controle de taxa por requisições por minuto (Rate Limiting) e métricas de evidência textual.
"""

import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
import pandas as pd
from config.settings import (
    DEFAULT_LLM_MODEL,
    SYSTEM_PROMPT_VERSION,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_NUM_CTX,
    DEFAULT_LLM_NUM_PREDICT,
    DEFAULT_LLM_SEED,
)
from src.sld.models.classification import ParagraphRecord
from src.sld.models.llm_extraction import ExtractionOutput, LLMParagraphResult, ExtractedItem
from src.sld.llm.llm_provider import LLMProvider, OllamaProvider
from src.sld.llm.rate_limiter import RateLimiter
from src.sld.utils.files import ensure_directory


def compute_cache_key(
    text: str,
    model_name: str,
    prompt_version: str,
    options: Dict[str, Any]
) -> str:
    """Calcula o hash de cache determinístico com base no texto, modelo, prompt e hiperparâmetros."""
    opts_str = json.dumps(options, sort_keys=True)
    raw = f"{text}||{model_name}||{prompt_version}||{opts_str}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_textual_evidence(original_text: str, evidence_snippets: List[str]) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Valida se cada citação de evidência retornada pelo LLM está contida literalmente 
    no texto do parágrafo original (com normalização de espaços e caixa).

    Retorna (all_valid, item_validations_list).
    """
    if not evidence_snippets:
        return True, []

    orig_clean = " ".join(original_text.lower().split())
    validations = []
    all_valid = True

    for snippet in evidence_snippets:
        if not snippet or not snippet.strip():
            continue
        snip_clean = " ".join(snippet.lower().split())
        is_valid = snip_clean in orig_clean
        if not is_valid:
            all_valid = False

        validations.append({
            "snippet": snippet,
            "valid": is_valid
        })

    return all_valid, validations


class LLMExtractionService:
    """Gerencia a fila de inferência por LLM, RateLimiter, persistência e estatísticas."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        llm_dir: Optional[Path] = None,
        run_id: str = "run_default",
        rpm_limit: Optional[float] = 10.0
    ):
        self.provider = llm_provider or OllamaProvider()
        self.llm_dir = Path(llm_dir).expanduser().resolve() if llm_dir else Path("./output/llm")
        ensure_directory(self.llm_dir)
        self.run_id = run_id

        self.rate_limiter = RateLimiter(rpm=rpm_limit)
        self.jsonl_path = self.llm_dir / "llm_results.jsonl"
        self.parquet_path = self.llm_dir / "llm_results.parquet"
        self.stats_path = self.llm_dir / "llm_statistics.json"
        self.errors_path = self.llm_dir / "llm_errors.csv"

        self.stop_requested: bool = False
        self.disk_cache: Dict[str, LLMParagraphResult] = {}
        self.completed_ids: set = set()
        self._load_existing_checkpoint()

    def set_rpm_limit(self, rpm: Optional[float]):
        """Atualiza dinamicamente o limite de requisições por minuto."""
        self.rate_limiter = RateLimiter(rpm=rpm)

    def request_stop(self) -> None:
        """Sinaliza a parada segura do processamento no próximo parágrafo ou timer."""
        self.stop_requested = True

    def _load_existing_checkpoint(self) -> None:
        """Lê os resultados já processados do arquivo JSONL incremental para suporte a Resume e Cache."""
        if not self.jsonl_path.exists():
            return

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    res = LLMParagraphResult(**data)
                    self.completed_ids.add(res.paragraph_id)
                    cache_key = compute_cache_key(
                        res.text,
                        res.llm_model,
                        res.prompt_version,
                        {
                            "temperature": res.temperature,
                            "num_ctx": res.num_ctx,
                            "num_predict": res.num_predict,
                            "seed": res.seed
                        }
                    )
                    self.disk_cache[cache_key] = res
                except Exception:
                    continue

    def process_corpus(
        self,
        paragraphs: List[ParagraphRecord],
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, int, LLMParagraphResult, Dict[str, Any]], None]] = None
    ) -> List[LLMParagraphResult]:
        """
        Processa sequencialmente o Corpus Final aplicando RateLimiting, Checkpointing e Validação.
        """
        self.stop_requested = False
        opts = {
            "temperature": DEFAULT_LLM_TEMPERATURE,
            "num_ctx": DEFAULT_LLM_NUM_CTX,
            "num_predict": DEFAULT_LLM_NUM_PREDICT,
            "seed": DEFAULT_LLM_SEED,
        }
        if options:
            opts.update(options)

        model_name = options.get("model_name", getattr(self.provider, "model_name", DEFAULT_LLM_MODEL)) if options else getattr(self.provider, "model_name", DEFAULT_LLM_MODEL)
        total_items = len(paragraphs)
        results: List[LLMParagraphResult] = []

        with open(self.jsonl_path, "a", encoding="utf-8") as jsonl_file:
            for idx, rec in enumerate(paragraphs, start=1):
                if self.stop_requested:
                    break

                cache_key = compute_cache_key(rec.text, model_name, SYSTEM_PROMPT_VERSION, opts)

                # Verifica se já está no cache de disco
                if cache_key in self.disk_cache:
                    cached_res = self.disk_cache[cache_key]
                    cached_res.cache_hit = True
                    results.append(cached_res)
                    if progress_callback:
                        progress_callback(idx, total_items, cached_res, {"waiting": False, "wait_time": 0.0})
                    continue

                # Aplica o Controle de Carga (Rate Limiting)
                wait_time = self.rate_limiter.wait_if_needed(stop_checker=lambda: self.stop_requested)
                if self.stop_requested:
                    break

                start_t = time.time()
                try:
                    raw_response, metrics = self.provider.extract_paragraph(rec, options=opts)
                    proc_t = time.time() - start_t

                    # Tenta parsear JSON no Schema Canônico
                    json_ok = False
                    schema_ok = False
                    output_obj = ExtractionOutput()

                    try:
                        parsed_json = json.loads(raw_response)
                        json_ok = True
                        output_obj = ExtractionOutput(**parsed_json)
                        schema_ok = True
                    except Exception:
                        pass

                    # Valida evidência textual
                    ev_valid, ev_details = validate_textual_evidence(rec.text, output_obj.evidence)

                    res = LLMParagraphResult(
                        run_id=self.run_id,
                        article_id=rec.article_id,
                        paragraph_id=rec.paragraph_id,
                        text=rec.text,
                        cosine_similarity=rec.semantic_score or 0.0,
                        class_probabilities=rec.predicted_probabilities or {},
                        class_labels=rec.predicted_labels or [],
                        llm_provider="ollama",
                        llm_model=model_name,
                        llm_model_tag=model_name,
                        endpoint=getattr(self.provider, "base_url", "http://localhost:11434"),
                        prompt_version=SYSTEM_PROMPT_VERSION,
                        schema_version="v2_canonical",
                        temperature=float(opts.get("temperature", 0.0)),
                        num_ctx=int(opts.get("num_ctx", 2048)),
                        num_predict=int(opts.get("num_predict", 256)),
                        seed=int(opts.get("seed", 42)),
                        requests_per_minute=self.rate_limiter.rpm,
                        rate_limit_wait_time=wait_time,
                        llm_output=output_obj,
                        raw_response=raw_response,
                        json_valid=json_ok,
                        schema_valid=schema_ok,
                        evidence_valid=ev_valid,
                        cache_hit=False,
                        prompt_tokens=metrics.get("prompt_tokens", 0),
                        output_tokens=metrics.get("output_tokens", 0),
                        processing_time=proc_t,
                        status="completed" if schema_ok else "failed"
                    )

                except Exception as e:
                    proc_t = time.time() - start_t
                    res = LLMParagraphResult(
                        run_id=self.run_id,
                        article_id=rec.article_id,
                        paragraph_id=rec.paragraph_id,
                        text=rec.text,
                        cosine_similarity=rec.semantic_score or 0.0,
                        llm_model=model_name,
                        processing_time=proc_t,
                        status="failed",
                        error_message=str(e)
                    )

                # Persiste incrementalmente em JSONL (Checkpoint)
                jsonl_file.write(res.model_dump_json() + "\n")
                jsonl_file.flush()

                self.disk_cache[cache_key] = res
                self.completed_ids.add(rec.paragraph_id)
                results.append(res)

                if progress_callback:
                    progress_callback(idx, total_items, res, {"waiting": False, "wait_time": wait_time})

        self._save_summary_statistics(results)
        return results

    def _save_summary_statistics(self, results: List[LLMParagraphResult]) -> Dict[str, Any]:
        """Calcula e persiste as estatísticas detalhadas de execução e qualidade das evidências."""
        n_total = len(results)
        if n_total == 0:
            return {}

        n_json_valid = sum(1 for r in results if r.json_valid)
        n_schema_valid = sum(1 for r in results if r.schema_valid)
        
        total_evidences_evaluated = sum(len(r.llm_output.evidence) for r in results)
        valid_evidences_count = 0
        for r in results:
            if r.llm_output.evidence:
                is_valid, details = validate_textual_evidence(r.text, r.llm_output.evidence)
                valid_evidences_count += sum(1 for d in details if d["valid"])

        n_paras_with_valid_evidence = sum(1 for r in results if r.evidence_valid and r.llm_output.evidence)

        evr_pct = (valid_evidences_count / total_evidences_evaluated * 100.0) if total_evidences_evaluated > 0 else 0.0
        pvr_pct = (n_paras_with_valid_evidence / n_total * 100.0) if n_total > 0 else 0.0

        stats = {
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "llm_model": results[0].llm_model if results else DEFAULT_LLM_MODEL,
            "processed_paragraphs": n_total,
            "json_valid_count": n_json_valid,
            "json_valid_pct": round(n_json_valid / n_total * 100.0, 2),
            "schema_valid_count": n_schema_valid,
            "schema_valid_pct": round(n_schema_valid / n_total * 100.0, 2),
            "total_evidences_evaluated": total_evidences_evaluated,
            "valid_evidences_count": valid_evidences_count,
            "evidence_validation_rate_evr": round(evr_pct, 2),
            "paragraphs_with_valid_evidence": n_paras_with_valid_evidence,
            "paragraph_validation_rate_pvr": round(pvr_pct, 2),
            "total_processing_time_sec": round(sum(r.processing_time for r in results), 2),
            "total_wait_time_sec": round(self.rate_limiter.total_wait_time, 2),
            "rpm_configured": self.rate_limiter.rpm
        }

        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        return stats

    def get_refined_corpus(self, require_valid_evidence: bool = True) -> List[LLMParagraphResult]:
        """
        Retorna exclusivamente os resultados válidos e estruturados que formam o Corpus Refinado.
        """
        all_results = list(self.disk_cache.values())
        refined = []
        for r in all_results:
            if not r.schema_valid:
                continue
            has_content = bool(
                r.llm_output.concepts or
                r.llm_output.definitions or
                r.llm_output.determinants or
                r.llm_output.dimensions or
                r.llm_output.causal_relations or
                r.llm_output.properties
            )
            if not has_content:
                continue
            if require_valid_evidence and not r.evidence_valid:
                continue
            refined.append(r)
        return refined
