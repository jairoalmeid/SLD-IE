"""
Construtor e gerenciador do Índice de Recuperação do Corpus Refinado com FAISS e Parquet.
"""

import os
import json
import time
import zipfile
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable, Any
from datetime import datetime

import faiss
import pyarrow as pa
import pyarrow.parquet as pq

from src.sld.models.concept_label import (
    ConceptLabel,
    CONCEPT_LABEL_NAMES,
    CONCEPT_LABEL_SHORT_NAMES,
    MULTILABEL_CLASSES,
)
from src.sld.models.classification import ParagraphRecord
from src.sld.rag_index.models import (
    RAGIndexConfig,
    RAGIndexManifest,
    IndexStats,
    CorpusDistributionStats,
    CoverageStats,
)


def compute_sha256(file_path: Path) -> str:
    """Calcula o hash SHA-256 de um arquivo em disco."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_corpus_distribution_stats(
    corpus_records: List[ParagraphRecord]
) -> CorpusDistributionStats:
    """
    Calcula as estatísticas de distribuição por classe do corpus classificado.
    """
    stats = CorpusDistributionStats()
    stats.total_classified = len(corpus_records)

    c0 = 0
    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    c5 = 0
    unique_rel_ids = set()

    for r in corpus_records:
        preds = set(r.predicted_labels or [])
        is_relevant = (r.status == "MODEL_RELEVANT") or bool(preds)

        if not is_relevant or not preds:
            c0 += 1
        else:
            unique_rel_ids.add(r.paragraph_id)
            if "definition" in preds:
                c1 += 1
            if "determinant" in preds:
                c2 += 1
            if "type_dimension" in preds:
                c3 += 1
            if "causal_relation" in preds:
                c4 += 1
            if "property" in preds:
                c5 += 1

    stats.class_0_not_relevant = c0
    stats.class_1_definition = c1
    stats.class_2_determinant = c2
    stats.class_3_type_dimension = c3
    stats.class_4_causal_relation = c4
    stats.class_5_property = c5
    stats.total_unique_relevant = len(unique_rel_ids)
    stats.total_multilabel_occurrences = c1 + c2 + c3 + c4 + c5

    if stats.total_classified > 0:
        stats.pct_relevant = (stats.total_unique_relevant / stats.total_classified) * 100.0
        stats.pct_class_0 = (stats.class_0_not_relevant / stats.total_classified) * 100.0

    return stats


def compute_coverage_stats(
    total_original_articles: int,
    refined_records: List[ParagraphRecord]
) -> CoverageStats:
    """
    Calcula as métricas de cobertura documental e densidade de parágrafos por artigo.
    """
    cov = CoverageStats()
    cov.total_original_articles = total_original_articles

    doc_counts: Dict[str, int] = {}
    for r in refined_records:
        doc_counts[r.article_id] = doc_counts.get(r.article_id, 0) + 1

    cov.refined_corpus_articles = len(doc_counts)
    cov.indexed_articles = len(doc_counts)

    if total_original_articles > 0:
        cov.pct_articles_represented = (cov.indexed_articles / total_original_articles) * 100.0

    if doc_counts:
        counts = list(doc_counts.values())
        cov.mean_paragraphs_per_article = float(np.mean(counts))
        cov.median_paragraphs_per_article = float(np.median(counts))
        cov.min_paragraphs_per_article = int(np.min(counts))
        cov.max_paragraphs_per_article = int(np.max(counts))

    return cov


class RAGIndexBuilder:
    """
    Construtor e validador de índices FAISS sobre o corpus refinado pela regressão logística.
    """

    def __init__(self, output_dir: Path, config: Optional[RAGIndexConfig] = None):
        self.output_dir = Path(output_dir)
        self.config = config or RAGIndexConfig()

    def filter_and_deduplicate(
        self, corpus_records: List[ParagraphRecord]
    ) -> List[ParagraphRecord]:
        """
        Filtra parágrafos excluindo a Classe 0 (Não Relevante) e mantendo parágrafos
        com pelo menos uma classe 1 a 5 ativa, removendo duplicações lógicas de paragraph_id.
        """
        seen_ids = set()
        refined: List[ParagraphRecord] = []

        for r in corpus_records:
            # Exclui explicitamente Classe 0 / MODEL_NOT_RELEVANT
            if r.status == "MODEL_NOT_RELEVANT":
                continue

            preds = set(r.predicted_labels or [])
            # Deve pertencer a pelo menos uma das classes conceituais 1 a 5
            if not preds:
                continue

            if r.paragraph_id in seen_ids:
                continue

            seen_ids.add(r.paragraph_id)
            refined.append(r)

        return refined

    def build(
        self,
        corpus_records: List[ParagraphRecord],
        embeddings_matrix: np.ndarray,
        all_corpus_records: Optional[List[ParagraphRecord]] = None,
        total_original_articles: int = 0,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        index_version: str = "v001",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[Path, Path, Path, Path, IndexStats, RAGIndexManifest]:
        """
        Executa a construção completa do índice FAISS, metadados Parquet, manifesto e pacote ZIP.
        """
        start_time = time.time()

        # 1. Filtragem e deduplicação lógica
        refined_records = self.filter_and_deduplicate(corpus_records)
        num_records = len(refined_records)

        if num_records == 0:
            raise ValueError(
                "Nenhum parágrafo relevante (Classes 1–5) encontrado no corpus fornecido para indexação."
            )

        # 2. Mapeamento e extração de embeddings sem recálculo
        # Mapeia paragraph_id -> primeiro índice na matriz original de embeddings
        base_records = all_corpus_records or corpus_records
        id_to_matrix_idx = {}
        for idx, r in enumerate(base_records):
            if r.paragraph_id not in id_to_matrix_idx:
                id_to_matrix_idx[r.paragraph_id] = idx

        # Coleta índices na matriz
        selected_matrix_indices = []
        valid_refined_records = []
        for r in refined_records:
            if r.paragraph_id in id_to_matrix_idx:
                selected_matrix_indices.append(id_to_matrix_idx[r.paragraph_id])
                valid_refined_records.append(r)

        refined_records = valid_refined_records
        num_records = len(refined_records)

        if num_records == 0:
            raise ValueError("Não foi possível correlacionar os parágrafos refinados com a matriz de embeddings.")

        # Validação da dimensão e tipo dos embeddings
        raw_dim = embeddings_matrix.shape[1]
        self.config.dimension = raw_dim

        # 3. Inicialização do Índice FAISS
        # Métricas: METRIC_INNER_PRODUCT equivale a Cosine Similarity quando normalizado L2
        metric = faiss.METRIC_INNER_PRODUCT if self.config.metric == "inner_product" else faiss.METRIC_L2

        if self.config.index_type == "HNSW":
            faiss_index = faiss.IndexHNSWFlat(self.config.dimension, self.config.M, metric)
            faiss_index.hnsw.efConstruction = self.config.efConstruction
            faiss_index.hnsw.efSearch = self.config.efSearch
        else:
            faiss_index = faiss.IndexFlatIP(self.config.dimension) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.config.dimension)

        # 4. Inserção dos vetores em batches com controle de memória RAM
        batch_sz = self.config.index_batch_size
        num_batches = (num_records + batch_sz - 1) // batch_sz

        for b_idx in range(num_batches):
            b_start = b_idx * batch_sz
            b_end = min(b_start + batch_sz, num_records)
            b_indices = selected_matrix_indices[b_start:b_end]

            # Extração de fatia em float32
            b_vecs = embeddings_matrix[b_indices].astype(np.float32, copy=False)

            # Normalização L2 para garantir Cosine Similarity via Produto Interno
            if self.config.normalize_embeddings:
                norms = np.linalg.norm(b_vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                b_vecs = b_vecs / norms

            faiss_index.add(b_vecs)

            if progress_callback:
                progress_callback(b_end, num_records, f"Indexando lote {b_idx + 1}/{num_batches} ({b_end:,} de {num_records:,} vetores)")

        # 5. Construção dos Metadados Parquet
        meta_rows = []
        class_counts = {"definition": 0, "determinant": 0, "type_dimension": 0, "causal_relation": 0, "property": 0}

        for faiss_id, r in enumerate(refined_records):
            preds = set(r.predicted_labels or [])
            c1 = "definition" in preds
            c2 = "determinant" in preds
            c3 = "type_dimension" in preds
            c4 = "causal_relation" in preds
            c5 = "property" in preds

            if c1:
                class_counts["definition"] += 1
            if c2:
                class_counts["determinant"] += 1
            if c3:
                class_counts["type_dimension"] += 1
            if c4:
                class_counts["causal_relation"] += 1
            if c5:
                class_counts["property"] += 1

            meta_rows.append({
                "faiss_id": int(faiss_id),
                "paragraph_id": str(r.paragraph_id),
                "document_id": str(r.article_id),
                "article_id": str(r.article_id),
                "section": str(getattr(r, "section", "") or ""),
                "text": str(r.text),
                "embedding_row": int(selected_matrix_indices[faiss_id]),
                "class_1": bool(c1),
                "class_2": bool(c2),
                "class_3": bool(c3),
                "class_4": bool(c4),
                "class_5": bool(c5),
                "semantic_similarity": float(r.semantic_score or 0.0),
                "title": str(getattr(r, "title", "") or ""),
                "authors": str(getattr(r, "authors", "") or ""),
                "year": str(getattr(r, "year", "") or ""),
                "doi": str(getattr(r, "doi", "") or ""),
                "source_file": str(getattr(r, "source_pdf", "") or ""),
                "markdown_file": str(getattr(r, "markdown_path", "") or ""),
            })

        df_metadata = pd.DataFrame(meta_rows)

        # 6. Criação dos diretórios de saída
        target_dir = self.output_dir if self.output_dir.name == "rag_index" else (self.output_dir / "rag_index")
        target_dir.mkdir(parents=True, exist_ok=True)

        faiss_path = target_dir / "corpus_refinado.faiss"
        parquet_path = target_dir / "metadata.parquet"
        manifest_path = target_dir / "manifest.json"
        checksums_path = target_dir / "checksums.sha256"
        report_path = target_dir / "index_report.md"
        readme_path = target_dir / "README.md"

        # 7. Persistência dos arquivos
        faiss.write_index(faiss_index, str(faiss_path))
        df_metadata.to_parquet(parquet_path, engine="pyarrow", index=False)

        # 8. Validação rigorosa de integridade
        test_index = faiss.read_index(str(faiss_path))
        if test_index.ntotal != num_records:
            raise RuntimeError(
                f"Falha de integridade: faiss.ntotal ({test_index.ntotal}) difere do número de parágrafos ({num_records})."
            )

        test_table = pq.read_table(parquet_path)
        if test_table.num_rows != num_records:
            raise RuntimeError(
                f"Falha de integridade: metadata.parquet ({test_table.num_rows}) difere do número de vetores ({num_records})."
            )

        # Executa pequeno teste de consulta operacional
        dummy_query = np.ones((1, self.config.dimension), dtype=np.float32)
        dummy_query /= np.linalg.norm(dummy_query)
        _, test_ids = test_index.search(dummy_query, min(5, num_records))
        if test_ids.shape[1] == 0 or test_ids[0][0] < 0:
            raise RuntimeError("Falha no teste operacional de consulta ao índice recarregado.")

        dur = time.time() - start_time

        # 9. Cálculo de Cobertura e Estatísticas
        cov_stats = compute_coverage_stats(
            total_original_articles=total_original_articles or len(set(r.article_id for r in base_records)),
            refined_records=refined_records,
        )

        stats = IndexStats(
            total_vectors=num_records,
            unique_paragraphs=num_records,
            represented_documents=cov_stats.indexed_articles,
            embedding_dimension=self.config.dimension,
            index_type=f"Index{self.config.index_type}Flat",
            metric=self.config.metric,
            faiss_file_size_bytes=faiss_path.stat().st_size,
            parquet_file_size_bytes=parquet_path.stat().st_size,
            build_duration_sec=dur,
            class_counts=class_counts,
            coverage=cov_stats,
        )

        # 10. Hashes de integridade SHA-256
        hashes = {
            "corpus_refinado.faiss": compute_sha256(faiss_path),
            "metadata.parquet": compute_sha256(parquet_path),
        }

        # 11. Manifesto de Reprodutibilidade
        manifest = RAGIndexManifest(
            index_name="corpus_refinado",
            index_version=index_version,
            faiss_version=faiss.__version__,
            pyarrow_version=pa.__version__,
            index_type=self.config.index_type,
            metric=self.config.metric,
            embedding_model=embedding_model_name,
            embedding_dimension=self.config.dimension,
            embedding_dtype="float32",
            normalized_embeddings=self.config.normalize_embeddings,
            indexed_paragraphs=num_records,
            indexed_documents=cov_stats.indexed_articles,
            class_counts=class_counts,
            parameters={
                "M": self.config.M if self.config.index_type == "HNSW" else None,
                "efConstruction": self.config.efConstruction if self.config.index_type == "HNSW" else None,
                "efSearch": self.config.efSearch if self.config.index_type == "HNSW" else None,
                "batch_size": self.config.index_batch_size,
            },
            source_corpus="SLD Refined Corpus",
            classifier_type="LogisticRegression Multilabel OneVsRest",
            checksums=hashes,
        )

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2, ensure_ascii=False)

        hashes["manifest.json"] = compute_sha256(manifest_path)

        # Salva arquivo checksums.sha256
        with open(checksums_path, "w", encoding="utf-8") as f:
            for fname, fhash in hashes.items():
                f.write(f"{fhash}  {fname}\n")

        # 12. Relatório Metodológico index_report.md
        report_md = f"""# Relatório Metodológico do Índice de Recuperação do Corpus Refinado

**Versão do Índice:** `{manifest.index_version}`  
**Data de Criação:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`  
**Framework de Indexação:** `FAISS v{faiss.__version__}` (PyArrow v{pa.__version__})

---

## 1. Parâmetros e Dimensões Vetoriais

- **Modelo de Embedding:** `{manifest.embedding_model}`
- **Dimensão dos Embeddings:** `{manifest.embedding_dimension}`
- **Tipo de Índice:** `Index{manifest.index_type}Flat`
- **Métrica:** `{manifest.metric}` (Cosine Similarity via Produto Interno com Normalização L2)
- **Parâmetros HNSW:** `M={self.config.M}`, `efConstruction={self.config.efConstruction}`, `efSearch={self.config.efSearch}`
- **Tempo de Construção:** `{dur:.2f} segundos`

---

## 2. Estatísticas do Corpus Indexado

- **Total de Vetores Indexados:** `{num_records:,}`
- **Parágrafos Únicos:** `{num_records:,}`
- **Artigos Científicos Representados:** `{cov_stats.indexed_articles:,}` de `{cov_stats.total_original_articles:,}` ({cov_stats.pct_articles_represented:.2f}% de cobertura)
- **Média de Parágrafos por Artigo:** `{cov_stats.mean_paragraphs_per_article:.2f}` (Mediana: `{cov_stats.median_paragraphs_per_article:.1f}`, Min: `{cov_stats.min_paragraphs_per_article}`, Max: `{cov_stats.max_paragraphs_per_article}`)

---

## 3. Distribuição por Classes Conceituais

| Classe Conceitual | Ocorrências | % dos Parágrafos Indexados |
| :--- | :---: | :---: |
| **Classe 1 — Definição / Conceituação** | {class_counts['definition']:,} | {(class_counts['definition']/num_records*100):.2f}% |
| **Classe 2 — Fator Determinante** | {class_counts['determinant']:,} | {(class_counts['determinant']/num_records*100):.2f}% |
| **Classe 3 — Tipo ou Dimensão** | {class_counts['type_dimension']:,} | {(class_counts['type_dimension']/num_records*100):.2f}% |
| **Classe 4 — Relação Causal** | {class_counts['causal_relation']:,} | {(class_counts['causal_relation']/num_records*100):.2f}% |
| **Classe 5 — Característica ou Propriedade** | {class_counts['property']:,} | {(class_counts['property']/num_records*100):.2f}% |

*(Nota: como a classificação é multilabel, a soma das ocorrências excede 100% pois um parágrafo pode conter múltiplas categorias).*

---

## 4. Hashes de Integridade (SHA-256)

```text
{hashes.get('corpus_refinado.faiss', '')}  corpus_refinado.faiss
{hashes.get('metadata.parquet', '')}  metadata.parquet
{hashes.get('manifest.json', '')}  manifest.json
```
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        # 13. README.md explicativo
        readme_md = f"""# Pacote de Recuperação Semântica do Corpus Refinado (SLD RAG)

Este pacote contém o índice vetorial FAISS e a tabela estruturada de metadados produzidos após a filtragem supervisionada por regressão logística multilabel.

## Estrutura do Pacote

- `corpus_refinado.faiss`: índice vetorial FAISS ({self.config.index_type}, dim={self.config.dimension}).
- `metadata.parquet`: tabela colunar com textos, IDs, tags conceituais e rastreabilidade documental.
- `manifest.json`: manifesto com configurações e metadados de reprodutibilidade.
- `checksums.sha256`: assinaturas criptográficas para validação de integridade.
- `index_report.md`: relatório metodológico para citação em teses e artigos.

## Utilização em RAG

Para carregar este índice no sistema de RAG:
```python
import faiss
import pandas as pd

index = faiss.read_index("corpus_refinado.faiss")
metadata = pd.read_parquet("metadata.parquet")

# Busca Top-k:
# distances, indices = index.search(query_vector, k=5)
# results = metadata.iloc[indices[0]]
```
"""
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_md)

        # 14. Empacotamento em arquivo ZIP
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"rag_index_{index_version}_{timestamp_str}.zip"
        zip_path = target_dir / zip_name

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(faiss_path, arcname=f"rag_index_{index_version}/corpus_refinado.faiss")
            zf.write(parquet_path, arcname=f"rag_index_{index_version}/metadata.parquet")
            zf.write(manifest_path, arcname=f"rag_index_{index_version}/manifest.json")
            zf.write(checksums_path, arcname=f"rag_index_{index_version}/checksums.sha256")
            zf.write(report_path, arcname=f"rag_index_{index_version}/index_report.md")
            zf.write(readme_path, arcname=f"rag_index_{index_version}/README.md")

        stats.zip_file_size_bytes = zip_path.stat().st_size

        return (faiss_path, parquet_path, manifest_path, zip_path, stats, manifest)
