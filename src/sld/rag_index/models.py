"""
Modelos de dados e schemas Pydantic para o Índice de Recuperação do Corpus Refinado.
"""

from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class RAGIndexConfig(BaseModel):
    """Configurações técnicas para a construção do índice vetorial FAISS."""
    index_type: Literal["HNSW", "FlatIP"] = Field(
        default="HNSW",
        description="Tipo de índice FAISS: 'HNSW' (IndexHNSWFlat - busca aproximada de alta velocidade) ou 'FlatIP' (IndexFlatIP - busca exata por produto interno)."
    )
    metric: Literal["inner_product", "l2"] = Field(
        default="inner_product",
        description="Métrica de similaridade. Produto Interno (inner_product) equivale a Cosine Similarity quando os embeddings são normalizados L2."
    )
    dimension: int = Field(default=384, description="Dimensão do espaço vetorial de embeddings.")
    M: int = Field(default=32, description="Número de conexões bidirecionais criadas para cada elemento no grafo HNSW (típico: 16 a 64).")
    efConstruction: int = Field(default=64, description="Tamanho da fila de exploração durante a construção do grafo HNSW.")
    efSearch: int = Field(default=64, description="Tamanho da fila de exploração durante as consultas no grafo HNSW.")
    index_batch_size: int = Field(default=8192, description="Tamanho do lote para inserção de vetores no índice FAISS.")
    normalize_embeddings: bool = Field(default=True, description="Normalizar vetores para norma L2 unitária antes da inserção no índice.")


class CorpusDistributionStats(BaseModel):
    """Estatísticas da distribuição conceitual do corpus classificado."""
    total_classified: int = 0
    class_0_not_relevant: int = 0
    class_1_definition: int = 0
    class_2_determinant: int = 0
    class_3_type_dimension: int = 0
    class_4_causal_relation: int = 0
    class_5_property: int = 0
    total_unique_relevant: int = 0
    total_multilabel_occurrences: int = 0
    pct_relevant: float = 0.0
    pct_class_0: float = 0.0


class CoverageStats(BaseModel):
    """Estatísticas de cobertura documental do corpus."""
    total_original_articles: int = 0
    refined_corpus_articles: int = 0
    indexed_articles: int = 0
    pct_articles_represented: float = 0.0
    mean_paragraphs_per_article: float = 0.0
    median_paragraphs_per_article: float = 0.0
    min_paragraphs_per_article: int = 0
    max_paragraphs_per_article: int = 0


class IndexStats(BaseModel):
    """Estatísticas consolidadas do índice criado."""
    total_vectors: int = 0
    unique_paragraphs: int = 0
    represented_documents: int = 0
    embedding_dimension: int = 0
    index_type: str = ""
    metric: str = ""
    faiss_file_size_bytes: int = 0
    parquet_file_size_bytes: int = 0
    zip_file_size_bytes: int = 0
    build_duration_sec: float = 0.0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    class_counts: Dict[str, int] = Field(default_factory=dict)
    coverage: Optional[CoverageStats] = None


class RAGIndexManifest(BaseModel):
    """Manifesto formal de reprodutibilidade metodológica do índice."""
    index_name: str = "corpus_refinado"
    index_version: str = "v001"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    faiss_version: str = ""
    pyarrow_version: str = ""
    index_type: str = "HNSW"
    metric: str = "inner_product"
    embedding_model: str = ""
    embedding_dimension: int = 384
    embedding_dtype: str = "float32"
    normalized_embeddings: bool = True
    indexed_paragraphs: int = 0
    indexed_documents: int = 0
    class_counts: Dict[str, int] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    source_corpus: str = "SLD Refined Corpus"
    classifier_type: str = "LogisticRegression OneVsRest"
    software_version: str = "1.0.0"
    checksums: Dict[str, str] = Field(default_factory=dict)


class RAGQueryResult(BaseModel):
    """Resultado individual de consulta vetorial Top-k no índice RAG."""
    rank: int
    faiss_id: int
    paragraph_id: str
    article_id: str
    score: float
    classes: List[str] = Field(default_factory=list)
    section: Optional[str] = None
    title: Optional[str] = None
    text: str = ""
