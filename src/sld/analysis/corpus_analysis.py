"""
Módulo de Análise Exploratória e Bibliométrica do Corpus (Estatísticas, Frequência de Termos, Co-ocorrência C_ij e Grafo).
"""

import re
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import networkx as nx
from wordcloud import WordCloud
from src.sld.models.classification import ParagraphRecord


STOPWORDS_PT_EN = set([
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com", "não", "uma", "os", "no", "se", "na",
    "por", "mais", "as", "dos", "como", "mas", "foi", "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser",
    "quando", "muito", "nos", "já", "está", "eu", "também", "só", "pelo", "pela", "até", "isso", "ela", "entre",
    "era", "depois", "sem", "mesmo", "aos", "ter", "seus", "quem", "nas", "me", "esse", "eles", "estão", "você",
    "tinha", "foram", "essa", "num", "nem", "suas", "meu", "às", "minha", "têm", "numa", "pelos", "elas", "havia",
    "seja", "qual", "será", "nós", "tenho", "lhe", "deles", "essas", "esses", "pelas", "este", "fosse", "dele",
    "tu", "te", "vocês", "vos", "lhes", "meus", "minhas", "teu", "tua", "teus", "tuas", "nosso", "nossa", "nossos",
    "nossas", "dela", "delas", "esta", "estes", "estas", "aquele", "aquela", "aqueles", "aquelas", "isto", "aquilo",
    "estou", "está", "estamos", "estão", "estive", "esteve", "estivemos", "estiveram", "estava", "estávamos", "estavam",
    "the", "and", "of", "to", "in", "a", "is", "that", "for", "it", "as", "was", "with", "on", "are", "by", "this",
    "be", "or", "from", "at", "an", "they", "which", "one", "you", "were", "her", "all", "she", "there", "would",
    "their", "we", "him", "been", "has", "had", "have", "more", "when", "who", "will", "more", "no", "if", "out",
    "so", "said", "what", "up", "its", "about", "into", "than", "them", "can", "only", "other", "new", "some", "time"
])


def tokenize_text(text: str, min_length: int = 3) -> List[str]:
    """Tokeniza o texto, converte para minúsculas e remove pontuação/stopwords."""
    words = re.findall(r"\b[a-zA-ZÀ-ÿ]{3,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS_PT_EN and len(w) >= min_length]


def compute_corpus_descriptors(records: List[ParagraphRecord]) -> Dict[str, Any]:
    """Calcula estatísticas bibliométricas gerais do corpus."""
    if not records:
        return {
            "n_documents": 0, "n_paragraphs": 0, "total_words": 0,
            "mean_words_per_doc": 0.0, "mean_paras_per_doc": 0.0,
            "median_words": 0.0, "std_words": 0.0, "min_words": 0, "max_words": 0
        }

    doc_ids = set(r.article_id for r in records)
    word_counts = [len(r.text.split()) for r in records]

    paras_per_doc = Counter(r.article_id for r in records)
    words_per_doc = {}
    for r in records:
        words_per_doc[r.article_id] = words_per_doc.get(r.article_id, 0) + len(r.text.split())

    doc_word_vals = list(words_per_doc.values())

    return {
        "n_documents": len(doc_ids),
        "n_paragraphs": len(records),
        "total_words": sum(word_counts),
        "mean_words_per_doc": float(np.mean(doc_word_vals)) if doc_word_vals else 0.0,
        "mean_paras_per_doc": float(np.mean(list(paras_per_doc.values()))) if paras_per_doc else 0.0,
        "median_words": float(np.median(word_counts)),
        "std_words": float(np.std(word_counts)),
        "min_words": int(np.min(word_counts)),
        "max_words": int(np.max(word_counts)),
    }


def compute_top_terms(records: List[ParagraphRecord], top_n: int = 50) -> pd.DataFrame:
    """Calcula o ranking dos termos mais frequentes no corpus."""
    counter = Counter()
    for r in records:
        counter.update(tokenize_text(r.text))

    most_common = counter.most_common(top_n)
    return pd.DataFrame(most_common, columns=["Termo", "Frequência"])


def generate_wordcloud_image(records: List[ParagraphRecord]):
    """Gera a imagem de WordCloud a partir do texto do corpus."""
    all_tokens = []
    for r in records:
        all_tokens.extend(tokenize_text(r.text))

    if not all_tokens:
        return None

    text_space = " ".join(all_tokens)
    wc = WordCloud(width=800, height=400, background_color="white", max_words=100).generate(text_space)
    return wc.to_image()


def compute_cooccurrence_matrix(
    records: List[ParagraphRecord],
    level: str = "document",
    top_n_terms: int = 30,
    min_cooccurrence: int = 2
) -> Tuple[pd.DataFrame, nx.Graph]:
    """
    Calcula a matriz de co-ocorrência C_ij = sum I(i in d)I(j in d) e cria o grafo NetworkX.
    level: 'document' ou 'paragraph'
    """
    if level == "document":
        grouped = {}
        for r in records:
            if r.article_id not in grouped:
                grouped[r.article_id] = []
            grouped[r.article_id].append(r.text)
        units = [" ".join(txts) for txts in grouped.values()]
    else:
        units = [r.text for r in records]

    # Seleciona os top N termos gerais
    overall_counter = Counter()
    unit_tokens = []
    for u in units:
        tokens = set(tokenize_text(u))
        unit_tokens.append(tokens)
        overall_counter.update(tokens)

    top_terms = [term for term, _ in overall_counter.most_common(top_n_terms)]
    top_set = set(top_terms)

    # Matriz C_ij
    matrix = pd.DataFrame(0, index=top_terms, columns=top_terms, dtype=int)

    for tokens in unit_tokens:
        relevant_tokens = list(tokens.intersection(top_set))
        for i in range(len(relevant_tokens)):
            for j in range(i + 1, len(relevant_tokens)):
                t1, t2 = relevant_tokens[i], relevant_tokens[j]
                matrix.loc[t1, t2] += 1
                matrix.loc[t2, t1] += 1

    # Constrói o Grafo NetworkX
    G = nx.Graph()
    for t1 in top_terms:
        G.add_node(t1, weight=overall_counter[t1])

    for i in range(len(top_terms)):
        for j in range(i + 1, len(top_terms)):
            t1, t2 = top_terms[i], top_terms[j]
            weight = matrix.loc[t1, t2]
            if weight >= min_cooccurrence:
                G.add_edge(t1, t2, weight=int(weight))

    return matrix, G
