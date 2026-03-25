"""Lightweight embedding engine for tool matching.

Uses TF-IDF + cosine similarity by default (no external deps).
Optional: sentence-transformers for higher quality.
"""

import re
import math
import logging
from typing import Optional
from collections import Counter

logger = logging.getLogger("tier.embeddings")


class TFIDFEmbedder:
    """TF-IDF based embedder. Zero dependencies, surprisingly effective for tool matching."""

    def __init__(self):
        self._idf: dict[str, float] = {}
        self._vocab: list[str] = []
        self._doc_count = 0

    def fit(self, documents: list[str]):
        """Build IDF from a corpus of tool descriptions."""
        self._doc_count = len(documents)
        df: Counter = Counter()
        all_terms: set = set()

        for doc in documents:
            terms = set(self._tokenize(doc))
            for term in terms:
                df[term] += 1
            all_terms.update(terms)

        self._vocab = sorted(all_terms)
        self._idf = {
            term: math.log((self._doc_count + 1) / (count + 1)) + 1
            for term, count in df.items()
        }

    def embed(self, text: str) -> list[float]:
        """Compute TF-IDF vector for text."""
        terms = self._tokenize(text)
        tf = Counter(terms)
        total = len(terms) or 1

        vector = []
        for term in self._vocab:
            tfidf = (tf.get(term, 0) / total) * self._idf.get(term, 0)
            vector.append(tfidf)

        return vector

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        return re.findall(r'[a-z0-9]+', text.lower())


class SentenceEmbedder:
    """Optional high-quality embedder using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._available = True
            logger.info("SentenceTransformer loaded: %s", model_name)
        except ImportError:
            self._model = None
            self._available = False
            logger.debug("sentence-transformers not available, using TF-IDF")

    @property
    def available(self) -> bool:
        return self._available

    def embed(self, text: str) -> list[float]:
        if not self._model:
            return []
        return self._model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not self._model:
            return [[] for _ in texts]
        return [v.tolist() for v in self._model.encode(texts)]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
