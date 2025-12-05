from functools import lru_cache
from typing import List

import numpy as np

from .config import settings
from .utils import log

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    log("[WARN] sentence-transformers not installed. Embeddings will be random.")


@lru_cache(maxsize=1)
def _load_model():
    if SentenceTransformer is None:
        return None
    log(f"Loading embedding model: {settings.embedding_model_name}")
    return SentenceTransformer(settings.embedding_model_name)


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Returns an array of shape (N, D).
    If sentence-transformers is not available, returns random vectors (for structure only).
    """
    if isinstance(texts, str):
        texts = [texts]

    model = _load_model()

    if model is None:
        log("[WARN] Using random embeddings as fallback.")
        return np.random.randn(len(texts), 384).astype("float32")

    embs = model.encode(texts, normalize_embeddings=True)
    return np.asarray(embs, dtype="float32")
