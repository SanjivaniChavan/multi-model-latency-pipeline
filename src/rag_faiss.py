import json
import os
from typing import List, Dict, Any

import faiss
import numpy as np

from .config import settings
from .embeddings import embed_texts
from .utils import log


class FaissRAGRetriever:
    """
    Simple FAISS-based retriever over a text corpus.
    """

    def __init__(self):
        self.index = None
        self.metadata: List[Dict[str, Any]] = []

    def build_from_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        if metadatas is None:
            metadatas = [{} for _ in texts]

        log(f"Building FAISS index over {len(texts)} texts.")
        embs = embed_texts(texts)
        dim = embs.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs)
        self.metadata = metadatas

        os.makedirs(os.path.dirname(settings.faiss_index_path), exist_ok=True)
        faiss.write_index(self.index, settings.faiss_index_path)
        with open(settings.faiss_metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        log("FAISS index and metadata saved.")

    def load(self):
        if not os.path.exists(settings.faiss_index_path):
            log("[WARN] No FAISS index found. RAG retrieval will return empty results.")
            return

        log(f"Loading FAISS index from {settings.faiss_index_path}")
        self.index = faiss.read_index(settings.faiss_index_path)

        if os.path.exists(settings.faiss_metadata_path):
            with open(settings.faiss_metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        if self.index is None:
            self.load()
        if self.index is None:
            return []

        if top_k is None:
            top_k = settings.top_k

        q_emb = embed_texts([query])
        scores, idxs = self.index.search(q_emb, top_k)
        idxs = idxs[0]
        scores = scores[0]

        results = []
        for i, s in zip(idxs, scores):
            if i < 0 or i >= len(self.metadata):
                continue
            item = self.metadata[i].copy()
            item["score"] = float(s)
            results.append(item)

        return results
