from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_PATH, EMBED_MODEL, TOP_K_RETRIEVAL, RRF_K


@dataclass
class Chunk:
    text: str
    source: str
    chunk_idx: int


class HybridRetriever:
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._embedder = SentenceTransformer(EMBED_MODEL, device=device)
        self._chroma = chromadb.PersistentClient(path=CHROMA_PATH)
        self._collection = self._chroma.get_or_create_collection(
            name="hospital_kb",
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[Chunk] = []

    def index_documents(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = self._embedder.encode(texts, show_progress_bar=True, batch_size=64).tolist()
        self._collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"source": c.source, "chunk_idx": c.chunk_idx} for c in chunks],
            ids=[f"{c.source}::{c.chunk_idx}" for c in chunks],
        )
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        existing = self._collection.get(include=["documents", "metadatas"])
        if not existing["documents"]:
            return
        self._bm25_chunks = [
            Chunk(text=t, source=m["source"], chunk_idx=m["chunk_idx"])
            for t, m in zip(existing["documents"], existing["metadatas"])
        ]
        self._bm25 = BM25Okapi([c.text.lower().split() for c in self._bm25_chunks])

    def rebuild_from_chroma(self) -> int:
        self._rebuild_bm25()
        return len(self._bm25_chunks)

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list[Chunk]:
        if self._bm25 is None or not self._bm25_chunks:
            return []

        fetch_n = min(len(self._bm25_chunks), max(top_k * 3, 15))

        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:fetch_n]

        query_emb = self._embedder.encode([query], show_progress_bar=False).tolist()
        dense_results = self._collection.query(
            query_embeddings=query_emb,
            n_results=min(fetch_n, self._collection.count() or 1),
            include=["documents", "metadatas"],
        )
        dense_chunks: list[Chunk] = []
        if dense_results["documents"] and dense_results["documents"][0]:
            for text, meta in zip(dense_results["documents"][0], dense_results["metadatas"][0]):
                dense_chunks.append(Chunk(text=text, source=meta["source"], chunk_idx=meta["chunk_idx"]))

        rrf_scores: dict[str, float] = defaultdict(float)

        def key(c: Chunk) -> str:
            return f"{c.source}::{c.chunk_idx}"

        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[key(self._bm25_chunks[idx])] += 1.0 / (RRF_K + rank + 1)
        for rank, chunk in enumerate(dense_chunks):
            rrf_scores[key(chunk)] += 1.0 / (RRF_K + rank + 1)

        lookup = {key(c): c for c in self._bm25_chunks}
        for c in dense_chunks:
            if key(c) not in lookup:
                lookup[key(c)] = c

        ranked = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
        return [lookup[k] for k in ranked[:top_k] if k in lookup]

    def is_empty(self) -> bool:
        return self._collection.count() == 0
