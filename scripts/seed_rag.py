"""
Chunk and embed all markdown files in app/knowledge_base/ into ChromaDB.
Run once after seed_db.py: python scripts/seed_rag.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from pathlib import Path
from rag import Chunk, HybridRetriever
import config


def chunk_markdown(text: str, source: str, max_tokens: int = 150) -> list[Chunk]:
    """Split on double newlines (paragraph breaks), targeting ~150 tokens per chunk."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buffer = []
    buffer_len = 0
    chunk_idx = 0

    for para in paragraphs:
        words = para.split()
        if buffer_len + len(words) > max_tokens and buffer:
            chunks.append(Chunk(text=" ".join(buffer), source=source, chunk_idx=chunk_idx))
            chunk_idx += 1
            buffer = []
            buffer_len = 0
        buffer.extend(words)
        buffer_len += len(words)

    if buffer:
        chunks.append(Chunk(text=" ".join(buffer), source=source, chunk_idx=chunk_idx))

    return chunks


def seed() -> None:
    kb_path = config.KB_PATH
    retriever = HybridRetriever()

    if not retriever.is_empty():
        print("ChromaDB already populated. Delete chroma_db/ to re-seed.")
        return

    all_chunks: list[Chunk] = []
    for md_file in sorted(kb_path.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        chunks = chunk_markdown(text, source=md_file.name)
        all_chunks.extend(chunks)
        print(f"  {md_file.name}: {len(chunks)} chunks")

    print(f"\nIndexing {len(all_chunks)} total chunks...")
    retriever.index_documents(all_chunks)
    print("Done. ChromaDB populated.")


if __name__ == "__main__":
    seed()
