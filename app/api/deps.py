from __future__ import annotations

from fastapi import Request

from app.services.rag import HybridRetriever


def get_retriever(request: Request) -> HybridRetriever:
    return request.app.state.retriever
