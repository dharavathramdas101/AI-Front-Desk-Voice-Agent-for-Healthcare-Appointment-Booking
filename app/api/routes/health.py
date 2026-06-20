from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

import os

router = APIRouter()

_static_dir = os.path.join(os.path.dirname(__file__), "..", "..", "static")


@router.get("/")
async def root():
    return FileResponse(os.path.join(_static_dir, "index.html"))


@router.get("/health")
async def health(request: Request):
    retriever = request.app.state.retriever
    count = retriever._collection.count() if retriever else 0
    return {"status": "ok", "rag_chunks": count}
