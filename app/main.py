from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import os

from app.models.db import create_tables
from app.services.rag import HybridRetriever
from app.api.routes import health, websocket, telephony, admin
from app.api.routes.websocket import sessions


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    app.state.retriever = HybridRetriever()
    count = app.state.retriever.rebuild_from_chroma()
    print(f"[startup] RAG index loaded: {count} chunks")
    if count == 0:
        print("[startup] WARNING: ChromaDB is empty. Run: python scripts/seed_rag.py")
    yield
    sessions.clear()


def create_app() -> FastAPI:
    app = FastAPI(title="AI Front Desk", version="1.0.0", lifespan=lifespan)

    _static_dir = os.path.join(os.path.dirname(__file__), "static")
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

    app.include_router(health.router)
    app.include_router(websocket.router)
    app.include_router(telephony.router)
    app.include_router(admin.router)

    return app


app = create_app()
