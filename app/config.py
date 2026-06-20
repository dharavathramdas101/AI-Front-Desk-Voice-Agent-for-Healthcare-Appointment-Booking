from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).parent.parent
load_dotenv(_root / ".env")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHROMA_PATH: str = os.getenv("CHROMA_PATH", str(_root / "chroma_db"))
DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{_root / 'ai_front_desk.db'}")
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
CANCELLATION_WINDOW_HOURS: int = int(os.getenv("CANCELLATION_WINDOW_HOURS", "24"))

TOP_K_RETRIEVAL: int = 3
RRF_K: int = 60

KB_PATH: Path = Path(__file__).parent / "knowledge_base"
