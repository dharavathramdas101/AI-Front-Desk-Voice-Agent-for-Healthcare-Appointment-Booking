"""
Print avg/p50/p95 latency per stage from call logs.
Run after a demo session: python scripts/latency_report.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sqlalchemy import text
from app.models.db import engine


def report() -> None:
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM call_logs ORDER BY timestamp"), conn)

    if df.empty:
        print("No call logs found. Run a demo session first.")
        return

    print(f"\nTotal turns logged: {len(df)}")
    print(f"Sessions: {df['session_id'].nunique()}")
    print(f"Intents: {df['intent'].value_counts().to_dict()}\n")

    stages = [
        ("STT (whisper)",    "latency_stt_ms"),
        ("Retrieval (RAG)",  "latency_retrieval_ms"),
        ("LLM (Groq)",       "latency_llm_ms"),
        ("TTS (pyttsx3)",    "latency_tts_ms"),
        ("Total round-trip", "latency_total_ms"),
    ]

    header = f"{'Stage':<22}  {'avg_ms':>8}  {'p50_ms':>8}  {'p95_ms':>8}"
    print(header)
    print("─" * len(header))
    for label, col in stages:
        s = df[col].dropna()
        if s.empty:
            print(f"{label:<22}  {'—':>8}  {'—':>8}  {'—':>8}")
        else:
            print(f"{label:<22}  {s.mean():>8.1f}  {s.quantile(0.50):>8.1f}  {s.quantile(0.95):>8.1f}")
    print()


if __name__ == "__main__":
    report()
