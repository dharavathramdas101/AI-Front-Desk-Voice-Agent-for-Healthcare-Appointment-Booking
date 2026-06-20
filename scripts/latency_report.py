"""
Latency report: reads CallLog from SQLite and prints avg/p50/p95 per stage.
Run after a demo session: python scripts/latency_report.py

Output example:
  Stage            avg_ms   p50_ms   p95_ms
  ─────────────────────────────────────────
  STT (whisper)     832.1    810.0    980.4
  Retrieval (RAG)    23.5     21.0     38.2
  LLM (Groq)        421.3    400.0    590.1
  TTS (pyttsx3)     215.8    210.0    280.5
  Total round-trip 1492.7   1450.0   1750.3
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import pandas as pd
from sqlalchemy import text

from db import engine


def report() -> None:
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT * FROM call_logs ORDER BY timestamp"),
            conn,
        )

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
        series = df[col].dropna()
        if series.empty:
            print(f"{label:<22}  {'—':>8}  {'—':>8}  {'—':>8}")
        else:
            avg = series.mean()
            p50 = series.quantile(0.50)
            p95 = series.quantile(0.95)
            print(f"{label:<22}  {avg:>8.1f}  {p50:>8.1f}  {p95:>8.1f}")

    print()


if __name__ == "__main__":
    report()
