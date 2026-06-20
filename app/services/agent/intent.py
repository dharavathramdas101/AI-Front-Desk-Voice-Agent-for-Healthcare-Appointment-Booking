from __future__ import annotations

import json
import time

from groq import Groq

from app.config import GROQ_API_KEY, LLM_MODEL
from app.services.agent.prompts import INTENT_SYSTEM

_client = Groq(api_key=GROQ_API_KEY)


def classify_intent(utterance: str, history: list[dict]) -> tuple[str, float]:
    """Returns (intent_label, elapsed_ms). Label ∈ {BOOK, RESCHEDULE, CANCEL, FAQ, SMALL_TALK, ESCALATE, UNCLEAR}."""
    t0 = time.perf_counter()
    messages = [{"role": "system", "content": INTENT_SYSTEM}]
    messages.extend(history[-4:])
    messages.append({"role": "user", "content": utterance})

    resp = _client.chat.completions.create(
        model=LLM_MODEL, messages=messages, max_tokens=64, temperature=0.0,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw).get("intent", "UNCLEAR"), elapsed
    except json.JSONDecodeError:
        for label in ("BOOK", "RESCHEDULE", "CANCEL", "FAQ", "SMALL_TALK", "ESCALATE"):
            if label in raw.upper():
                return label, elapsed
        return "UNCLEAR", elapsed
