from __future__ import annotations

import time

from groq import Groq

from app.config import GROQ_API_KEY, LLM_MODEL, TOP_K_RETRIEVAL
from app.services.agent.prompts import FAQ_SYSTEM, HARD_RULES
from app.services.rag import Chunk, HybridRetriever

_client = Groq(api_key=GROQ_API_KEY)


def run_faq_agent(
    utterance: str,
    history: list[dict],
    retriever: HybridRetriever,
) -> tuple[str, list[Chunk], float, float]:
    """Retrieve + LLM compose answer. Returns (response, chunks, retrieval_ms, llm_ms)."""
    t0 = time.perf_counter()
    chunks = retriever.retrieve(utterance, top_k=TOP_K_RETRIEVAL)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    if not chunks:
        return (
            "I don't have that information on hand. Let me connect you with a team member who can help.",
            [], retrieval_ms, 0.0,
        )

    context = "\n\n---\n\n".join(c.text for c in chunks)
    messages = [
        {"role": "system", "content": FAQ_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {utterance}"},
    ]

    t1 = time.perf_counter()
    resp = _client.chat.completions.create(model=LLM_MODEL, messages=messages, max_tokens=128, temperature=0.1)
    llm_ms = (time.perf_counter() - t1) * 1000
    return resp.choices[0].message.content.strip(), chunks, retrieval_ms, llm_ms


def run_small_talk(utterance: str, history: list[dict]) -> tuple[str, float]:
    """Friendly short reply for greetings/farewells."""
    t0 = time.perf_counter()
    messages = [
        {"role": "system", "content": (
            "You are a warm hospital front desk assistant. "
            "Reply naturally to greetings, thanks, and goodbyes in 1–2 sentences. "
            "NEVER mention or invent services, portals, websites, or phone numbers not in this conversation. "
            "Stick to what you can actually do: book, reschedule, cancel appointments, and answer general hospital FAQs. "
            + HARD_RULES
        )},
    ]
    messages.extend(history[-4:])
    messages.append({"role": "user", "content": utterance})
    resp = _client.chat.completions.create(model=LLM_MODEL, messages=messages, max_tokens=64, temperature=0.7)
    return resp.choices[0].message.content.strip(), (time.perf_counter() - t0) * 1000
