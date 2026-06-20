from __future__ import annotations

import json
import time
from datetime import datetime

from groq import Groq

from app.config import GROQ_API_KEY, LLM_MODEL
from app.services.agent.prompts import BOOKING_SYSTEM
from app.services.agent.tools import TOOLS, TOOL_MAP

_client = Groq(api_key=GROQ_API_KEY)


def run_booking_agent(utterance: str, history: list[dict]) -> tuple[str, float]:
    """Multi-turn tool-calling agent for BOOK / RESCHEDULE / CANCEL. Returns (response, elapsed_ms)."""
    t0 = time.perf_counter()
    today = datetime.now().strftime("%A, %d %B %Y")
    system = BOOKING_SYSTEM + f"\n\nTODAY'S DATE: {today}. Use this to resolve relative dates like 'tomorrow', 'next Monday', 'this Friday'."

    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": utterance})

    for _ in range(5):
        resp = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=512,
            temperature=0.2,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content or "I'm sorry, I couldn't complete that action.", (time.perf_counter() - t0) * 1000

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            fn = TOOL_MAP.get(tc.function.name)
            result = fn(json.loads(tc.function.arguments)) if fn else {"error": f"Unknown tool: {tc.function.name}"}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

    return "I wasn't able to complete that — please try again or speak with a staff member.", (time.perf_counter() - t0) * 1000
