from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.models.db import CallLog, get_session
from app.services.stt import transcribe
from app.services.tts import synthesize, chunk_wav
from app.services.agent.intent import classify_intent
from app.services.agent.booking import run_booking_agent
from app.services.agent.faq import run_faq_agent, run_small_talk

router = APIRouter()


# ── Session state ─────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    session_id: str
    history: list[dict] = field(default_factory=list)
    booking_context: dict = field(default_factory=dict)
    audio_buffer: bytearray = field(default_factory=bytearray)
    tts_cancelled: bool = False

    def add_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > 24:
            self.history = self.history[-24:]


sessions: dict[str, SessionState] = {}


# ── Turn processing ───────────────────────────────────────────────────────────

async def _route_and_respond(
    ws: WebSocket,
    session: SessionState,
    transcript: str,
    t_start: float,
    stt_ms: float,
) -> None:
    latency: dict[str, float] = {"stt_ms": stt_ms}
    loop = asyncio.get_event_loop()
    retriever = ws.app.state.retriever

    session.add_turn("user", transcript)

    intent, intent_ms = await loop.run_in_executor(None, classify_intent, transcript, session.history)
    latency["intent_ms"] = intent_ms
    await ws.send_json({"type": "intent", "intent": intent})

    retrieval_ms = 0.0
    llm_ms = 0.0

    if intent == "ESCALATE":
        response_text = "Of course! Let me connect you with one of our team members right away. Please hold for a moment."
        latency["retrieval_ms"] = 0.0
        latency["llm_ms"] = 0.0
        session.add_turn("assistant", response_text)
        await ws.send_json({"type": "response_text", "text": response_text})
        wav_bytes, tts_ms = await loop.run_in_executor(None, synthesize, response_text)
        latency["tts_ms"] = tts_ms
        for chunk in chunk_wav(wav_bytes, chunk_size=4096):
            await ws.send_bytes(chunk)
            await asyncio.sleep(0)
        latency["total_ms"] = (time.perf_counter() - t_start) * 1000
        await ws.send_json({"type": "escalate", "reason": "patient_requested"})
        await ws.send_json({"type": "turn_done", "latency": latency})
        _write_call_log(session.session_id, "ESCALATE", transcript[:512], response_text[:512], latency)
        return

    elif intent in ("BOOK", "RESCHEDULE", "CANCEL"):
        response_text, llm_ms = await loop.run_in_executor(None, run_booking_agent, transcript, session.history)

    elif intent == "FAQ":
        response_text, _chunks, retrieval_ms, llm_ms = await loop.run_in_executor(
            None, run_faq_agent, transcript, session.history, retriever
        )

    elif intent == "SMALL_TALK":
        response_text, llm_ms = await loop.run_in_executor(None, run_small_talk, transcript, session.history)

    else:
        response_text = "I'm sorry, I didn't quite catch that. Could you rephrase — for example, tell me if you'd like to book, reschedule, or cancel an appointment?"
        llm_ms = 0.0

    latency["retrieval_ms"] = retrieval_ms
    latency["llm_ms"] = llm_ms

    session.add_turn("assistant", response_text)
    await ws.send_json({"type": "response_text", "text": response_text})

    session.tts_cancelled = False
    wav_bytes, tts_ms = await loop.run_in_executor(None, synthesize, response_text)
    latency["tts_ms"] = tts_ms

    for chunk in chunk_wav(wav_bytes, chunk_size=4096):
        if session.tts_cancelled:
            break
        await ws.send_bytes(chunk)
        await asyncio.sleep(0)

    latency["total_ms"] = (time.perf_counter() - t_start) * 1000
    await ws.send_json({"type": "turn_done", "latency": latency})
    _write_call_log(session.session_id, intent, transcript[:512], response_text[:512], latency)


async def _process_turn(ws: WebSocket, session: SessionState, pcm_bytes: bytes) -> None:
    t_start = time.perf_counter()
    loop = asyncio.get_event_loop()

    transcript, stt_ms = await loop.run_in_executor(None, transcribe, pcm_bytes)
    if not transcript or len(transcript.strip()) < 2:
        await ws.send_json({"type": "vad_discard"})
        return

    await ws.send_json({"type": "transcript", "text": transcript, "is_final": True})
    await _route_and_respond(ws, session, transcript, t_start, stt_ms)


def _write_call_log(session_id: str, intent: str, transcript: str, outcome: str, latency: dict) -> None:
    db = get_session()
    try:
        db.add(CallLog(
            session_id=session_id, intent=intent,
            transcript_snippet=transcript, outcome=outcome,
            latency_stt_ms=latency.get("stt_ms"),
            latency_retrieval_ms=latency.get("retrieval_ms"),
            latency_llm_ms=latency.get("llm_ms"),
            latency_tts_ms=latency.get("tts_ms"),
            latency_total_ms=latency.get("total_ms"),
        ))
        db.commit()
    except Exception as exc:
        print(f"[CallLog] write failed: {exc}")
    finally:
        db.close()


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    await ws.accept()

    if session_id not in sessions:
        sessions[session_id] = SessionState(session_id=session_id)
    session = sessions[session_id]

    if not session.history:
        greeting = "Hello! Welcome to City General Hospital. How can I help you today?"
        await ws.send_json({"type": "response_text", "text": greeting})
        loop = asyncio.get_event_loop()
        wav_bytes, _ = await loop.run_in_executor(None, synthesize, greeting)
        for chunk in chunk_wav(wav_bytes):
            await ws.send_bytes(chunk)
        session.add_turn("assistant", greeting)

    try:
        while True:
            msg = await ws.receive()

            if "bytes" in msg and msg["bytes"]:
                session.audio_buffer.extend(msg["bytes"])

            elif "text" in msg and msg["text"]:
                data = json.loads(msg["text"])
                msg_type = data.get("type", "")

                if msg_type == "audio_end":
                    pcm_bytes = bytes(session.audio_buffer)
                    session.audio_buffer.clear()
                    if pcm_bytes:
                        try:
                            await _process_turn(ws, session, pcm_bytes)
                        except Exception as exc:
                            print(f"[WS] turn error: {exc}")
                            await ws.send_json({"type": "vad_discard"})
                    else:
                        await ws.send_json({"type": "vad_discard"})

                elif msg_type == "text_input":
                    text = data.get("text", "").strip()
                    if text:
                        try:
                            await ws.send_json({"type": "transcript", "text": text, "is_final": True})
                            await _route_and_respond(ws, session, text, time.perf_counter(), 0.0)
                        except Exception as exc:
                            print(f"[WS] text_input error: {exc}")
                            await ws.send_json({"type": "error", "message": str(exc)})

                elif msg_type == "barge_in":
                    session.tts_cancelled = True
                    session.audio_buffer.clear()
                    await ws.send_json({"type": "barge_in_ack"})

    except WebSocketDisconnect:
        sessions.pop(session_id, None)
    except Exception as exc:
        print(f"[WS] session {session_id} fatal error: {exc}")
        try:
            await ws.send_json({"type": "error", "message": "Connection error — please refresh."})
        except Exception:
            pass
