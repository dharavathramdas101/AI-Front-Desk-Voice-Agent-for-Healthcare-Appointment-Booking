from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

import app.config as config
from app.api.routes.websocket import SessionState
from app.models.db import CallLog, get_session
from app.services.audio_convert import mulaw8k_to_pcm16k, rms_of_mulaw, wav_to_mulaw8k
from app.services.stt import transcribe
from app.services.tts import synthesize
from app.services.agent.intent import classify_intent
from app.services.agent.booking import run_booking_agent
from app.services.agent.faq import run_faq_agent, run_small_talk

router = APIRouter()

# ── Constants ─────────────────────────────────────────────────────────────────

SPEECH_THRESHOLD_RMS = 200   # raise to ~300 if background noise is a problem
SILENCE_FRAMES_TRIGGER = 30  # 30 × 20ms = 600ms silence → end of utterance
FRAME_SIZE = 160             # 20ms of μ-law audio at 8 kHz

_GREETING = "Hello! Welcome to City General Hospital. How can I help you today?"

# ── Per-call state ─────────────────────────────────────────────────────────────


@dataclass
class TwilioSession:
    call_sid: str
    stream_sid: str = ""
    state: SessionState = field(default_factory=lambda: SessionState(session_id=""))
    audio_buffer: bytearray = field(default_factory=bytearray)
    speech_detected: bool = False
    silence_frames: int = 0
    processing: bool = False
    tts_active: bool = False


phone_sessions: dict[str, TwilioSession] = {}

# ── TwiML incoming-call webhook ───────────────────────────────────────────────


@router.post("/twilio/incoming-call")
async def incoming_call(request: Request) -> Response:
    """Return TwiML that connects the call to our media-stream WebSocket."""
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")

    # Signature validation disabled for local dev (ngrok URL mismatch)
    # Re-enable in production by uncommenting:
    # if config.TWILIO_AUTH_TOKEN:
    #     from twilio.request_validator import RequestValidator
    #     validator = RequestValidator(config.TWILIO_AUTH_TOKEN)
    #     signature = request.headers.get("X-Twilio-Signature", "")
    #     if not validator.validate(str(request.url), dict(form), signature):
    #         return Response(content="Forbidden", status_code=403)

    domain = (
        config.PUBLIC_BASE_URL
        .replace("https://", "")
        .replace("http://", "")
        .rstrip("/")
    )

    from twilio.twiml.voice_response import VoiceResponse, Connect
    resp = VoiceResponse()
    connect = Connect()
    connect.stream(url=f"wss://{domain}/twilio/media-stream/{call_sid}")
    resp.append(connect)

    return Response(content=str(resp), media_type="application/xml")

# ── Helpers ───────────────────────────────────────────────────────────────────


async def _send_tts(ws: WebSocket, ts: TwilioSession, text: str) -> float:
    """Synthesize text, convert to μ-law 8 kHz, stream to Twilio. Returns tts_ms."""
    loop = asyncio.get_event_loop()
    wav_bytes, tts_ms = await loop.run_in_executor(None, synthesize, text)
    mulaw = wav_to_mulaw8k(wav_bytes)

    ts.tts_active = True
    ts.state.tts_cancelled = False
    try:
        for i in range(0, len(mulaw), FRAME_SIZE):
            if ts.state.tts_cancelled:
                break
            chunk = mulaw[i: i + FRAME_SIZE]
            payload = base64.b64encode(chunk).decode()
            await ws.send_text(json.dumps({
                "event": "media",
                "streamSid": ts.stream_sid,
                "media": {"payload": payload},
            }))
            await asyncio.sleep(0.02)  # pace at real-time; also yields for barge-in
    finally:
        ts.tts_active = False

    return tts_ms


async def _process_phone_turn(ws: WebSocket, ts: TwilioSession, pcm16k: bytes) -> None:
    """Full STT → intent → agent → TTS turn for one phone utterance."""
    ts.processing = True
    t_start = time.perf_counter()
    loop = asyncio.get_event_loop()

    try:
        transcript, stt_ms = await loop.run_in_executor(None, transcribe, pcm16k)
        if not transcript or len(transcript.strip()) < 2:
            return

        print(f"[Twilio:{ts.call_sid}] transcript: {transcript!r}")
        latency: dict[str, float] = {"stt_ms": stt_ms}

        ts.state.add_turn("user", transcript)

        intent, intent_ms = await loop.run_in_executor(
            None, classify_intent, transcript, ts.state.history
        )
        latency["intent_ms"] = intent_ms

        retrieval_ms = 0.0
        llm_ms = 0.0

        if intent == "ESCALATE":
            response_text = (
                "Of course! Let me connect you with one of our team members. "
                "Please hold for a moment."
            )
        elif intent in ("BOOK", "RESCHEDULE", "CANCEL"):
            response_text, llm_ms = await loop.run_in_executor(
                None, run_booking_agent, transcript, ts.state.history
            )
        elif intent == "FAQ":
            retriever = ws.app.state.retriever
            response_text, _chunks, retrieval_ms, llm_ms = await loop.run_in_executor(
                None, run_faq_agent, transcript, ts.state.history, retriever
            )
        elif intent == "SMALL_TALK":
            response_text, llm_ms = await loop.run_in_executor(
                None, run_small_talk, transcript, ts.state.history
            )
        else:
            response_text = (
                "I'm sorry, I didn't quite catch that. "
                "You can book, reschedule, or cancel an appointment, "
                "or ask about visiting hours."
            )
            llm_ms = 0.0

        latency["retrieval_ms"] = retrieval_ms
        latency["llm_ms"] = llm_ms

        ts.state.add_turn("assistant", response_text)

        tts_ms = await _send_tts(ws, ts, response_text)
        latency["tts_ms"] = tts_ms
        latency["total_ms"] = (time.perf_counter() - t_start) * 1000

        _write_phone_call_log(ts.state.session_id, intent, transcript, response_text, latency)

    except Exception as exc:
        import traceback
        print(f"[Twilio:{ts.call_sid}] turn error: {exc}")
        traceback.print_exc()
    finally:
        ts.processing = False


def _write_phone_call_log(
    session_id: str, intent: str, transcript: str, outcome: str, latency: dict
) -> None:
    db = get_session()
    try:
        db.add(CallLog(
            session_id=session_id,
            intent=intent,
            transcript_snippet=transcript[:512],
            outcome=outcome[:512],
            latency_stt_ms=latency.get("stt_ms"),
            latency_retrieval_ms=latency.get("retrieval_ms"),
            latency_llm_ms=latency.get("llm_ms"),
            latency_tts_ms=latency.get("tts_ms"),
            latency_total_ms=latency.get("total_ms"),
            source="phone",
        ))
        db.commit()
    except Exception as exc:
        print(f"[CallLog] phone write failed: {exc}")
    finally:
        db.close()

# ── Media Stream WebSocket ─────────────────────────────────────────────────────


@router.websocket("/twilio/media-stream/{call_sid}")
async def twilio_media_stream(ws: WebSocket, call_sid: str) -> None:
    await ws.accept()
    ts = TwilioSession(call_sid=call_sid)
    phone_sessions[call_sid] = ts

    try:
        while True:
            text = await ws.receive_text()
            data = json.loads(text)
            event = data.get("event")

            if event == "connected":
                print(f"[Twilio] connected: {call_sid}")

            elif event == "start":
                ts.stream_sid = data["start"]["streamSid"]
                ts.state = SessionState(session_id=call_sid)
                print(f"[Twilio] stream started sid={ts.stream_sid} call={call_sid}")
                ts.state.add_turn("assistant", _GREETING)
                await _send_tts(ws, ts, _GREETING)

            elif event == "media":
                track = data.get("media", {}).get("track", "inbound")
                if track != "inbound":
                    continue

                payload = data["media"]["payload"]
                mulaw = base64.b64decode(payload)
                rms = rms_of_mulaw(mulaw)
                pcm16k_chunk = mulaw8k_to_pcm16k(mulaw)

                if rms > SPEECH_THRESHOLD_RMS:
                    # Speech frame: barge-in + accumulate
                    if ts.tts_active:
                        ts.state.tts_cancelled = True
                    ts.speech_detected = True
                    ts.silence_frames = 0
                    ts.audio_buffer.extend(pcm16k_chunk)

                elif ts.speech_detected:
                    # Silence after speech: accumulate + check for end-of-utterance
                    ts.silence_frames += 1
                    ts.audio_buffer.extend(pcm16k_chunk)

                    if ts.silence_frames >= SILENCE_FRAMES_TRIGGER and not ts.processing:
                        pcm_snapshot = bytes(ts.audio_buffer)
                        ts.audio_buffer = bytearray()
                        ts.speech_detected = False
                        ts.silence_frames = 0
                        asyncio.create_task(
                            _process_phone_turn(ws, ts, pcm_snapshot)
                        )

            elif event == "stop":
                print(f"[Twilio] call stopped: {call_sid}")
                break

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[Twilio] WS error ({call_sid}): {exc}")
    finally:
        phone_sessions.pop(call_sid, None)
