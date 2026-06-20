"""
STT wrapper.

Primary path  : Groq Whisper API (whisper-large-v3-turbo) — high accuracy, fast.
Fallback/tests: local openai-whisper base model (transcribe / transcribe_wav).
"""

from __future__ import annotations

import io
import os
import time
import wave

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__))
import config

# ── Groq Whisper (primary) ────────────────────────────────────────────────────

def transcribe(pcm_bytes: bytes, sample_rate: int = 16000) -> tuple[str, float]:
    """
    Transcribe raw int16 PCM bytes via Groq Whisper API.
    PCM is wrapped into a WAV container before sending (no ffmpeg needed).
    Returns (transcript_text, elapsed_ms).
    """
    if not pcm_bytes:
        return "", 0.0

    # Wrap PCM → in-memory WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # int16 = 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    wav_bytes = buf.getvalue()

    from groq import Groq
    client = Groq(api_key=config.GROQ_API_KEY)

    t0 = time.perf_counter()
    result = client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=("audio.wav", wav_bytes, "audio/wav"),
        language="en",
        response_format="text",
        prompt="Hospital front desk conversation. Patient may say: Hi, Hello, Bye, book appointment, cancel, reschedule, visiting hours, doctor name.",
    )
    elapsed = (time.perf_counter() - t0) * 1000
    text = result.strip() if isinstance(result, str) else result.text.strip()
    print(f"[STT/whisper-large-v3] {elapsed:.0f}ms → {text!r}")
    return text, elapsed


# ── Local whisper (tests / offline fallback) ──────────────────────────────────

_model = None

def _get_model():
    global _model
    if _model is None:
        import whisper
        print(f"[STT/local] Loading whisper model: {config.WHISPER_MODEL}")
        _model = whisper.load_model(config.WHISPER_MODEL)
        print("[STT/local] Model loaded.")
    return _model


def transcribe_wav(wav_bytes: bytes) -> tuple[str, float]:
    """Transcribe a WAV file given as bytes — used by tests (calls local whisper)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())

    audio_int16 = np.frombuffer(pcm, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    if sr != 16000:
        import scipy.signal
        audio_float32 = scipy.signal.resample(
            audio_float32, int(len(audio_float32) * 16000 / sr)
        )

    import whisper as _whisper
    model = _get_model()
    t0 = time.perf_counter()
    result = model.transcribe(audio_float32, language="en", task="transcribe", fp16=False)
    elapsed = (time.perf_counter() - t0) * 1000
    return result["text"].strip(), elapsed
