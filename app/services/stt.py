from __future__ import annotations

import io
import time
import wave

import numpy as np

from app.config import GROQ_API_KEY, WHISPER_MODEL


def transcribe(pcm_bytes: bytes, sample_rate: int = 16000) -> tuple[str, float]:
    """Transcribe raw int16 PCM bytes via Groq Whisper API."""
    if not pcm_bytes:
        return "", 0.0

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    wav_bytes = buf.getvalue()

    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

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
    print(f"[STT] {elapsed:.0f}ms → {text!r}")
    return text, elapsed


_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        import whisper
        _local_model = whisper.load_model(WHISPER_MODEL)
    return _local_model


def transcribe_wav(wav_bytes: bytes) -> tuple[str, float]:
    """Transcribe WAV bytes using local whisper — used by tests."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != 16000:
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr))

    model = _get_local_model()
    t0 = time.perf_counter()
    result = model.transcribe(audio, language="en", task="transcribe", fp16=False)
    return result["text"].strip(), (time.perf_counter() - t0) * 1000
