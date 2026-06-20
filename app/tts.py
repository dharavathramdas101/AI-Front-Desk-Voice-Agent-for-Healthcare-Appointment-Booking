"""
TTS wrapper using pyttsx3.

Note: Piper TTS is not installed in bespin_env2.
pyttsx3 (2.99) is used instead — produces robotic but functional speech.
In production, swap for ElevenLabs or Piper for natural-sounding voice.

pyttsx3 is synchronous and COM-based on Windows. We run synthesis in a
thread executor to avoid blocking the asyncio event loop.

Usage:
    from tts import synthesize
    wav_bytes, elapsed_ms = synthesize("Hello, how can I help you?")
"""

from __future__ import annotations

import io
import os
import tempfile
import threading
import time
import wave


_lock = threading.Lock()  # pyttsx3 is not thread-safe on Windows


def synthesize(text: str) -> tuple[bytes, float]:
    """
    Convert text to speech and return raw WAV bytes.

    Returns:
        (wav_bytes, elapsed_ms)
    """
    import pyttsx3  # imported here to avoid COM init on import

    t0 = time.perf_counter()

    with _lock:
        engine = pyttsx3.init()
        engine.setProperty("rate", 155)   # slightly slower than default for clarity
        engine.setProperty("volume", 1.0)

        # Choose a clearer voice if available
        voices = engine.getProperty("voices")
        for v in voices:
            if "zira" in v.name.lower() or "female" in v.name.lower():
                engine.setProperty("voice", v.id)
                break

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()

            with open(tmp_path, "rb") as f:
                wav_bytes = f.read()
        finally:
            engine.stop()
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    elapsed = (time.perf_counter() - t0) * 1000
    return wav_bytes, elapsed


def chunk_wav(wav_bytes: bytes, chunk_size: int = 4096) -> list[bytes]:
    """
    Split WAV bytes into chunks for streaming over WebSocket.
    First chunk includes the WAV header so the client can decode.
    """
    if not wav_bytes:
        return []
    chunks = []
    offset = 0
    while offset < len(wav_bytes):
        chunks.append(wav_bytes[offset: offset + chunk_size])
        offset += chunk_size
    return chunks
