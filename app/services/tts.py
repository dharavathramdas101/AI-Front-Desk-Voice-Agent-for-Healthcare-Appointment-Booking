from __future__ import annotations

import os
import tempfile
import threading
import time

_lock = threading.Lock()


def synthesize(text: str) -> tuple[bytes, float]:
    """Convert text to speech, return (wav_bytes, elapsed_ms)."""
    import pyttsx3

    t0 = time.perf_counter()
    with _lock:
        engine = pyttsx3.init()
        engine.setProperty("rate", 155)
        engine.setProperty("volume", 1.0)
        for v in engine.getProperty("voices"):
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

    return wav_bytes, (time.perf_counter() - t0) * 1000


def chunk_wav(wav_bytes: bytes, chunk_size: int = 4096) -> list[bytes]:
    """Split WAV into chunks for WebSocket streaming."""
    if not wav_bytes:
        return []
    chunks = []
    offset = 0
    while offset < len(wav_bytes):
        chunks.append(wav_bytes[offset: offset + chunk_size])
        offset += chunk_size
    return chunks
