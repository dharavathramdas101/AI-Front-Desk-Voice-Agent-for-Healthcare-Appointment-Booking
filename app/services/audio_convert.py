from __future__ import annotations

import io
import wave

import numpy as np

try:
    import audioop
except ImportError:
    import audioop_lts as audioop  # type: ignore[no-redef]  # Python 3.13+


def mulaw8k_to_pcm16k(mulaw_bytes: bytes) -> bytes:
    """Decode μ-law 8 kHz payload to int16 PCM at 16 kHz."""
    if not mulaw_bytes:
        return b""
    pcm8k = audioop.ulaw2lin(mulaw_bytes, 2)
    pcm16k, _ = audioop.ratecv(pcm8k, 2, 1, 8000, 16000, None)
    return pcm16k


def pcm16k_to_mulaw8k(pcm16k_bytes: bytes) -> bytes:
    """Encode int16 PCM 16 kHz to μ-law 8 kHz bytes."""
    if not pcm16k_bytes:
        return b""
    pcm8k, _ = audioop.ratecv(pcm16k_bytes, 2, 1, 16000, 8000, None)
    return audioop.lin2ulaw(pcm8k, 2)


def wav_to_mulaw8k(wav_bytes: bytes) -> bytes:
    """Convert a WAV file (any sample rate, mono or stereo) to μ-law 8 kHz bytes."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        src_rate = wf.getframerate()
        n_ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())

    # Stereo → mono via numpy averaging
    if n_ch == 2:
        samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
        raw = samples.mean(axis=1).astype(np.int16).tobytes()

    # Resample to 8 kHz if needed
    if src_rate != 8000:
        raw, _ = audioop.ratecv(raw, 2, 1, src_rate, 8000, None)

    return audioop.lin2ulaw(raw, 2)


def rms_of_mulaw(mulaw_bytes: bytes) -> int:
    """Compute RMS amplitude of μ-law audio — used for VAD silence detection."""
    if not mulaw_bytes:
        return 0
    return audioop.rms(audioop.ulaw2lin(mulaw_bytes, 2), 2)
