import io
import math
import struct
import wave

import pytest

from app.services.audio_convert import (
    mulaw8k_to_pcm16k,
    pcm16k_to_mulaw8k,
    rms_of_mulaw,
    wav_to_mulaw8k,
)

try:
    import audioop
except ImportError:
    import audioop_lts as audioop  # type: ignore[no-redef]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pcm16(n_samples: int, sample_rate: int = 16000, freq: int = 440) -> bytes:
    """Generate n_samples of int16 sine at the given sample rate."""
    amplitude = 16000
    samples = [
        int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
        for i in range(n_samples)
    ]
    return struct.pack(f"<{n_samples}h", *samples)


def _make_wav(pcm: bytes, sample_rate: int = 16000, n_channels: int = 1) -> bytes:
    """Wrap raw PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# ── mulaw8k_to_pcm16k ─────────────────────────────────────────────────────────

class TestMulaw8kToPcm16k:
    def test_length_doubles(self):
        # 160 mu-law bytes @ 8 kHz → ~320 PCM16 samples @ 16 kHz → ~640 bytes
        # audioop.ratecv can be 1 sample short at end-of-buffer; allow ±4 bytes
        mulaw = bytes(160)
        result = mulaw8k_to_pcm16k(mulaw)
        assert abs(len(result) - 640) <= 4

    def test_returns_bytes(self):
        assert isinstance(mulaw8k_to_pcm16k(bytes(160)), bytes)

    def test_empty_input_returns_empty(self):
        assert mulaw8k_to_pcm16k(b"") == b""

    def test_single_frame(self):
        # Smallest valid mu-law frame — must not crash
        result = mulaw8k_to_pcm16k(bytes(1))
        assert isinstance(result, bytes)


# ── pcm16k_to_mulaw8k ─────────────────────────────────────────────────────────

class TestPcm16kToMulaw8k:
    def test_length_halves(self):
        # 320 PCM16 samples @ 16 kHz (640 bytes) → 160 mu-law bytes @ 8 kHz
        pcm = _make_pcm16(320)
        result = pcm16k_to_mulaw8k(pcm)
        assert len(result) == 160

    def test_returns_bytes(self):
        assert isinstance(pcm16k_to_mulaw8k(_make_pcm16(320)), bytes)

    def test_empty_input_returns_empty(self):
        assert pcm16k_to_mulaw8k(b"") == b""

    def test_roundtrip_length(self):
        """mulaw8k_to_pcm16k then pcm16k_to_mulaw8k restores original byte count."""
        original = bytes(160)
        pcm = mulaw8k_to_pcm16k(original)
        back = pcm16k_to_mulaw8k(pcm)
        assert len(back) == len(original)


# ── wav_to_mulaw8k ────────────────────────────────────────────────────────────

class TestWavToMulaw8k:
    def test_100ms_at_16k(self):
        # 16 000 samples/s × 0.1 s = 1 600 samples → resample to 800 @ 8 kHz
        pcm = _make_pcm16(1600, sample_rate=16000)
        wav = _make_wav(pcm, sample_rate=16000)
        result = wav_to_mulaw8k(wav)
        assert len(result) == 800

    def test_44100hz_input(self):
        # 44 100 samples/s × 0.1 s ≈ 441 samples → resample to ~441 * 8000 / 44100 ≈ 80 bytes
        pcm = _make_pcm16(4410, sample_rate=44100)
        wav = _make_wav(pcm, sample_rate=44100)
        result = wav_to_mulaw8k(wav)
        assert len(result) > 0
        # 4410 * 8000 / 44100 = 800 samples exactly
        assert len(result) == 800

    def test_stereo_input(self):
        # Stereo WAV (n_channels=2) should be mixed to mono then converted
        n_samples = 1600
        # Stereo interleaved: left + right = same sine twice
        pcm_mono = _make_pcm16(n_samples, sample_rate=16000)
        pcm_stereo = b"".join(
            struct.pack("<hh", s, s)
            for s in struct.unpack(f"<{n_samples}h", pcm_mono)
        )
        wav = _make_wav(pcm_stereo, sample_rate=16000, n_channels=2)
        result = wav_to_mulaw8k(wav)
        assert len(result) == 800  # same as mono equivalent

    def test_returns_bytes(self):
        wav = _make_wav(_make_pcm16(160), sample_rate=16000)
        assert isinstance(wav_to_mulaw8k(wav), bytes)


# ── rms_of_mulaw ──────────────────────────────────────────────────────────────

class TestRmsOfMulaw:
    def test_silence_has_low_rms(self):
        # Encode silent PCM (all zeros) to mu-law → RMS should be near zero
        silent_pcm = bytes(320)  # 160 int16 samples of silence
        silent_mulaw = audioop.lin2ulaw(silent_pcm, 2)
        assert rms_of_mulaw(silent_mulaw) < 50

    def test_signal_has_higher_rms(self):
        # Encode a sine wave PCM to mu-law → RMS should be measurably nonzero
        signal_pcm = _make_pcm16(160)
        signal_mulaw = audioop.lin2ulaw(signal_pcm, 2)
        assert rms_of_mulaw(signal_mulaw) > 100

    def test_signal_louder_than_silence(self):
        silent_pcm = bytes(320)
        signal_pcm = _make_pcm16(160)
        silent_mulaw = audioop.lin2ulaw(silent_pcm, 2)
        signal_mulaw = audioop.lin2ulaw(signal_pcm, 2)
        assert rms_of_mulaw(signal_mulaw) > rms_of_mulaw(silent_mulaw)

    def test_empty_input_returns_zero(self):
        assert rms_of_mulaw(b"") == 0
