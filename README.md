# AI Front Desk — Voice Agent for Hospital Appointment Booking

> A production-style voice AI receptionist that handles patient calls for City General Hospital 24/7. Speak into your browser — the agent books, reschedules, and cancels appointments, answers FAQs, and escalates to human staff when needed.

---

## Demo

<video src="demo.mp4" controls width="100%"></video>

---

## What This Project Does

Most hospital front desks lose patients to missed calls, long hold times, and after-hours voicemail. This agent handles the entire appointment lifecycle over voice — no staff required.

A patient opens the browser, holds the microphone button, and speaks naturally:

- *"I want to book a cardiology appointment for next Monday morning"*
- *"Cancel my appointment, my ID is 42"*
- *"What are your visiting hours?"*
- *"I need to speak to a real person"*

The agent understands intent, collects required information turn by turn, queries a live database of doctors and slots, confirms the booking, and responds in synthesized speech — all within ~1.6 seconds round-trip on CPU.

Every conversation is logged with per-stage latency (STT / LLM / TTS) for observability.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BROWSER (Frontend)                        │
│                                                                  │
│  Hold-to-Speak Button                                            │
│       │                                                          │
│       ▼                                                          │
│  MediaRecorder (WebM/Opus)                                       │
│       │  decodeAudioData()                                       │
│       ▼                                                          │
│  OfflineAudioContext  ──► resample to 16kHz mono PCM Int16       │
│       │                                                          │
│       ▼                          ┌─────────────────────┐         │
│  WebSocket ──────────────────────►  FastAPI WS Server  │         │
│  (binary PCM frames)             └────────┬────────────┘         │
│                                           │                      │
│  ◄──────────── JSON messages ─────────────┤                      │
│  ◄──────────── WAV audio chunks ──────────┤                      │
│                                           │                      │
│  Web Audio API plays TTS stream           │                      │
│  ScriptProcessorNode RMS → barge-in       │                      │
└───────────────────────────────────────────┼──────────────────────┘
                                            │
                    ┌───────────────────────▼───────────────────────┐
                    │              FastAPI Server (app/main.py)      │
                    │                                               │
                    │  PCM bytes  ──►  openai-whisper STT           │
                    │                       │ transcript            │
                    │                       ▼                       │
                    │              Intent Classifier (Groq LLM)     │
                    │          BOOK │ RESCHEDULE │ CANCEL │ FAQ     │
                    │          SMALL_TALK │ ESCALATE │ UNCLEAR      │
                    │                       │                       │
                    │         ┌─────────────┼──────────────┐        │
                    │         ▼             ▼              ▼        │
                    │   Booking Agent    FAQ Agent    Small Talk    │
                    │   (tool-calling)   (RAG)        (Groq LLM)   │
                    │         │             │                       │
                    │         ▼             ▼                       │
                    │      SQLite      ChromaDB + BM25              │
                    │   (Doctors,      RRF Fusion                   │
                    │    Slots,        (Hybrid RAG)                 │
                    │    Appointments)                              │
                    │         │                                     │
                    │         └──────────► pyttsx3 TTS              │
                    │                      WAV → 4KB chunks         │
                    │                      streamed to browser      │
                    └───────────────────────────────────────────────┘
```

### Data Flow (one voice turn)

```
User holds button → speaks → releases
        ↓
MediaRecorder captures WebM/Opus
        ↓
Browser decodes → resamples to 16kHz Int16 PCM
        ↓
PCM frames sent over WebSocket (binary)
        ↓
Server receives → sends {"type":"audio_end"}
        ↓
openai-whisper transcribes PCM → text          [~900ms]
        ↓
Groq LLM classifies intent (BOOK/FAQ/etc.)     [~100ms]
        ↓
Agent executes (tool calls / RAG retrieval)    [~450ms]
        ↓
pyttsx3 synthesizes speech → WAV bytes         [~220ms]
        ↓
WAV streamed in 4KB chunks → browser plays
        ↓
Total round-trip: ~1.6 seconds (CPU, no GPU)
```

---

## Features

| Feature | Detail |
|---|---|
| Push-to-talk voice input | Hold orb → speak → release. No VAD false triggers. |
| Appointment booking | Multi-turn dialog collects name, phone, department, date, doctor, slot |
| Smart slot search | Shows numbered options across 14-day window; never says "no slots" without alternatives |
| Existing client recognition | Phone lookup → greet by name → offer "same as last time" rebooking |
| Cancellation policy | Blocks cancellations within 24h; agent pivots to reschedule instead |
| FAQ via hybrid RAG | ChromaDB (dense) + BM25 (keyword) fused with RRF — handles visiting hours, insurance, parking, departments |
| Human escalation | Say "speak to staff" → amber transfer bubble, voice input disabled |
| Barge-in | Speak while agent talks → audio stops instantly, new turn begins |
| Latency dashboard | STT / LLM / TTS / total shown live in UI after each turn |
| Full call logging | Every turn persisted to SQLite with latency breakdown |

---

## Stack

| Layer | Technology | Why |
|---|---|---|
| Backend | FastAPI + uvicorn | Async WebSocket streaming, minimal overhead |
| STT | openai-whisper `base` | Self-hosted, no per-minute cost, works offline |
| LLM | Groq `llama-3.3-70b-versatile` | Fastest hosted LLM API (~450ms), OpenAI-compatible tool calling |
| TTS | pyttsx3 | Fully local, zero latency network hop |
| RAG | ChromaDB + rank_bm25 + RRF | Hybrid retrieval: dense (semantic) + sparse (keyword) fusion |
| DB | SQLite + SQLAlchemy 2.0 | Zero-ops local demo; swap PostgreSQL for production |
| Frontend | Vanilla HTML/JS | WebSocket + Web Audio API, no build step needed |

---

## Project Structure

```
ai-front-desk/
├── app/
│   ├── main.py              FastAPI app, WebSocket endpoint, session manager
│   ├── agent.py             Intent classifier + booking tool-calling + FAQ RAG agent
│   ├── stt.py               openai-whisper wrapper (PCM bytes → transcript)
│   ├── tts.py               pyttsx3 wrapper (text → WAV bytes, chunked streaming)
│   ├── rag.py               HybridRetriever: ChromaDB + BM25 + RRF fusion
│   ├── db.py                SQLAlchemy models: Doctor, Slot, Appointment, CallLog
│   ├── config.py            Settings via pydantic + .env
│   ├── knowledge_base/      8 markdown files (departments, doctors, FAQ, insurance…)
│   └── static/
│       └── index.html       Push-to-talk UI with barge-in and live latency display
├── scripts/
│   ├── seed_db.py           Seed 5 doctors + 280 slots across 14 days
│   ├── seed_rag.py          Chunk + embed knowledge base into ChromaDB
│   └── latency_report.py   Print avg/p50/p95 latency from call logs
├── tests/
│   ├── test_intent.py       10 intent classification tests
│   ├── test_booking.py      13 booking/reschedule/cancel/policy tests
│   └── test_rag.py          6 hybrid retrieval tests
├── demo.mp4                 Demo video
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### Prerequisites

- Conda environment `bespin_env2` with all dependencies installed
- A free [Groq API key](https://console.groq.com)

### 1. Configure environment

```bash
cp .env.example .env
# Open .env and set your key:
# GROQ_API_KEY=your_key_here
```

### 2. Seed the database

```bash
conda activate bespin_env2
python scripts/seed_db.py
# Output: Seeded 5 doctors and 280 slots
```

### 3. Seed the RAG knowledge base

```bash
python scripts/seed_rag.py
# Output: Seeded 17 chunks into ChromaDB
```

### 4. Start the server

```bash
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser. Allow microphone access. Hold the orb and speak.

### 5. Run tests

```bash
python -m pytest tests/ -v
# 30/30 tests pass
```

---

## Voice Flow Examples

**Book an appointment**
> "I want to book an appointment" → "What's your name?" → "Raju" → "Phone number?" → "7386549432" → "Which department?" → "Cardiology" → "Which doctor?" → "Dr. Meera Sharma" → "What date?" → "Next Monday morning" → Agent lists numbered slots → "Option 1" → Confirmed ✓

**Cancel an appointment**
> "Cancel my appointment" → "Phone number?" → Agent finds appointment → Checks 24h policy → Confirms cancellation ✓

**Reschedule**
> "I need to reschedule" → "Phone number?" → Agent shows current appointment → "What new date?" → "Thursday afternoon" → Lists available slots → User picks → Rescheduled ✓

**FAQ**
> "What are your visiting hours?" → RAG retrieves knowledge base → Agent answers in 2 sentences ✓

**Escalation**
> "I want to speak to a real person" → Agent transfers → Amber bubble appears → Input disabled ✓

---

## Latency Breakdown

Measured on Windows 11 laptop, CPU only (no GPU):

| Stage | Avg | p50 | p95 |
|---|---|---|---|
| STT (whisper base, CPU) | ~900ms | ~850ms | ~1100ms |
| Intent (Groq LLM) | ~100ms | ~90ms | ~150ms |
| Agent / RAG (Groq LLM) | ~450ms | ~420ms | ~600ms |
| TTS (pyttsx3, local) | ~220ms | ~210ms | ~290ms |
| **Total round-trip** | **~1.6s** | **~1.5s** | **~2.0s** |

STT dominates on CPU. Swapping to `faster-whisper` on the same `base` model cuts STT by ~4×.

---

## Barge-In

While the agent speaks, the browser runs a `ScriptProcessorNode` that computes RMS amplitude every 1024 samples (~23ms). If RMS exceeds `0.02` for 3 consecutive frames, it sends `{"type":"barge_in"}` over the WebSocket. The server sets `tts_cancelled = True`, breaking the WAV chunk stream on the next iteration. The session immediately resets to listening state.

Most portfolio voice bots omit barge-in. This handles it at both the client (stops playback) and server (cancels stream).

---

## Design Decisions

**Why push-to-talk instead of always-on VAD?**
VAD (voice activity detection) in-browser with WebM/MediaRecorder has a critical bug: the WebM EBML header only exists in the first chunk. A rolling preroll buffer evicts it after ~320ms, causing `decodeAudioData()` to throw silently on every subsequent press. Push-to-talk creates a fresh `MediaRecorder` per press — always gets the header, always works.

**Why hybrid RAG over pure vector search?**
BM25 catches exact keyword matches (doctor names, insurance brands, specific numbers) that dense embeddings blur. Vector search handles paraphrase and synonyms. RRF fusion gets both. Same pattern used in production retrieval systems.

**Why SQLite?**
Zero ops for a local demo. Production: swap for PostgreSQL.

**Why Groq?**
Fastest hosted LLM API for open models (~450ms). OpenAI-compatible — zero rework to switch models or migrate to self-hosted vLLM.

---

## Honest Limitations

- **Not HIPAA-compliant** — no encryption, no access controls, no audit trail
- **Not on a real phone line** — browser WebSocket only; Twilio Media Streams would replace the audio layer
- **Mock data** — doctors and slots are seeded, not connected to an EHR/PMS
- **Robotic TTS** — pyttsx3 is functional but not production-quality; ElevenLabs or Piper gives human-quality voice
- **No auth** — any caller can book for any name; production needs caller ID or patient portal auth

---

## Future Work

1. `faster-whisper` → 4× STT latency reduction on same hardware
2. ElevenLabs / Piper TTS → natural voice, streaming first chunk <300ms
3. Redis session store → survives restarts, supports multi-instance
4. Twilio Media Streams → real phone number, no browser required
5. webrtcvad → continuous hands-free VAD (now unblocked — per-press recorder pattern is the right base)
6. HIPAA compliance → encryption at rest and in transit, audit log, BAA
7. EHR integration → replace SQLite with Epic/Cerner API
8. Confidence-gated escalation → auto-transfer when RAG score is below threshold
