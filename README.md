# AI Front Desk — Voice Agent for Hospital Appointment Booking

A voice AI receptionist for City General Hospital. Patients speak into their browser microphone — the agent books, reschedules, and cancels appointments, answers FAQs, and transfers to human staff when requested.

---

## Demo

[▶ Watch Demo Video]([https://github.com/dharavathramdas101/ai-front-desk/raw/main/demo.mp4](https://github.com/dharavathramdas101/AI-Front-Desk-Voice-Agent-for-Healthcare-Appointment-Booking/blob/main/demo.mp4))

![AI Front Desk — booking conversation screenshot](screenshot.png)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BROWSER (Frontend)                        │
│                                                                  │
│  Hold-to-Speak Button (Push-to-Talk)                            │
│       │                                                          │
│       ▼                                                          │
│  MediaRecorder  ──►  decodeAudioData()                          │
│       │                                                          │
│       ▼                                                          │
│  OfflineAudioContext  ──►  resample to 16kHz mono PCM Int16     │
│       │                                                          │
│       ▼  binary PCM frames                                       │
│  WebSocket ─────────────────────────────────────────────────►   │
│                                                                  │
│  ◄─────────── JSON control messages (transcript, intent, etc.)  │
│  ◄─────────── WAV audio chunks (TTS streamed in 4KB blocks)     │
│                                                                  │
│  Web Audio API  ──►  plays TTS stream                           │
│  ScriptProcessorNode RMS  ──►  barge-in detection               │
└──────────────────────────────────────────────────────────────────┘
                          │ WebSocket
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                   FastAPI Server (app/main.py)                   │
│                                                                  │
│  PCM bytes  ──►  openai-whisper STT  ──►  transcript            │
│                                              │                   │
│                                              ▼                   │
│                                 Intent Classifier (Groq LLM)    │
│                                              │                   │
│                    ┌─────────────────────────┼──────────────┐   │
│                    ▼                         ▼              ▼   │
│             Booking Agent               FAQ Agent       Small   │
│             (Groq tool-calling)         (Hybrid RAG)    Talk    │
│                    │                         │                   │
│                    ▼                         ▼                   │
│                 SQLite                  ChromaDB + BM25          │
│            (Doctors, Slots,             RRF Fusion               │
│             Appointments,                                        │
│             Call Logs)                                           │
│                    │                                             │
│                    └─────────────►  pyttsx3 TTS                 │
│                                    WAV ──► 4KB chunks            │
│                                    streamed to browser           │
└──────────────────────────────────────────────────────────────────┘
```

### One voice turn — step by step

```
User holds orb → speaks → releases
        ↓
MediaRecorder captures WebM/Opus  (fresh recorder per press)
        ↓
Browser decodes → resamples to 16kHz Int16 PCM
        ↓
PCM frames sent over WebSocket (binary)
        ↓
Server accumulates → audio_end signal received
        ↓
openai-whisper transcribes PCM → text           (~900ms)
        ↓
Groq LLM classifies intent                      (~100ms)
        ↓
Agent executes (tool calls / RAG retrieval)     (~450ms)
        ↓
pyttsx3 synthesizes speech → WAV bytes          (~220ms)
        ↓
WAV streamed in 4KB chunks → browser plays
        ↓
Total round-trip: ~1.6 seconds  (CPU, no GPU)
```

---

## Features

| Feature | Detail |
|---|---|
| Push-to-talk voice input | Hold orb → speak → release. Fresh `MediaRecorder` per press avoids WebM header eviction bug. |
| Appointment booking | Multi-turn dialog: collects name, phone, department, doctor, date — one question at a time |
| Smart slot search | Numbered options across 14-day window; never says "no slots" without listing alternatives |
| Existing client recognition | Phone lookup → greet by name → offer "same as last time" rebooking |
| Cancellation policy | Blocks cancellations within 24h; agent suggests reschedule instead |
| FAQ via hybrid RAG | ChromaDB (dense) + BM25 (keyword) fused with RRF — visiting hours, insurance, parking, departments |
| Human escalation | "Speak to staff" → amber transfer bubble appears, voice input disabled |
| Barge-in | Speak while agent talks → audio stops instantly, new turn begins |
| Live latency display | STT / LLM / TTS / total shown in UI after every turn |
| Call logging | Every turn logged to SQLite with full latency breakdown |

---

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + uvicorn |
| STT | openai-whisper `base` (self-hosted, offline) |
| LLM | Groq `llama-3.3-70b-versatile` |
| TTS | pyttsx3 (local, zero network cost) |
| RAG | ChromaDB + rank_bm25 + RRF fusion |
| DB | SQLite + SQLAlchemy 2.0 |
| Frontend | Vanilla HTML/JS — WebSocket + Web Audio API |

---

## Project Structure

```
ai-front-desk/
├── app/
│   ├── main.py              FastAPI app, WebSocket endpoint, session manager
│   ├── agent.py             Intent classifier + booking agent + FAQ RAG agent
│   ├── stt.py               openai-whisper wrapper (PCM → transcript)
│   ├── tts.py               pyttsx3 wrapper (text → WAV chunks)
│   ├── rag.py               HybridRetriever: ChromaDB + BM25 + RRF
│   ├── db.py                SQLAlchemy models: Doctor, Slot, Appointment, CallLog
│   ├── config.py            Settings from .env
│   ├── knowledge_base/      8 markdown files (departments, doctors, FAQ, insurance…)
│   └── static/
│       └── index.html       Push-to-talk UI with barge-in and latency panel
├── scripts/
│   ├── seed_db.py           Seed 5 doctors + 280 slots across 14 days
│   ├── seed_rag.py          Chunk + embed knowledge base into ChromaDB
│   └── latency_report.py   Print avg/p50/p95 latency from call logs
├── tests/
│   ├── test_intent.py       Intent classification tests
│   ├── test_booking.py      Booking / reschedule / cancel / policy tests
│   └── test_rag.py          Hybrid retrieval tests
├── demo.mp4
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### Prerequisites

- Python environment with all dependencies (see `requirements.txt`)
- A free [Groq API key](https://console.groq.com)

### 1. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your values:

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | **Yes** | — | Groq API key from console.groq.com |
| `LLM_MODEL` | No | `llama-3.3-70b-versatile` | Groq model ID |
| `EMBED_MODEL` | No | `all-MiniLM-L6-v2` | Sentence-transformers model for RAG |
| `WHISPER_MODEL` | No | `base` | Whisper model size (`tiny`, `base`, `small`) |
| `CHROMA_PATH` | No | `./chroma_db` | ChromaDB storage path |
| `DATABASE_URL` | No | `sqlite:///./ai_front_desk.db` | SQLAlchemy database URL |
| `CANCELLATION_WINDOW_HOURS` | No | `24` | Hours before appointment within which cancellation is blocked |

### 2. Seed the database

```bash
python scripts/seed_db.py
# Seeded 5 doctors and 280 slots
```

### 3. Seed the RAG knowledge base

```bash
python scripts/seed_rag.py
# Seeded 17 chunks into ChromaDB
```

### 4. Start the server

```bash
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) — allow microphone access — hold the orb to speak.

### 5. Run tests

```bash
python -m pytest tests/ -v
# 30/30 pass
```

---

## Voice Flow Examples

**Book an appointment**
> "I want to book an appointment" → name → phone → department → doctor → date → numbered slots listed → pick option → confirmed

**Reschedule**
> "Reschedule my appointment" → phone → agent shows current booking → new date → new slot options → confirmed

**Cancel**
> "Cancel my appointment" → phone → agent checks 24h policy → confirmed (or blocked if within window)

**FAQ**
> "What are your visiting hours?" → RAG retrieves → 2-sentence answer

**Escalation**
> "I want to speak to a real person" → amber transfer bubble → voice input disabled

---

## Latency

Measured on Windows 11, CPU only (no GPU):

| Stage | Avg | p50 | p95 |
|---|---|---|---|
| STT (whisper base, CPU) | ~900ms | ~850ms | ~1100ms |
| Intent (Groq) | ~100ms | ~90ms | ~150ms |
| Agent / RAG (Groq) | ~450ms | ~420ms | ~600ms |
| TTS (pyttsx3, local) | ~220ms | ~210ms | ~290ms |
| **Total** | **~1.6s** | **~1.5s** | **~2.0s** |

---

## Barge-In

While the agent speaks, the browser runs a `ScriptProcessorNode` computing RMS amplitude every 1024 samples. If RMS exceeds `0.02` for 3 consecutive frames (~70ms), it sends `{"type":"barge_in"}` over the WebSocket. The server sets `tts_cancelled = True`, which breaks the WAV chunk stream on the next iteration. Session resets to listening state immediately.
