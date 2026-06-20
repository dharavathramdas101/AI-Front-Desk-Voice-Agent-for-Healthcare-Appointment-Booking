"""
AI Front Desk agent:
  1. classify_intent()  — LLM routes utterance to BOOK/RESCHEDULE/CANCEL/FAQ/SMALL_TALK/UNCLEAR
  2. run_booking_agent() — Groq tool-calling for appointment CRUD
  3. run_faq_agent()    — hybrid RAG → grounded LLM response

HARD RULES (baked into every system prompt):
  - Never give medical advice, diagnoses, or clinical recommendations.
  - If the patient asks anything clinical, route to medical staff.
  - Keep all spoken responses to 2–3 sentences max (voice UX).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from typing import Any

from groq import Groq

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config
from db import Appointment, Doctor, Slot, get_session
from rag import Chunk, HybridRetriever

if not config.GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")
_client = Groq(api_key=config.GROQ_API_KEY)

# ──────────────────────────────────────────────
# System prompt fragments
# ──────────────────────────────────────────────

_HARD_RULES = """
ABSOLUTE RULES — never violate these:
- You are an administrative front desk AI. You handle only: appointment booking, rescheduling, cancellation, visiting hours, doctor info, insurance, parking, departments, and general FAQs.
- NEVER provide medical advice, diagnoses, treatment recommendations, or drug information.
- If the patient asks ANY clinical question (symptoms, medications, test results, treatment options), say exactly: "I can connect you with our medical team for that — I handle scheduling and general information only."
- All spoken responses must be 2 to 3 sentences maximum. You are a voice agent. Be concise.
- NEVER output stage directions, parenthetical remarks, or meta-commentary in parentheses — e.g., never write "(speaking clearly)", "(pause)", "(checking)", "(incomplete sentence)". Output only what you would say aloud to the patient.
"""

_INTENT_SYSTEM = f"""You are the intent classifier for City General Hospital's voice front desk.
Classify the patient's utterance into exactly one of: BOOK, RESCHEDULE, CANCEL, FAQ, SMALL_TALK, ESCALATE, UNCLEAR.

BOOK       — patient wants to book a new appointment
RESCHEDULE — patient wants to change an existing appointment date/time
CANCEL     — patient wants to cancel an existing appointment
FAQ        — patient is asking about visiting hours, departments, doctors, insurance, parking, or general hospital info
SMALL_TALK — greeting, thanks, goodbye, or unrelated chatter
ESCALATE   — patient explicitly asks to speak with a human, staff member, real person, or representative ("speak to someone", "connect me to staff", "I want a human", "transfer me")
UNCLEAR    — cannot determine intent with confidence

CRITICAL CONTEXT RULE: Check the conversation history carefully.
If the assistant's most recent message was asking for information to complete a BOOK, RESCHEDULE, or CANCEL task
(e.g., asking for appointment ID, patient name, phone number, preferred date, department),
then the patient's reply — even if short, vague, or unclear — is almost certainly a continuation of that same task.
Classify it as the SAME intent (BOOK / RESCHEDULE / CANCEL), NOT as SMALL_TALK or UNCLEAR.
Only override this rule if the patient explicitly asks about a completely different topic.

{_HARD_RULES}

Respond with valid JSON only: {{"intent": "<INTENT>", "reasoning": "<one sentence>"}}
"""

_BOOKING_SYSTEM = """You are City General Hospital's appointment booking assistant.
Use the provided tools to check availability and book, reschedule, or cancel appointments.

CRITICAL: You cannot look up patients by name. You CAN look up by phone number using lookup_patient_history. Never say "I found your information by name." If the patient says "cancel my appointment" without an ID, ask for their phone number first — then call lookup_patient_history to find their appointments.

EXISTING CLIENT RECOGNITION: When the patient provides their phone number, ALWAYS call lookup_patient_history first. If they are an existing client, greet them by name and show their recent appointments. If they say "same as last time" or "same doctor", use the doctor and department from their most recent appointment when calling check_availability. This makes rebooking seamless.

MULTI-TURN CONTEXT: If you previously asked the patient a question (e.g., "What is your appointment ID?", "What is your name?", "What date works for you?"), and the patient's latest message is a reply — treat it as the answer to your question and continue the flow. Do NOT restart or ask the same question again.

SHOWING AVAILABILITY — IMPORTANT: When check_availability returns slots, ALWAYS list them as numbered options so the patient can choose. Example format:
"I have these slots available:
1) Monday, 22 June at 9:00 AM — Dr. Meera Sharma (Cardiology)
2) Monday, 22 June at 11:00 AM — Dr. Arjun Nair (Cardiology)
3) Tuesday, 23 June at 9:00 AM — Dr. Meera Sharma (Cardiology)
Which would you prefer?"
If slots_on_requested_date is empty, say the requested date has no slots, then list upcoming_slots as alternatives. NEVER just say "no slots available" without listing what IS available.

DATE RESOLUTION: Use today's date (injected at start of conversation) to compute exact calendar dates for "tomorrow", "next Monday", "this Friday", etc. Always pass YYYY-MM-DD format to check_availability.

When booking: ask for patient name, phone number, preferred department/doctor, and date. Ask one thing at a time.
When rescheduling/cancelling: ask for phone number first (to look up history), or appointment ID if they know it.
Always confirm the exact date, time, and doctor before calling book_appointment.
NEVER mention or invent external services, websites, or phone numbers not in this conversation.
""" + _HARD_RULES

_FAQ_SYSTEM = f"""You are City General Hospital's information assistant.
Answer the patient's question using ONLY the provided context passages.
If the context does not contain enough information, say: "I don't have that information — please hold while I connect you with our team."
{_HARD_RULES}
"""

# ──────────────────────────────────────────────
# Tool definitions (OpenAI-compatible format for Groq)
# ──────────────────────────────────────────────

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Return available appointment slots for a department or doctor on a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {"type": "string", "description": "Department name, e.g. Cardiology"},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book an appointment for a patient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string"},
                    "phone": {"type": "string"},
                    "doctor_name": {"type": "string", "description": "Full doctor name, e.g. Dr. Meera Sharma"},
                    "slot_id": {"type": "integer", "description": "Slot ID from check_availability"},
                },
                "required": ["patient_name", "phone", "doctor_name", "slot_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_appointment",
            "description": "Move an existing appointment to a new slot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {"type": "integer"},
                    "new_slot_id": {"type": "integer", "description": "New slot ID from check_availability"},
                },
                "required": ["appointment_id", "new_slot_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": "Cancel an existing appointment by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {"type": "integer"},
                },
                "required": ["appointment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_patient_history",
            "description": "Look up a patient's appointment history by phone number. Call this when the patient provides their phone number to check if they are an existing client and to find their appointment IDs for rescheduling or cancellation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {"type": "string", "description": "Patient phone number"},
                },
                "required": ["phone"],
            },
        },
    },
]


# ──────────────────────────────────────────────
# Tool implementations (DB calls)
# ──────────────────────────────────────────────

def check_availability(department: str | None, date: str) -> dict:
    try:
        target = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {"error": f"Invalid date format: {date}. Use YYYY-MM-DD."}

    session = get_session()
    try:
        # First try exact date
        q = (
            session.query(Slot, Doctor)
            .join(Doctor, Slot.doctor_id == Doctor.id)
            .filter(Slot.slot_datetime >= target)
            .filter(Slot.slot_datetime < target + timedelta(days=1))
            .filter(Slot.is_available == True)
        )
        if department:
            q = q.filter(Doctor.department.ilike(f"%{department}%"))
        exact_rows = q.order_by(Slot.slot_datetime).all()

        # Always also fetch next 14 days so agent can suggest alternatives
        q2 = (
            session.query(Slot, Doctor)
            .join(Doctor, Slot.doctor_id == Doctor.id)
            .filter(Slot.slot_datetime >= target)
            .filter(Slot.slot_datetime < target + timedelta(days=14))
            .filter(Slot.is_available == True)
        )
        if department:
            q2 = q2.filter(Doctor.department.ilike(f"%{department}%"))
        all_rows = q2.order_by(Slot.slot_datetime).limit(12).all()

        def fmt(slot, doctor):
            return {
                "slot_id": slot.id,
                "doctor": doctor.name,
                "department": doctor.department,
                "date": slot.slot_datetime.strftime("%A, %d %B %Y"),
                "time": slot.slot_datetime.strftime("%I:%M %p"),
                "datetime_key": slot.slot_datetime.strftime("%Y-%m-%d %H:%M"),
            }

        return {
            "requested_date": target.strftime("%A, %d %B %Y"),
            "slots_on_requested_date": [fmt(s, d) for s, d in exact_rows],
            "upcoming_slots": [fmt(s, d) for s, d in all_rows],
        }
    finally:
        session.close()


def book_appointment(patient_name: str, phone: str, doctor_name: str, slot_id: int) -> dict:
    session = get_session()
    try:
        slot = session.get(Slot, slot_id)
        if slot is None:
            return {"error": f"Slot {slot_id} not found."}
        if not slot.is_available:
            return {"error": "That slot is no longer available."}
        doctor = session.get(Doctor, slot.doctor_id)
        if doctor is None or doctor_name.lower() not in doctor.name.lower():
            return {"error": f"Doctor mismatch: slot {slot_id} belongs to {doctor.name if doctor else 'unknown'}."}

        slot.is_available = False
        appt = Appointment(
            patient_name=patient_name,
            phone=phone,
            doctor_id=slot.doctor_id,
            slot_id=slot_id,
            status="booked",
        )
        session.add(appt)
        session.commit()
        session.refresh(appt)
        return {
            "appointment_id": appt.id,
            "patient": patient_name,
            "doctor": doctor.name,
            "datetime": slot.slot_datetime.strftime("%A, %d %B %Y at %I:%M %p"),
            "status": "booked",
        }
    finally:
        session.close()


def reschedule_appointment(appointment_id: int, new_slot_id: int) -> dict:
    session = get_session()
    try:
        appt = session.get(Appointment, appointment_id)
        if appt is None:
            return {"error": f"Appointment {appointment_id} not found."}
        if appt.status == "cancelled":
            return {"error": "Cannot reschedule a cancelled appointment."}

        new_slot = session.get(Slot, new_slot_id)
        if new_slot is None:
            return {"error": f"Slot {new_slot_id} not found."}
        if not new_slot.is_available:
            return {"error": "New slot is not available."}

        # Free the old slot
        old_slot = session.get(Slot, appt.slot_id)
        if old_slot:
            old_slot.is_available = True

        new_slot.is_available = False
        appt.slot_id = new_slot_id
        appt.status = "rescheduled"
        session.commit()
        return {
            "appointment_id": appointment_id,
            "new_datetime": new_slot.slot_datetime.strftime("%A, %d %B %Y at %I:%M %p"),
            "status": "rescheduled",
        }
    finally:
        session.close()


def cancel_appointment(appointment_id: int) -> dict:
    session = get_session()
    try:
        appt = session.get(Appointment, appointment_id)
        if appt is None:
            return {"error": f"Appointment {appointment_id} not found."}
        if appt.status == "cancelled":
            return {"error": "Appointment already cancelled."}

        slot = session.get(Slot, appt.slot_id)
        if slot:
            # Cancellation policy: enforce notice window
            now = datetime.now()
            hours_until = (slot.slot_datetime - now).total_seconds() / 3600
            if 0 < hours_until < config.CANCELLATION_WINDOW_HOURS:
                return {
                    "policy_violation": True,
                    "hours_until_appointment": round(hours_until, 1),
                    "slot_id": slot.id,
                    "message": (
                        f"Appointment is in {round(hours_until, 1)} hours. "
                        f"Our policy requires {config.CANCELLATION_WINDOW_HOURS}h advance notice to cancel. "
                        "Please suggest the patient reschedule to a later date instead."
                    ),
                }
            slot.is_available = True
        appt.status = "cancelled"
        session.commit()
        return {"appointment_id": appointment_id, "status": "cancelled"}
    finally:
        session.close()


def lookup_patient_history(phone: str) -> dict:
    session = get_session()
    try:
        rows = (
            session.query(Appointment, Slot, Doctor)
            .join(Slot, Appointment.slot_id == Slot.id)
            .join(Doctor, Appointment.doctor_id == Doctor.id)
            .filter(Appointment.phone == phone)
            .filter(Appointment.status != "cancelled")
            .order_by(Appointment.created_at.desc())
            .limit(3)
            .all()
        )
        if not rows:
            return {"found": False, "message": "No previous appointments found for this phone number."}
        history = [
            {
                "appointment_id": appt.id,
                "doctor": doctor.name,
                "department": doctor.department,
                "datetime": slot.slot_datetime.strftime("%A, %d %B %Y at %I:%M %p"),
                "status": appt.status,
            }
            for appt, slot, doctor in rows
        ]
        return {
            "found": True,
            "patient_name": rows[0][0].patient_name,
            "history": history,
        }
    finally:
        session.close()


_TOOL_MAP = {
    "check_availability": lambda args: check_availability(args.get("department"), args["date"]),
    "book_appointment": lambda args: book_appointment(args["patient_name"], args["phone"], args["doctor_name"], args["slot_id"]),
    "reschedule_appointment": lambda args: reschedule_appointment(args["appointment_id"], args["new_slot_id"]),
    "cancel_appointment": lambda args: cancel_appointment(args["appointment_id"]),
    "lookup_patient_history": lambda args: lookup_patient_history(args["phone"]),
}


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def classify_intent(utterance: str, history: list[dict]) -> tuple[str, float]:
    """
    Returns (intent_label, elapsed_ms).
    intent_label ∈ {BOOK, RESCHEDULE, CANCEL, FAQ, SMALL_TALK, UNCLEAR}
    """
    t0 = time.perf_counter()
    messages = [{"role": "system", "content": _INTENT_SYSTEM}]
    # Include last 2 turns for context
    messages.extend(history[-4:])
    messages.append({"role": "user", "content": utterance})

    resp = _client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        max_tokens=64,
        temperature=0.0,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    raw = resp.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
        return data.get("intent", "UNCLEAR"), elapsed
    except json.JSONDecodeError:
        # Fallback: scan for intent keyword
        for label in ("BOOK", "RESCHEDULE", "CANCEL", "FAQ", "SMALL_TALK"):
            if label in raw.upper():
                return label, elapsed
        return "UNCLEAR", elapsed


def run_booking_agent(utterance: str, history: list[dict]) -> tuple[str, float]:
    """
    Multi-turn tool-calling agent for BOOK/RESCHEDULE/CANCEL.
    Returns (spoken_response, elapsed_ms).
    """
    t0 = time.perf_counter()
    today = datetime.now().strftime("%A, %d %B %Y")
    system = _BOOKING_SYSTEM + f"\n\nTODAY'S DATE: {today}. Use this to resolve relative dates like 'tomorrow', 'next Monday', 'this Friday'."
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": utterance})

    # Agentic loop: allow up to 5 tool call rounds
    for _ in range(5):
        resp = _client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            tools=_TOOLS,
            tool_choice="auto",
            max_tokens=512,
            temperature=0.2,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            elapsed = (time.perf_counter() - t0) * 1000
            return msg.content or "I'm sorry, I couldn't complete that action.", elapsed

        # Execute tool calls
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]})
        for tc in msg.tool_calls:
            fn = _TOOL_MAP.get(tc.function.name)
            if fn is None:
                result = {"error": f"Unknown tool: {tc.function.name}"}
            else:
                args = json.loads(tc.function.arguments)
                result = fn(args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    elapsed = (time.perf_counter() - t0) * 1000
    return "I wasn't able to complete that — please try again or speak with a staff member.", elapsed


def run_faq_agent(utterance: str, history: list[dict], retriever: HybridRetriever) -> tuple[str, list[Chunk], float, float]:
    """
    Retrieve + LLM compose answer.
    Returns (spoken_response, retrieved_chunks, retrieval_ms, llm_ms).
    """
    t_ret0 = time.perf_counter()
    chunks = retriever.retrieve(utterance, top_k=config.TOP_K_RETRIEVAL)
    retrieval_ms = (time.perf_counter() - t_ret0) * 1000

    if not chunks:
        return (
            "I don't have that information on hand. Let me connect you with a team member who can help.",
            [],
            retrieval_ms,
            0.0,
        )

    context = "\n\n---\n\n".join(c.text for c in chunks)
    messages = [
        {"role": "system", "content": _FAQ_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {utterance}"},
    ]

    t_llm0 = time.perf_counter()
    resp = _client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        max_tokens=128,
        temperature=0.1,
    )
    llm_ms = (time.perf_counter() - t_llm0) * 1000
    answer = resp.choices[0].message.content.strip()
    return answer, chunks, retrieval_ms, llm_ms


def run_small_talk(utterance: str, history: list[dict]) -> tuple[str, float]:
    """Friendly short reply for greetings/farewells."""
    t0 = time.perf_counter()
    messages = [
        {"role": "system", "content": (
            "You are a warm hospital front desk assistant. "
            "Reply naturally to greetings, thanks, and goodbyes in 1–2 sentences. "
            "NEVER mention or invent services, portals, websites, or phone numbers that were not mentioned in this conversation. "
            "Stick to what you can actually do: book, reschedule, cancel appointments, and answer general hospital FAQs. "
            + _HARD_RULES
        )},
    ]
    messages.extend(history[-4:])
    messages.append({"role": "user", "content": utterance})
    resp = _client.chat.completions.create(
        model=config.LLM_MODEL, messages=messages, max_tokens=64, temperature=0.7
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return resp.choices[0].message.content.strip(), elapsed
