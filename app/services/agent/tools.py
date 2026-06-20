from __future__ import annotations

import json
from datetime import datetime, timedelta

from app.config import CANCELLATION_WINDOW_HOURS
from app.models.db import Appointment, Doctor, Slot, get_session


# ── Tool schemas (OpenAI-compatible for Groq) ─────────────────────────────────

TOOLS = [
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
                    "doctor_name": {"type": "string"},
                    "slot_id": {"type": "integer"},
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
                    "new_slot_id": {"type": "integer"},
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
                "properties": {"appointment_id": {"type": "integer"}},
                "required": ["appointment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_patient_history",
            "description": "Look up a patient's appointment history by phone number.",
            "parameters": {
                "type": "object",
                "properties": {"phone": {"type": "string"}},
                "required": ["phone"],
            },
        },
    },
]


# ── Tool implementations ──────────────────────────────────────────────────────

def check_availability(department: str | None, date: str) -> dict:
    try:
        target = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {"error": f"Invalid date format: {date}. Use YYYY-MM-DD."}

    session = get_session()
    try:
        def _query(start, end):
            q = (
                session.query(Slot, Doctor)
                .join(Doctor, Slot.doctor_id == Doctor.id)
                .filter(Slot.slot_datetime >= start, Slot.slot_datetime < end, Slot.is_available == True)
            )
            if department:
                q = q.filter(Doctor.department.ilike(f"%{department}%"))
            return q.order_by(Slot.slot_datetime).all()

        def _fmt(slot, doctor):
            return {
                "slot_id": slot.id,
                "doctor": doctor.name,
                "department": doctor.department,
                "date": slot.slot_datetime.strftime("%A, %d %B %Y"),
                "time": slot.slot_datetime.strftime("%I:%M %p"),
                "datetime_key": slot.slot_datetime.strftime("%Y-%m-%d %H:%M"),
            }

        exact = _query(target, target + timedelta(days=1))
        upcoming = _query(target, target + timedelta(days=14))

        return {
            "requested_date": target.strftime("%A, %d %B %Y"),
            "slots_on_requested_date": [_fmt(s, d) for s, d in exact],
            "upcoming_slots": [_fmt(s, d) for s, d in upcoming[:12]],
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
            return {"error": f"Doctor mismatch: slot belongs to {doctor.name if doctor else 'unknown'}."}

        slot.is_available = False
        appt = Appointment(
            patient_name=patient_name, phone=phone,
            doctor_id=slot.doctor_id, slot_id=slot_id, status="booked",
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
            hours_until = (slot.slot_datetime - datetime.now()).total_seconds() / 3600
            if 0 < hours_until < CANCELLATION_WINDOW_HOURS:
                return {
                    "policy_violation": True,
                    "hours_until_appointment": round(hours_until, 1),
                    "message": (
                        f"Appointment is in {round(hours_until, 1)} hours. "
                        f"Policy requires {CANCELLATION_WINDOW_HOURS}h notice. "
                        "Please suggest the patient reschedule instead."
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
            .filter(Appointment.phone == phone, Appointment.status != "cancelled")
            .order_by(Appointment.created_at.desc())
            .limit(3)
            .all()
        )
        if not rows:
            return {"found": False, "message": "No previous appointments found for this phone number."}
        return {
            "found": True,
            "patient_name": rows[0][0].patient_name,
            "history": [
                {
                    "appointment_id": appt.id,
                    "doctor": doctor.name,
                    "department": doctor.department,
                    "datetime": slot.slot_datetime.strftime("%A, %d %B %Y at %I:%M %p"),
                    "status": appt.status,
                }
                for appt, slot, doctor in rows
            ],
        }
    finally:
        session.close()


TOOL_MAP = {
    "check_availability": lambda a: check_availability(a.get("department"), a["date"]),
    "book_appointment": lambda a: book_appointment(a["patient_name"], a["phone"], a["doctor_name"], a["slot_id"]),
    "reschedule_appointment": lambda a: reschedule_appointment(a["appointment_id"], a["new_slot_id"]),
    "cancel_appointment": lambda a: cancel_appointment(a["appointment_id"]),
    "lookup_patient_history": lambda a: lookup_patient_history(a["phone"]),
}
