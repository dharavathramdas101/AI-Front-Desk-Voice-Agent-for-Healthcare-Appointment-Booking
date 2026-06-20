from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class SlotOut(BaseModel):
    slot_id: int
    doctor: str
    department: str
    date: str
    time: str
    datetime_key: str


class AppointmentOut(BaseModel):
    appointment_id: int
    patient: str
    doctor: str
    datetime: str
    status: str


class CallLogOut(BaseModel):
    session_id: str
    intent: Optional[str]
    transcript_snippet: Optional[str]
    outcome: Optional[str]
    latency_total_ms: Optional[float]
    timestamp: datetime
