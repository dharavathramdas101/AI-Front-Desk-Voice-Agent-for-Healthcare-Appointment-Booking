from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker,
)

from app.config import DATABASE_URL


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


class Doctor(Base):
    __tablename__ = "doctors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    department: Mapped[str] = mapped_column(String(64))
    bio: Mapped[str] = mapped_column(Text, default="")

    slots: Mapped[list["Slot"]] = relationship(back_populates="doctor")
    appointments: Mapped[list["Appointment"]] = relationship(back_populates="doctor")


class Slot(Base):
    __tablename__ = "slots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doctor_id: Mapped[int] = mapped_column(ForeignKey("doctors.id"))
    slot_datetime: Mapped[datetime] = mapped_column(DateTime)
    is_available: Mapped[bool] = mapped_column(Boolean, default=True)

    doctor: Mapped["Doctor"] = relationship(back_populates="slots")

    __table_args__ = (Index("ix_slots_doctor_dt", "doctor_id", "slot_datetime"),)


class Appointment(Base):
    __tablename__ = "appointments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_name: Mapped[str] = mapped_column(String(128))
    phone: Mapped[str] = mapped_column(String(20))
    doctor_id: Mapped[int] = mapped_column(ForeignKey("doctors.id"))
    slot_id: Mapped[int] = mapped_column(ForeignKey("slots.id"))
    status: Mapped[str] = mapped_column(String(20), default="booked")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    doctor: Mapped["Doctor"] = relationship(back_populates="appointments")
    slot: Mapped["Slot"] = relationship()


class CallLog(Base):
    __tablename__ = "call_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    intent: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    transcript_snippet: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    outcome: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latency_stt_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    latency_retrieval_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    latency_llm_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    latency_tts_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    latency_total_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (Index("ix_call_logs_session", "session_id"),)


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    return SessionLocal()
