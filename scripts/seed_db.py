"""
Seed the database with 5 doctors and 280 slots across 14 days.
Run once before starting the server: python scripts/seed_db.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta
from app.models.db import Doctor, Slot, create_tables, get_session

DOCTORS = [
    {"name": "Dr. Meera Sharma",  "department": "Cardiology",       "bio": "Interventional cardiologist with 15 years of experience in heart failure and coronary artery disease."},
    {"name": "Dr. Arjun Nair",    "department": "Cardiology",       "bio": "Electrophysiologist specializing in arrhythmia management and cardiac ablation procedures."},
    {"name": "Dr. Priya Desai",   "department": "Orthopedics",      "bio": "Joint replacement and sports medicine specialist with expertise in knee and hip surgeries."},
    {"name": "Dr. Rahul Mehta",   "department": "Orthopedics",      "bio": "Spine surgeon focusing on minimally invasive techniques for back pain and disc disorders."},
    {"name": "Dr. Sunita Iyer",   "department": "General Medicine", "bio": "General physician with broad expertise in preventive care, diabetes management, and chronic disease."},
]


def seed() -> None:
    create_tables()
    session = get_session()

    if session.query(Doctor).count() > 0:
        print("DB already seeded. Delete ai_front_desk.db to re-seed.")
        session.close()
        return

    doctor_objs = []
    for d in DOCTORS:
        doc = Doctor(**d)
        session.add(doc)
        doctor_objs.append(doc)
    session.flush()

    slot_count = 0
    for doc in doctor_objs:
        base = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
        while base.weekday() >= 5:
            base += timedelta(days=1)

        hours = [9, 11, 14, 16]
        for i, hour in enumerate(hours * 14):
            day = base + timedelta(days=(i // 4))
            while day.weekday() >= 5:
                day += timedelta(days=1)
            session.add(Slot(doctor_id=doc.id, slot_datetime=day.replace(hour=hour), is_available=True))
            slot_count += 1

    session.commit()
    session.close()
    print(f"Seeded {len(DOCTORS)} doctors and {slot_count} slots.")


if __name__ == "__main__":
    seed()
