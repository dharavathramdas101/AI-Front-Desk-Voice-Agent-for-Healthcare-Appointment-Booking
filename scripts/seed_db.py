"""
Seed the database with 5 mock doctors, 3 departments, and ~20 available slots.
Run once before starting the server: python scripts/seed_db.py
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from db import Doctor, Slot, create_tables, get_session

DOCTORS = [
    {
        "name": "Dr. Meera Sharma",
        "department": "Cardiology",
        "bio": "Interventional cardiologist with 15 years of experience in heart failure and coronary artery disease.",
    },
    {
        "name": "Dr. Arjun Nair",
        "department": "Cardiology",
        "bio": "Electrophysiologist specializing in arrhythmia management and cardiac ablation procedures.",
    },
    {
        "name": "Dr. Priya Desai",
        "department": "Orthopedics",
        "bio": "Joint replacement and sports medicine specialist with expertise in knee and hip surgeries.",
    },
    {
        "name": "Dr. Rahul Mehta",
        "department": "Orthopedics",
        "bio": "Spine surgeon focusing on minimally invasive techniques for back pain and disc disorders.",
    },
    {
        "name": "Dr. Sunita Iyer",
        "department": "General Medicine",
        "bio": "General physician with broad expertise in preventive care, diabetes management, and chronic disease.",
    },
]

# Generate slots starting from tomorrow, Mon–Fri, 9am–4pm in 1-hour blocks
def generate_slots(doctor_id: int, days_out: int = 14) -> list[dict]:
    slots = []
    base = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
    day = base
    count = 0
    while count < 20 and (day - base).days < days_out:
        if day.weekday() < 5:  # Mon–Fri
            for hour in range(9, 17):
                slots.append({"doctor_id": doctor_id, "slot_datetime": day.replace(hour=hour), "is_available": True})
                count += 1
                if count >= 20:
                    break
        day += timedelta(days=1)
    return slots


def seed() -> None:
    create_tables()
    session = get_session()

    if session.query(Doctor).count() > 0:
        print("DB already seeded. Delete ai_front_desk.db to re-seed.")
        session.close()
        return

    # Insert doctors
    doctor_objs = []
    for d in DOCTORS:
        doc = Doctor(**d)
        session.add(doc)
        doctor_objs.append(doc)
    session.flush()  # assign IDs

    # Insert slots — 4 slots per doctor (compact, not 20 each)
    slot_count = 0
    for doc in doctor_objs:
        base = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
        # Skip to next weekday
        while base.weekday() >= 5:
            base += timedelta(days=1)

        hours = [9, 11, 14, 16]
        day_offset = 0
        for i, hour in enumerate(hours * 14):  # 56 slots per doctor across 14 days
            day = base + timedelta(days=(i // 4))
            while day.weekday() >= 5:
                day += timedelta(days=1)
            slot = Slot(
                doctor_id=doc.id,
                slot_datetime=day.replace(hour=hour),
                is_available=True,
            )
            session.add(slot)
            slot_count += 1

    session.commit()
    session.close()
    print(f"Seeded {len(DOCTORS)} doctors and {slot_count} slots.")


if __name__ == "__main__":
    seed()
