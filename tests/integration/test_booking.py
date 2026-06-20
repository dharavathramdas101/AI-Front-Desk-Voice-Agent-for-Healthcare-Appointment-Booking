import os
import unittest
from datetime import datetime, timedelta

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from app.models.db import Base, Doctor, Slot, Appointment, create_tables, engine, SessionLocal
from app.services.agent.tools import (
    book_appointment, cancel_appointment,
    check_availability, reschedule_appointment,
)


def _seed():
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    doc = Doctor(name="Dr. Test Physician", department="General Medicine", bio="Test doctor")
    session.add(doc)
    session.flush()
    tomorrow = (datetime.now() + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0)
    slot1 = Slot(doctor_id=doc.id, slot_datetime=tomorrow, is_available=True)
    slot2 = Slot(doctor_id=doc.id, slot_datetime=tomorrow.replace(hour=14), is_available=True)
    session.add_all([slot1, slot2])
    session.commit()
    ids = doc.id, slot1.id, slot2.id
    session.close()
    return ids


class TestCheckAvailability(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(bind=engine)
        self.doc_id, self.slot1_id, self.slot2_id = _seed()

    def test_returns_available_slots(self):
        date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        result = check_availability(department="General Medicine", date=date)
        self.assertIn("slots_on_requested_date", result)
        self.assertGreater(len(result["slots_on_requested_date"]), 0)
        self.assertIn("slot_id", result["slots_on_requested_date"][0])

    def test_no_slots_wrong_date(self):
        result = check_availability(department="General Medicine", date="2000-01-01")
        self.assertEqual(result["slots_on_requested_date"], [])

    def test_invalid_date_format(self):
        result = check_availability(department=None, date="next Monday")
        self.assertIn("error", result)


class TestBookAppointment(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(bind=engine)
        self.doc_id, self.slot1_id, self.slot2_id = _seed()

    def test_successful_booking(self):
        result = book_appointment("Alice Kumar", "9876543210", "Dr. Test Physician", self.slot1_id)
        self.assertIn("appointment_id", result)
        self.assertEqual(result["status"], "booked")

    def test_double_booking_rejected(self):
        book_appointment("Alice Kumar", "9876543210", "Dr. Test Physician", self.slot1_id)
        result = book_appointment("Bob Singh", "1234567890", "Dr. Test Physician", self.slot1_id)
        self.assertIn("error", result)

    def test_wrong_doctor_name_rejected(self):
        result = book_appointment("Alice Kumar", "9876543210", "Dr. Fake Name", self.slot1_id)
        self.assertIn("error", result)

    def test_invalid_slot_id(self):
        result = book_appointment("Alice Kumar", "9876543210", "Dr. Test Physician", 99999)
        self.assertIn("error", result)


class TestRescheduleAppointment(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(bind=engine)
        self.doc_id, self.slot1_id, self.slot2_id = _seed()
        self.appt_id = book_appointment("Alice Kumar", "9876543210", "Dr. Test Physician", self.slot1_id)["appointment_id"]

    def test_successful_reschedule(self):
        result = reschedule_appointment(self.appt_id, self.slot2_id)
        self.assertEqual(result["status"], "rescheduled")

    def test_reschedule_nonexistent(self):
        result = reschedule_appointment(99999, self.slot2_id)
        self.assertIn("error", result)

    def test_reschedule_to_unavailable_slot(self):
        book_appointment("Bob Singh", "1111111111", "Dr. Test Physician", self.slot2_id)
        result = reschedule_appointment(self.appt_id, self.slot2_id)
        self.assertIn("error", result)


class TestCancelAppointment(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(bind=engine)
        self.doc_id, self.slot1_id, self.slot2_id = _seed()
        self.appt_id = book_appointment("Alice Kumar", "9876543210", "Dr. Test Physician", self.slot1_id)["appointment_id"]

    def test_successful_cancel(self):
        result = cancel_appointment(self.appt_id)
        self.assertEqual(result["status"], "cancelled")

    def test_cancel_already_cancelled(self):
        cancel_appointment(self.appt_id)
        result = cancel_appointment(self.appt_id)
        self.assertIn("error", result)

    def test_cancel_nonexistent(self):
        result = cancel_appointment(99999)
        self.assertIn("error", result)

    def test_slot_freed_after_cancel(self):
        cancel_appointment(self.appt_id)
        result = book_appointment("Charlie Dev", "2222222222", "Dr. Test Physician", self.slot1_id)
        self.assertIn("appointment_id", result)


if __name__ == "__main__":
    unittest.main()
