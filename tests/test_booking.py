"""
Unit tests for booking tool functions (DB layer).
Uses an in-memory SQLite DB — no real DB file needed.
"""

import sys, os
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from db import Base, Doctor, Slot, Appointment, create_tables, engine, SessionLocal
from agent import (
    book_appointment,
    cancel_appointment,
    check_availability,
    reschedule_appointment,
)


def _setup_test_db():
    """Create schema and seed one doctor + two slots."""
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
    doc_id, slot1_id, slot2_id = doc.id, slot1.id, slot2.id
    session.close()
    return doc_id, slot1_id, slot2_id


class TestCheckAvailability(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(bind=engine)
        self.doc_id, self.slot1_id, self.slot2_id = _setup_test_db()

    def test_returns_available_slots(self):
        date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        results = check_availability(department="General Medicine", date=date)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIn("slot_id", results[0])

    def test_no_slots_wrong_date(self):
        results = check_availability(department="General Medicine", date="2000-01-01")
        self.assertEqual(results, [])

    def test_invalid_date_format(self):
        results = check_availability(department=None, date="next Monday")
        self.assertTrue(any("error" in r for r in results))


class TestBookAppointment(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(bind=engine)
        self.doc_id, self.slot1_id, self.slot2_id = _setup_test_db()

    def test_successful_booking(self):
        result = book_appointment("Alice Kumar", "9876543210", "Dr. Test Physician", self.slot1_id)
        self.assertIn("appointment_id", result)
        self.assertEqual(result["status"], "booked")

    def test_double_booking_rejected(self):
        book_appointment("Alice Kumar", "9876543210", "Dr. Test Physician", self.slot1_id)
        result2 = book_appointment("Bob Singh", "1234567890", "Dr. Test Physician", self.slot1_id)
        self.assertIn("error", result2)
        self.assertIn("no longer available", result2["error"])

    def test_wrong_doctor_name_rejected(self):
        result = book_appointment("Alice Kumar", "9876543210", "Dr. Fake Name", self.slot1_id)
        self.assertIn("error", result)

    def test_invalid_slot_id(self):
        result = book_appointment("Alice Kumar", "9876543210", "Dr. Test Physician", 99999)
        self.assertIn("error", result)


class TestRescheduleAppointment(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(bind=engine)
        self.doc_id, self.slot1_id, self.slot2_id = _setup_test_db()
        result = book_appointment("Alice Kumar", "9876543210", "Dr. Test Physician", self.slot1_id)
        self.appt_id = result["appointment_id"]

    def test_successful_reschedule(self):
        result = reschedule_appointment(self.appt_id, self.slot2_id)
        self.assertEqual(result["status"], "rescheduled")

    def test_reschedule_nonexistent_appointment(self):
        result = reschedule_appointment(99999, self.slot2_id)
        self.assertIn("error", result)

    def test_reschedule_to_unavailable_slot(self):
        # Book slot2 to make it unavailable
        book_appointment("Bob Singh", "1111111111", "Dr. Test Physician", self.slot2_id)
        result = reschedule_appointment(self.appt_id, self.slot2_id)
        self.assertIn("error", result)


class TestCancelAppointment(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(bind=engine)
        self.doc_id, self.slot1_id, self.slot2_id = _setup_test_db()
        result = book_appointment("Alice Kumar", "9876543210", "Dr. Test Physician", self.slot1_id)
        self.appt_id = result["appointment_id"]

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
        """Cancelling an appointment frees the slot for rebooking."""
        cancel_appointment(self.appt_id)
        result = book_appointment("Charlie Dev", "2222222222", "Dr. Test Physician", self.slot1_id)
        self.assertIn("appointment_id", result)


if __name__ == "__main__":
    unittest.main()
