import json
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("GROQ_API_KEY", "test-key")


class TestClassifyIntent(unittest.TestCase):

    def _mock_response(self, intent: str):
        choice = MagicMock()
        choice.message.content = json.dumps({"intent": intent, "reasoning": "test"})
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def _run(self, utterance: str, expected: str):
        with patch("app.services.agent.intent._client") as mock_client:
            mock_client.chat.completions.create.return_value = self._mock_response(expected)
            from app.services.agent.intent import classify_intent
            intent, elapsed = classify_intent(utterance, history=[])
            self.assertEqual(intent, expected)
            self.assertGreaterEqual(elapsed, 0)

    def test_book_appointment(self):
        self._run("I'd like to book an appointment with Dr. Priya Desai", "BOOK")

    def test_book_appointment_2(self):
        self._run("Can I schedule a visit to see a cardiologist next Monday?", "BOOK")

    def test_reschedule(self):
        self._run("I need to change my appointment to Thursday", "RESCHEDULE")

    def test_cancel(self):
        self._run("Please cancel my appointment, I can't make it", "CANCEL")

    def test_faq_visiting_hours(self):
        self._run("What are the visiting hours for the ICU?", "FAQ")

    def test_faq_insurance(self):
        self._run("Do you accept HDFC health insurance?", "FAQ")

    def test_small_talk_greeting(self):
        self._run("Hello, good morning!", "SMALL_TALK")

    def test_small_talk_thanks(self):
        self._run("Thank you, bye!", "SMALL_TALK")

    def test_unclear(self):
        self._run("Um, I don't know, maybe?", "UNCLEAR")

    def test_fallback_on_bad_json(self):
        with patch("app.services.agent.intent._client") as mock_client:
            choice = MagicMock()
            choice.message.content = "sorry cannot classify"
            resp = MagicMock()
            resp.choices = [choice]
            mock_client.chat.completions.create.return_value = resp
            from app.services.agent.intent import classify_intent
            intent, _ = classify_intent("asdfqwer", history=[])
            self.assertEqual(intent, "UNCLEAR")


if __name__ == "__main__":
    unittest.main()
