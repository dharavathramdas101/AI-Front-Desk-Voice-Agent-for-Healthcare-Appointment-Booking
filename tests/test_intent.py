"""
Unit tests for intent classification.
Mocks the Groq client — no API calls made.
"""

import json
import sys, os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

# Patch env before importing agent
os.environ.setdefault("GROQ_API_KEY", "test-key")


class TestClassifyIntent(unittest.TestCase):
    """Verify classify_intent correctly parses Groq JSON responses."""

    def _make_mock_response(self, intent: str):
        content = json.dumps({"intent": intent, "reasoning": "test"})
        choice = MagicMock()
        choice.message.content = content
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def _run(self, utterance: str, expected_intent: str):
        with patch("groq.Groq") as MockGroq:
            client = MagicMock()
            MockGroq.return_value = client
            client.chat.completions.create.return_value = self._make_mock_response(expected_intent)

            # Import after patching
            import importlib
            import agent as ag
            ag._client = client

            intent, elapsed = ag.classify_intent(utterance, history=[])
            self.assertEqual(intent, expected_intent, f"Utterance: '{utterance}'")
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

    def test_unclear_fallback_on_bad_json(self):
        """If LLM returns non-JSON, intent falls back to UNCLEAR."""
        with patch("groq.Groq") as MockGroq:
            client = MagicMock()
            MockGroq.return_value = client
            choice = MagicMock()
            choice.message.content = "sorry, I cannot classify this"
            resp = MagicMock()
            resp.choices = [choice]
            client.chat.completions.create.return_value = resp

            import agent as ag
            ag._client = client
            intent, _ = ag.classify_intent("asdfqwer", history=[])
            self.assertEqual(intent, "UNCLEAR")


if __name__ == "__main__":
    unittest.main()
