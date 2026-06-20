import os
import tempfile
import unittest

os.environ.setdefault("GROQ_API_KEY", "test-key")
_tmp_chroma = tempfile.mkdtemp()
os.environ["CHROMA_PATH"] = _tmp_chroma

from app.services.rag import Chunk, HybridRetriever

SAMPLE_DOCS = [
    Chunk(text="Visiting hours are from 10 AM to 12 PM and 4 PM to 7 PM on all days.", source="visiting_hours.md", chunk_idx=0),
    Chunk(text="ICU visiting is restricted to 11 AM to 12 PM for immediate family only.", source="visiting_hours.md", chunk_idx=1),
    Chunk(text="The hospital accepts Star Health Insurance and HDFC ERGO Health Insurance.", source="insurance.md", chunk_idx=0),
    Chunk(text="Parking is available in a multi-level facility, charges ₹50 for the first two hours.", source="parking.md", chunk_idx=0),
    Chunk(text="Dr. Meera Sharma is an interventional cardiologist specializing in heart failure.", source="doctors.md", chunk_idx=0),
    Chunk(text="Emergency Department is open 24 hours. Call +91-22-4567-8910 for emergencies.", source="emergency.md", chunk_idx=0),
    Chunk(text="The Cardiology Department handles heart disease, ECG, and echocardiography.", source="departments.md", chunk_idx=0),
    Chunk(text="Masks are required in the ICU and isolation rooms. Hand sanitizers at all entry points.", source="health_safety.md", chunk_idx=0),
]


class TestHybridRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = HybridRetriever()
        cls.retriever.index_documents(SAMPLE_DOCS)

    def test_visiting_hours_query(self):
        results = self.retriever.retrieve("What are the visiting hours?", top_k=2)
        texts = [r.text for r in results]
        self.assertTrue(any("visiting" in t.lower() or "10 AM" in t for t in texts))

    def test_insurance_query(self):
        results = self.retriever.retrieve("Do you accept HDFC health insurance?", top_k=2)
        texts = [r.text for r in results]
        self.assertTrue(any("insurance" in t.lower() or "HDFC" in t for t in texts))

    def test_emergency_query(self):
        results = self.retriever.retrieve("What is the emergency contact number?", top_k=2)
        texts = [r.text for r in results]
        self.assertTrue(any("emergency" in t.lower() or "4567-8910" in t for t in texts))

    def test_doctor_query(self):
        results = self.retriever.retrieve("Tell me about the cardiologist Dr. Sharma", top_k=2)
        texts = [r.text for r in results]
        self.assertTrue(any("Sharma" in t or "cardiologist" in t.lower() for t in texts))

    def test_returns_top_k(self):
        results = self.retriever.retrieve("hospital", top_k=3)
        self.assertLessEqual(len(results), 3)

    def test_empty_retriever_returns_empty(self):
        tmp = tempfile.mkdtemp()
        import app.config as cfg
        cfg.CHROMA_PATH = tmp
        empty = HybridRetriever()
        self.assertEqual(empty.retrieve("visiting hours"), [])
        cfg.CHROMA_PATH = _tmp_chroma


if __name__ == "__main__":
    unittest.main()
