.PHONY: run seed test lint clean

run:
	uvicorn app.main:app --reload --port 8000

seed:
	python scripts/seed_db.py
	python scripts/seed_rag.py

test:
	python -m pytest tests/ -v

lint:
	python -m ruff check app/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -f ai_front_desk.db
	rm -rf chroma_db/
