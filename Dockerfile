FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bootstrap a catalog so /health is ready even before a scrape runs.
RUN python scripts/seed_catalog.py

# Pre-download the embedding model at build time so the first /chat is fast
# and stays within the 30s budget (cold-start friendly).
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')" || \
    echo "embedding model prefetch skipped (offline build)"

ENV LLM_PROVIDER=groq
ENV MAX_TURNS=8
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
