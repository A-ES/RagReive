# Deployment Runbook

Start the full stack with `docker-compose up --build`.

The compose file runs Qdrant on port 6333, FastAPI on port 8000, and the Streamlit dashboard on port 3000.

The API container runs `seed.py` before Uvicorn starts. The seed step ingests files from `data/corpus`, writes embeddings to Qdrant, and persists the BM25 index.

If `OPENAI_API_KEY` is missing, the embedder logs the missing variable and falls back to `sentence-transformers/all-MiniLM-L6-v2`.
