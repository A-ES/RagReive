# Troubleshooting

If Qdrant is unavailable, verify that the `qdrant` container is healthy and that `QDRANT_HOST` points to `qdrant` inside Docker.

If the dashboard cannot reach the API, confirm that the API service is listening on port 8000.

If reranking is slow on CPU, switch `RERANKER_MODE` to `llm_judge` only when an OpenAI key is configured, or rely on retrieval scores for a local demo.
