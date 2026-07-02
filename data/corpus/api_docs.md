# API Documentation

The Hybrid RAG API exposes `POST /v1/ask`, `GET /v1/documents`, and `POST /v1/ingest`.

`POST /v1/ask` accepts `query`, `dense_only`, `rrf_weight`, and `chunking_strategy`.

`POST /v1/ingest` accepts a multipart file upload and indexes the file with the configured chunking strategy.

Malformed JSON requests return HTTP 422 with structured validation details. Unhandled server errors return HTTP 500 with a request ID.
