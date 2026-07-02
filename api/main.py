"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from api.middleware import register_middleware
from api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Hybrid RAG Pipeline",
        description="Dense + BM25 retrieval, RRF fusion, reranking, grounded generation, and citation verification.",
        version="0.1.0",
    )
    register_middleware(app)
    app.include_router(router)
    return app


app = create_app()
