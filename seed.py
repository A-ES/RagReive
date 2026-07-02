"""Seed the sample corpus into Qdrant and BM25."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from config import PipelineConfig
from ingestion.pipeline import run_ingestion

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


async def main() -> None:
    config = PipelineConfig()
    corpus_dir = Path("data/corpus")
    files = sorted(p for p in corpus_dir.iterdir() if p.suffix.lower() in {".md", ".txt", ".html", ".pdf"})
    result = await run_ingestion(files, config)
    logging.info("Seed complete: %s", result.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
