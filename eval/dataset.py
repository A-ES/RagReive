"""Evaluation dataset loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class QAPair(BaseModel):
    id: str
    question: str
    expected_answer: str
    category: Literal["factual", "multi_hop", "no_answer", "ambiguous"]


def load_dataset(path: Path = Path("data/eval_dataset.json")) -> list[QAPair]:
    with path.open("r", encoding="utf-8") as fh:
        return [QAPair(**row) for row in json.load(fh)]
