from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    meta: Dict[str, Any]

