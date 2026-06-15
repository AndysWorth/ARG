"""Shared fake embedder and LLM for unit tests."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator

_VEC_DIM = 32


class _TagEmbedder:
    """Deterministic embedder: sets dimensions based on QUERY_<TAG> markers."""

    _BASE = ord("A")

    def embed(self, text: str) -> list[float]:
        vec = [0.001] * _VEC_DIM
        for m in re.finditer(r"QUERY_([A-Z])", text):
            idx = (ord(m.group(1)) - self._BASE) % _VEC_DIM
            vec[idx] += 1.0
        vec[abs(hash(text)) % _VEC_DIM] += 0.05
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class _ScriptedLLM:
    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = "ANSWER FROM LLM",
    ) -> None:
        self.responses = responses or {}
        self.default = default
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        for trigger, response in self.responses.items():
            if trigger in prompt:
                return response
        return self.default

    def complete_structured(self, prompt: str, schema: dict) -> str:
        return self.complete(prompt)

    def stream_complete(self, prompt: str) -> Iterator[str]:
        yield from self.complete(prompt)
