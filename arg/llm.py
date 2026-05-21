"""LLM protocol — shared by QueryProcessor, Generator, and CorpusAnalyst.

Three consumers, one tiny surface. Production wires this to Ollama via
:class:`llama_index.llms.ollama.Ollama`; unit tests inject a deterministic
fake so the suite stays offline.

Design
------
The Protocol is intentionally narrower than LlamaIndex's full LLM interface
— Section 9's needs are a single string-in/string-out completion plus an
optional streaming variant. A real Ollama LLM trivially satisfies it via an
adapter at the call-site rather than the protocol picking up Ollama-specific
types here.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol


class LLM(Protocol):
    """Pluggable LLM source."""

    def complete(self, prompt: str) -> str:
        """Return the full completion for ``prompt`` as a single string."""
        ...

    def stream_complete(self, prompt: str) -> Iterator[str]:
        """Yield completion text incrementally as it arrives from the model."""
        ...

    def complete_structured(self, prompt: str, schema: dict[str, Any]) -> str:
        """Like ``complete`` but passes a JSON Schema to constrain the output.

        Ollama enforces the schema via grammar-based sampling so the response
        is guaranteed to be valid JSON matching ``schema``. Callers should
        still wrap ``json.loads`` in a try/except for robustness.
        """
        ...
