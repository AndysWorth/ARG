"""ARG generator package — query processing + LLM answer generation."""

from arg.generator.generator import ARGResult, Generator, SourceRef
from arg.generator.query_processor import ProcessedQuery, QueryProcessor

__all__ = [
    "ARGResult",
    "Generator",
    "ProcessedQuery",
    "QueryProcessor",
    "SourceRef",
]
