"""Small, framework-independent interfaces between pipeline stages."""

from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol


class SequenceLogProbs(NamedTuple):
    """One unpadded token sequence and N-1 next-token log probabilities."""

    tokens: list[str]
    log_probs: list[float]


class SequenceBackend(Protocol):
    """A backend returns one ordered sequence result per input text."""

    def sequence_log_probs(self, texts: list[str]) -> list[SequenceLogProbs]: ...

    def metadata(self) -> dict[str, Any]: ...


@dataclass
class EvaluationResult:
    sentence_scores: list[dict[str, Any]]
    stereotype_scores: list[dict[str, Any]]
    summary: dict[str, Any]
