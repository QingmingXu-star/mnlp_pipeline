"""Validated immutable settings; no model or file I/O during construction."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re

from . import MODEL_ID, MODEL_REVISION


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = MODEL_ID
    revision: str | None = None
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None

    def __post_init__(self):
        if self.revision is None:
            if self.model_id != MODEL_ID:
                raise ValueError("A different model requires its own pinned revision")
            object.__setattr__(self, "revision", MODEL_REVISION)
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if not re.fullmatch(r"[a-fA-F0-9]{40}", self.revision):
            raise ValueError("Model revision must be a full immutable HF commit SHA")
        if self.tokenizer_id is not None and not self.tokenizer_id.strip():
            raise ValueError("tokenizer_id must not be empty")
        if self.tokenizer_id not in (None, self.model_id) and self.tokenizer_revision is None:
            raise ValueError("A separate tokenizer requires its own pinned revision")
        if self.tokenizer_revision is not None and not re.fullmatch(r"[a-fA-F0-9]{40}", self.tokenizer_revision):
            raise ValueError("Tokenizer revision must be a full immutable HF commit SHA")

    @property
    def resolved_tokenizer_id(self):
        return self.tokenizer_id or self.model_id

    @property
    def resolved_tokenizer_revision(self):
        return self.tokenizer_revision or self.revision


@dataclass(frozen=True)
class EvaluationConfig:
    dataset: Path = Path("German.csv")
    output: Path = Path("outputs/german-gpt2")
    language: str = "de"
    batch_size: int = 4
    device: str = "cpu"
    limit: int | None = None
    local_files_only: bool = False
    threads: int = 4
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self):
        object.__setattr__(self, "dataset", Path(self.dataset))
        object.__setattr__(self, "output", Path(self.output))
        if self.language != "de":
            raise ValueError("This MVP supports German (de) only")
        for name in ("batch_size", "threads"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.limit is not None and (type(self.limit) is not int or self.limit < 1):
            raise ValueError("limit must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if self.device not in {"cpu", "mps", "cuda", "auto"}:
            raise ValueError("device must be cpu, mps, cuda, or auto")

    def to_dict(self):
        result = asdict(self)
        result["dataset"] = str(self.dataset)
        result["output"] = str(self.output)
        return result
