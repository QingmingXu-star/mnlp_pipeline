"""Coordinate validated data, a sequence backend, pure scoring, and result export."""

from collections import Counter
import time
from typing import Callable

from .config import EvaluationConfig
from .contracts import EvaluationResult, SequenceBackend
from .dataset import load_eurogest
from .metrics import aggregate
from .outputs import save_results, validate_output
from .provenance import dataset_metadata, run_metadata, sha256
from .scoring import PairScorer

BackendFactory = Callable[[EvaluationConfig], SequenceBackend]
ProgressCallback = Callable[[int, int], None]


def run_evaluation(config: EvaluationConfig, *, backend_factory: BackendFactory | None = None,
                   progress: ProgressCallback | None = None) -> EvaluationResult:
    """Run one model/language. Optional injection allows offline orchestration tests."""
    validate_output(config.output)
    dataset_info = dataset_metadata(config.dataset)
    pairs = load_eurogest(config.dataset, config.language)
    total_available = len(pairs)
    if config.limit is not None:
        pairs = pairs[:config.limit]
    if backend_factory is None:
        from .models import load_backend
        backend_factory = load_backend
    backend = backend_factory(config)
    backend_info = backend.metadata()
    if not isinstance(backend_info.get("model"), str) or not backend_info["model"]:
        raise ValueError("Backend metadata must identify the model being evaluated")
    scorer = PairScorer(backend)
    started = time.monotonic()
    results = []
    for start in range(0, len(pairs), config.batch_size):
        results.extend(scorer.score_pairs(pairs[start:start + config.batch_size]))
        if progress is not None:
            progress(len(results), len(pairs))
    for row in results:
        row["model"] = backend_info["model"]
    stereotypes, summary = aggregate(results)
    if sha256(config.dataset) != dataset_info["dataset_sha256"]:
        raise ValueError("Dataset changed during evaluation; results were not saved")
    summary.update(run_metadata(config, dataset_info, backend_info, time.monotonic() - started))
    summary.update({
        "n_available_samples": total_available,
        "is_full_dataset_run": len(results) == total_available,
        "condition_counts": dict(Counter(p["condition"] for p in pairs)),
    })
    result = EvaluationResult(results, stereotypes, summary)
    save_results(result, config.output)
    return result
