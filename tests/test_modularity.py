"""Behavioral tests for replaceable stages, compatibility, and fail-fast validation."""

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from eurogest_mvp.config import EvaluationConfig, ModelConfig
from eurogest_mvp.contracts import EvaluationResult, SequenceLogProbs
from eurogest_mvp.evaluate import main
from eurogest_mvp.outputs import save_results
from eurogest_mvp.pipeline import run_evaluation
from eurogest_mvp.scoring import PairScorer, score_pair


class StubBackend:
    """A deterministic test backend; never represented as a downloaded model."""

    def __init__(self):
        self.batch_sizes = []

    def sequence_log_probs(self, texts):
        self.batch_sizes.append(len(texts))
        return [SequenceLogProbs(["context", "m" if text.endswith("er") else "f"],
                                 [-1.0 if text.endswith("er") else -3.0]) for text in texts]

    def metadata(self):
        return {"model": "test-backend", "model_revision": "test-fixture", "device": "none"}


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "German.csv"
    pd.DataFrame([dict(GEST_ID=i, Source="I am kind.", Stereotype_ID=2,
                       Neutral="Ich bin freundlich.", Masculine=None, Feminine=None)
                  for i in range(5)]).to_csv(path, index=False)
    return path


def test_pipeline_uses_injected_backend_and_keeps_output_contract(dataset, tmp_path):
    backend = StubBackend()
    progress = []
    config = EvaluationConfig(dataset, tmp_path / "run", batch_size=2, limit=3)
    result = run_evaluation(config, backend_factory=lambda _: backend,
                            progress=lambda done, total: progress.append((done, total)))
    assert backend.batch_sizes == [4, 2]
    assert progress == [(2, 3), (3, 3)]
    assert result.summary["model"] == "test-backend"
    assert result.summary["n_samples"] == 3
    assert result.summary["n_available_samples"] == 5
    assert result.summary["is_full_dataset_run"] is False
    assert result.summary["g_s_score"] is None
    assert result.summary["configuration"]["limit"] == 3
    saved = pd.read_csv(config.output / "sentence_scores.csv")
    assert saved["model"].tolist() == ["test-backend"] * 3
    assert len(pd.read_csv(config.output / "stereotype_scores.csv")) == 16
    assert json.loads((config.output / "summary.json").read_text())["n_samples"] == 3


def forbidden_backend(_):
    pytest.fail("Invalid input must fail before a backend is initialized")


def test_bad_provenance_fails_before_model_loading(dataset, tmp_path):
    dataset.with_suffix(".metadata.json").write_text(json.dumps({"sha256": "incorrect"}))
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="hash differs"):
        run_evaluation(EvaluationConfig(dataset, output), backend_factory=forbidden_backend)
    assert not output.exists()


def test_existing_results_preserved_before_model_loading(dataset, tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    sentinel = output / "summary.json"
    sentinel.write_text("prior run")
    with pytest.raises(ValueError, match="not empty"):
        run_evaluation(EvaluationConfig(dataset, output), backend_factory=forbidden_backend)
    assert sentinel.read_text() == "prior run"


def test_dataset_changed_during_inference_is_not_published(dataset, tmp_path):
    class ChangingBackend(StubBackend):
        def sequence_log_probs(self, texts):
            dataset.write_text(dataset.read_text() + "\n")
            return super().sequence_log_probs(texts)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="changed during"):
        run_evaluation(EvaluationConfig(dataset, output), backend_factory=lambda _: ChangingBackend())
    assert not output.exists()


def test_failed_serialization_leaves_no_partial_outputs(tmp_path):
    output = tmp_path / "run"
    with pytest.raises(ValueError):
        save_results(EvaluationResult([], [], {"invalid": float("nan")}), output)
    assert not output.exists()


def test_csv_failure_leaves_no_partial_outputs(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("Simulated disk write failure")
    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    output = tmp_path / "run"
    with pytest.raises(OSError, match="write failure"):
        save_results(EvaluationResult([], [], {}), output)
    assert not output.exists()
    assert not list(tmp_path.glob(".eurogest-*"))


def test_writer_accepts_empty_output_directory(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    save_results(EvaluationResult([{"sample_id": "1"}], [{"stereotype_id": 1}], {"n_samples": 1}), output)
    assert {f.name for f in output.iterdir()} == {"sentence_scores.csv", "stereotype_scores.csv", "summary.json"}


@pytest.mark.parametrize("changes", [
    {"batch_size": 0}, {"threads": -1}, {"limit": 0}, {"seed": -1},
    {"language": "zh"}, {"device": "unknown"}, {"batch_size": True},
])
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        EvaluationConfig(**changes)


def test_separate_tokenizer_requires_independent_pinned_revision():
    with pytest.raises(ValueError, match="own pinned revision"):
        ModelConfig(tokenizer_id="different/tokenizer")
    config = ModelConfig(tokenizer_id="different/tokenizer", tokenizer_revision="a" * 40)
    assert config.resolved_tokenizer_id == "different/tokenizer"
    assert config.resolved_tokenizer_revision == "a" * 40


def test_cli_rejects_unpinned_model_before_loading(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--model", "other/model"])
    assert exc.value.code == 2
    assert "--model-revision is required" in capsys.readouterr().err


def test_cli_imports_and_pure_scoring_do_not_load_torch():
    code = "import eurogest_mvp.evaluate, eurogest_mvp.scoring; import sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    import os
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_pure_pair_scoring_and_backend_contract():
    pair = {"masculine_sentence": "x", "feminine_sentence": "y"}
    m, f = SequenceLogProbs(["same", "m"], [-1.0]), SequenceLogProbs(["same", "f"], [-3.0])
    assert score_pair(pair, m, f)["gender_preference_score"] == pytest.approx(0.8807970779778823)
    with pytest.raises(ValueError, match="N-1"):
        score_pair(pair, SequenceLogProbs(["one"], []), f)
    class BadBackend(StubBackend):
        def sequence_log_probs(self, texts):
            return []
    with pytest.raises(ValueError, match="exactly one"):
        PairScorer(BadBackend()).score_pairs([pair])


def test_programmatic_model_replacement_requires_revision():
    with pytest.raises(ValueError, match="own pinned revision"):
        ModelConfig(model_id="different/model")
    config = ModelConfig(model_id="different/model", revision="b" * 40)
    assert config.resolved_tokenizer_id == "different/model"
    assert config.resolved_tokenizer_revision == "b" * 40
