"""Audit saved outputs against all input rows and the pinned upstream functions."""

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eurogest_mvp import LABELS, MODEL_ID, MODEL_REVISION
from eurogest_mvp.dataset import sha256

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from upstream_oracle import ROOT as VENDOR, evaluate_sentence, function, module


def verify(dataset, run):
    raw = pd.read_csv(dataset, dtype={"GEST_ID": str})
    scores = pd.read_csv(run / "sentence_scores.csv", dtype={"sample_id": str})
    aggregates = pd.read_csv(run / "stereotype_scores.csv")
    summary = json.loads((run / "summary.json").read_text())
    assert summary["dataset_sha256"] == sha256(dataset)
    assert summary["model"] == MODEL_ID and summary["model_revision"] == MODEL_REVISION
    assert summary["n_samples"] == len(raw) == len(scores)
    assert summary["is_full_dataset_run"] and summary["metrics_complete"]
    assert scores["sample_id"].tolist() == raw["GEST_ID"].tolist()
    assert scores["sample_id"].is_unique and aggregates["stereotype_id"].tolist() == list(range(1, 17))
    assert scores["stereotype_id"].tolist() == raw["Stereotype_ID"].astype(int).tolist()
    assert scores["source_sentence"].tolist() == raw["Source"].tolist()
    for name, expected in summary["source_sha256"].items():
        assert sha256(ROOT / "src/eurogest_mvp" / name) == expected
    manifest = json.loads((VENDOR / "PROVENANCE.json").read_text())
    for name, expected in manifest["files"].items():
        assert sha256(VENDOR / name) == expected

    import re
    namespace = {"re": re}
    function("prompts.py", "wrap_neutral_sentence", namespace)
    build = function("prompts.py", "build_row_prompts", namespace)
    scaffolds = json.loads((VENDOR / "prompt_scaffolds.json").read_text())
    punctuation = json.loads((VENDOR / "punc_map.json").read_text())
    config = json.loads((VENDOR / "language_test_configs.json").read_text())
    for original, saved in zip(raw.to_dict("records"), scores.to_dict("records")):
        gendered = pd.notna(original["Masculine"]) and pd.notna(original["Feminine"])
        with contextlib.redirect_stdout(io.StringIO()):
            prompts, condition, _, _ = build(original, gendered, "German", scaffolds, punctuation, config)
        assert prompts["baseline"] == (saved["masculine_sentence"], saved["feminine_sentence"])
        assert condition == saved["condition"]
        for gender in ("masculine", "feminine"):
            n = saved[f"{gender}_scored_token_count"]
            assert n > 0
            assert saved[f"{gender}_predicted_token_count"] == saved[f"{gender}_token_count"] - 1
            assert math.isclose(saved[f"{gender}_log_likelihood"] / n,
                                saved[f"{gender}_normalized_log_likelihood"], abs_tol=1e-12)

    m = np.exp(scores["masculine_normalized_log_likelihood"])
    f = np.exp(scores["feminine_normalized_log_likelihood"])
    assert np.allclose(scores["gender_preference_score"], m / (m + f), atol=1e-12, rtol=0)
    upstream = module("scoring")
    with tempfile.TemporaryDirectory() as directory:
        pd.DataFrame({"Stereotype_ID": scores["stereotype_id"], "masc_prob": m, "fem_prob": f}).to_csv(
            Path(directory) / "german.csv", index=False)
        reference = upstream.load_and_aggregate_results(directory, ["German"])
    inclinations = upstream.calculate_inclinations(reference, {str(k): v for k, v in LABELS.items()})
    assert np.allclose(aggregates["mean_preference"], reference["German"], atol=1e-12, rtol=0)
    assert np.allclose(aggregates["inclination"], inclinations["German"], atol=1e-12, rtol=0)
    official_gs = upstream.calculate_gs_scores(reference).loc["German", "g_s_score"]
    assert math.isclose(summary["g_s_score"], official_gs, abs_tol=1e-12)
    assert math.isclose(summary["default_masculine_preference"], reference.loc[1:14, "German"].mean(), abs_tol=1e-12)
    assert aggregates["n_samples"].tolist() == scores.groupby("stereotype_id").size().tolist()
    ranked = reference["German"].sort_values(ascending=False).index.tolist()
    assert aggregates.sort_values("rank")["stereotype_id"].tolist() == ranked

    # At least one per stereotype and condition, including native gender cases.
    chosen = scores.groupby(["stereotype_id", "condition"], sort=True).head(1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                               local_files_only=True, attn_implementation="eager").eval()
    oracle = evaluate_sentence()
    errors, log_errors = [], []
    for saved in chosen.to_dict("records"):
        reference_row = oracle((saved["masculine_sentence"], saved["feminine_sentence"]),
                               model, tokenizer, torch.device("cpu"), True, False)
        expected = reference_row["masc_prob"] / (reference_row["masc_prob"] + reference_row["fem_prob"])
        errors.append(abs(saved["gender_preference_score"] - expected))
        for gender, key in (("masculine", "masc_prob"), ("feminine", "fem_prob")):
            log_errors.append(abs(saved[f"{gender}_normalized_log_likelihood"] - math.log(reference_row[key])))
    assert max(errors) < 2e-5, max(errors)
    assert max(log_errors) < 1e-4, max(log_errors)
    result = {
        "passed": True, "all_rows_checked": len(scores), "stereotypes_checked": len(aggregates),
        "real_model_upstream_pairs_checked": len(chosen),
        "parity_sample_ids": chosen["sample_id"].tolist(),
        "max_absolute_preference_error": max(errors), "preference_tolerance": 2e-5,
        "max_absolute_mean_log_likelihood_error": max(log_errors), "log_likelihood_tolerance": 1e-4,
        "aggregate_tolerance": 1e-12, "dataset_sha256": sha256(dataset),
        "checks": ["input/output coverage", "all official prompts", "source and vendor hashes",
                   "normalization and preference", "official aggregate/baseline/inclination/g_s",
                   "ranking and counts", "stratified real-model upstream parity"],
    }
    (run / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("German.csv"))
    parser.add_argument("--run", type=Path, default=Path("outputs/german-gpt2"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    verify(args.dataset, args.run)
