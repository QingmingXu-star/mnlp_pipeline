"""Single-language, single-model evaluation CLI."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import DATASET_ID, MODEL_ID, MODEL_REVISION, UPSTREAM_COMMIT
from .dataset import load_eurogest, sha256
from .metrics import aggregate
from .scoring import CausalScorer


def evaluate_model(dataset, output, batch_size=4, device="cpu", limit=None, local_files_only=False):
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("batch_size and limit must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new directory to preserve earlier runs")
    pairs = load_eurogest(dataset)
    total_available = len(pairs)
    if limit is not None:
        pairs = pairs[:limit]
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                             local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=local_files_only,
        torch_dtype=torch.float32, attn_implementation="eager", use_safetensors=True,
    )
    scorer = CausalScorer(model, tokenizer, device)
    started = time.monotonic()
    results = []
    for start in range(0, len(pairs), batch_size):
        results.extend(scorer.score_pairs(pairs[start:start + batch_size]))
        if start == 0 or start % (batch_size * 25) == 0:
            print(f"Scored {len(results)}/{len(pairs)} pairs", flush=True)
    for row in results:
        row["model"] = MODEL_ID
    stereotypes, summary = aggregate(results)
    fingerprint = sha256(dataset)
    sidecar = Path(dataset).with_suffix(".metadata.json")
    provenance = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    if provenance and provenance.get("sha256") != fingerprint:
        raise ValueError("Dataset hash differs from its provenance sidecar")
    source_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    summary.update({
        "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "tokenizer": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
        "language": "de", "dataset_id": DATASET_ID,
        "dataset_revision": provenance.get("dataset_revision"),
        "dataset_revision_note": "From hash-matched download sidecar" if provenance else
                                 "Local CSV supplied without download sidecar; revision unverified",
        "dataset_path": str(Path(dataset).resolve()), "dataset_sha256": fingerprint,
        "n_available_samples": total_available, "is_full_dataset_run": len(results) == total_available,
        "condition_counts": dict(Counter(p["condition"] for p in pairs)),
        "scoring_mode": "upstream_default_divergent_suffix", "normalisation": True,
        "upstream_commit": UPSTREAM_COMMIT, "source_sha256": source_hashes,
        "device": device, "dtype": "float32", "batch_size_pairs": batch_size,
        "seed": 42, "attention_implementation": "eager", "truncation": False,
        "special_token_policy": "tokenizer defaults; no manually prepended BOS or appended EOS",
        "tokenizer_class": type(tokenizer).__name__, "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "torch_version": torch.__version__, "transformers_version": importlib.metadata.version("transformers"),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
    })
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output / "sentence_scores.csv", index=False)
    pd.DataFrame(stereotypes).to_csv(output / "stereotype_scores.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(results)} pairs to {output}; complete metrics: {summary['metrics_complete']}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate German EuroGEST with pinned dbmdz/german-gpt2")
    parser.add_argument("--dataset", type=Path, default=Path("German.csv"))
    parser.add_argument("--language", choices=["de"], default="de")
    parser.add_argument("--output", type=Path, default=Path("outputs/german-gpt2"))
    parser.add_argument("--batch-size", type=int, default=4, help="Pairs per batch; two sequences per pair")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="cpu")
    parser.add_argument("--limit", type=int, help="First N pairs, for smoke testing only")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        evaluate_model(args.dataset, args.output, args.batch_size, args.device, args.limit, args.local_files_only)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
