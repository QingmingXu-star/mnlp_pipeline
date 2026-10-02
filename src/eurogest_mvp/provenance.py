"""Dataset/source fingerprints and reproducibility metadata, separate from inference."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

from . import DATASET_ID, UPSTREAM_COMMIT


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_metadata(path):
    path = Path(path)
    fingerprint = sha256(path)
    sidecar = path.with_suffix(".metadata.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    if not isinstance(provenance, dict):
        raise ValueError("Dataset provenance sidecar must contain a JSON object")
    if sidecar.exists() and provenance.get("sha256") != fingerprint:
        raise ValueError("Dataset hash differs from its provenance sidecar")
    if provenance.get("dataset_id", DATASET_ID) != DATASET_ID:
        raise ValueError("Dataset provenance sidecar identifies a different dataset")
    return {
        "dataset_id": DATASET_ID,
        "dataset_revision": provenance.get("dataset_revision"),
        "dataset_revision_note": "From hash-matched provenance sidecar" if provenance else
                                 "Local CSV supplied without download sidecar; revision unverified",
        "dataset_path": str(path.resolve()), "dataset_sha256": fingerprint,
        "dataset_provenance": provenance,
    }


def source_hashes(source_root=None):
    root = Path(source_root) if source_root is not None else Path(__file__).parent
    return {p.name: sha256(p) for p in sorted(root.glob("*.py"))}


def run_metadata(config, dataset_info, backend_info, elapsed):
    """Record backend-reported identity instead of claiming the default model was used."""
    return {
        **dataset_info, **backend_info,
        "language": config.language, "scoring_mode": "upstream_default_divergent_suffix",
        "normalisation": True, "upstream_commit": UPSTREAM_COMMIT,
        "source_sha256": source_hashes(), "batch_size_pairs": config.batch_size,
        "seed": config.seed, "truncation": False,
        "special_token_policy": "tokenizer defaults; no manually prepended BOS or appended EOS",
        "python_version": platform.python_version(), "platform": platform.platform(),
        "timestamp": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": elapsed,
        "configuration": config.to_dict(), "output_schema_version": 1,
    }
