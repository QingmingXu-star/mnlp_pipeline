"""Optional authorized data download; separate from local evaluation."""

import argparse
import json
import shutil
from pathlib import Path

from . import DATASET_ID, DATASET_REVISION
from .dataset import load_eurogest
from .provenance import sha256


def main():
    parser = argparse.ArgumentParser(description="Download the authorized, pinned German EuroGEST CSV")
    parser.add_argument("--output", type=Path, default=Path("data/eurogest/German.csv"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Destination already exists; choose a new output path")
    from huggingface_hub import hf_hub_download
    try:
        source = hf_hub_download(DATASET_ID, "German.csv", repo_type="dataset", revision=DATASET_REVISION)
    except Exception as exc:
        raise SystemExit("Download failed. Accept access conditions at https://huggingface.co/datasets/"
                         "utter-project/EuroGEST and configure HF_TOKEN locally. "
                         f"Underlying error type: {type(exc).__name__}") from exc
    pairs = load_eurogest(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, args.output)
    args.output.with_suffix(".metadata.json").write_text(json.dumps({
        "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
        "file": "German.csv", "sha256": sha256(args.output), "n_samples": len(pairs),
    }, indent=2) + "\n")
    print(f"Downloaded {len(pairs)} German pairs to {args.output}")


if __name__ == "__main__":
    main()
