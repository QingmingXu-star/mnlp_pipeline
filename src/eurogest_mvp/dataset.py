"""Load the official German CSV; never translate or re-filter curated data."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import pandas as pd

from . import DATASET_ID, DATASET_REVISION, LABELS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def german_pair(row):
    """Reproduce build_row_prompts / wrap_neutral_sentence for German."""
    if pd.notna(row.get("Masculine")) and pd.notna(row.get("Feminine")):
        return row["Masculine"], row["Feminine"], "G"
    if pd.notna(row.get("Neutral")):
        sentence = re.sub(r'^[^\w\s]+|[^\w\s]+\Z', '', row["Neutral"].strip())
        return f"„{sentence}“, sagte er", f"„{sentence}“, sagte sie", "P"
    raise ValueError("Row has neither a complete gendered pair nor a neutral sentence")


def load_eurogest(path, language="de"):
    if language != "de":
        raise ValueError("This MVP supports German (de) only")
    df = pd.read_csv(path, dtype={"GEST_ID": "string"})
    required = {"GEST_ID", "Source", "Stereotype_ID", "Masculine", "Feminine", "Neutral"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing official EuroGEST columns: {sorted(missing)}")
    if df.empty or df["GEST_ID"].isna().any() or df["GEST_ID"].duplicated().any():
        raise ValueError("Dataset must contain nonempty, unique GEST_ID values")
    pairs = []
    for row in df.to_dict("records"):
        sid = row["Stereotype_ID"]
        if pd.isna(sid) or float(sid) != int(sid) or int(sid) not in LABELS:
            raise ValueError(f"Invalid stereotype ID: {sid}")
        m, f, condition = german_pair(row)
        if not all(isinstance(s, str) and s.strip() for s in (m, f)):
            raise ValueError(f"Invalid text for GEST_ID {row['GEST_ID']}")
        pairs.append({
            "sample_id": str(row["GEST_ID"]), "language": language,
            "stereotype_id": int(sid), "stereotype": LABELS[int(sid)],
            "source_sentence": row["Source"], "masculine_sentence": m,
            "feminine_sentence": f, "condition": condition,
            "template_type": "native_gender_marking" if condition == "G" else "pronoun",
            "original_split": "German",
        })
    return pairs


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
