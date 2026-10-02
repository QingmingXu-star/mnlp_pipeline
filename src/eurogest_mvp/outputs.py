"""Write complete runs without mixing new files with an earlier result directory."""

import json
from pathlib import Path
import tempfile

import pandas as pd

from .contracts import EvaluationResult


def validate_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Output directory is not empty; choose a new directory to preserve earlier runs")


def save_results(result: EvaluationResult, output):
    output = Path(output)
    validate_output(output)
    # Validate strict JSON before creating any output files.
    summary_text = json.dumps(result.summary, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".eurogest-", dir=output.parent) as temporary:
        staging = Path(temporary)
        pd.DataFrame(result.sentence_scores).to_csv(staging / "sentence_scores.csv", index=False)
        pd.DataFrame(result.stereotype_scores).to_csv(staging / "stereotype_scores.csv", index=False)
        (staging / "summary.json").write_text(summary_text, encoding="utf-8")
        validate_output(output)
        if output.exists():
            output.rmdir()  # Only empty directories are removable; never delete existing results.
        staging.rename(output)
