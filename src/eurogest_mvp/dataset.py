"""Read and validate official German EuroGEST rows."""

import pandas as pd

from . import LABELS
from .provenance import sha256  # Backwards-compatible import location.
from .templates import german_pair  # Backwards-compatible import location.


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
    """Preserve python -m eurogest_mvp.dataset for existing users."""
    from .download import main as download_main
    download_main()


if __name__ == "__main__":
    main()
