"""Official German gender-pair construction, independent of dataset I/O."""

import re

import pandas as pd


def german_pair(row):
    """Reproduce build_row_prompts / wrap_neutral_sentence for German."""
    if pd.notna(row.get("Masculine")) and pd.notna(row.get("Feminine")):
        return row["Masculine"], row["Feminine"], "G"
    if pd.notna(row.get("Neutral")):
        sentence = re.sub(r'^[^\w\s]+|[^\w\s]+\Z', '', row["Neutral"].strip())
        return f"„{sentence}“, sagte er", f"„{sentence}“, sagte sie", "P"
    raise ValueError("Row has neither a complete gendered pair nor a neutral sentence")


