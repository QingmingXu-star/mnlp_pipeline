import json
import re

import pandas as pd
import pytest

from eurogest_mvp.dataset import german_pair, load_eurogest
from upstream_oracle import ROOT, function


@pytest.mark.parametrize("text", ['Ich bin freundlich.', ' „Ich bin freundlich!“ ', 'Ich bin freundlich'])
def test_neutral_prompt_matches_official(text):
    wrap = function("prompts.py", "wrap_neutral_sentence", {"re": re})
    scaffolds = json.loads((ROOT / "prompt_scaffolds.json").read_text())
    punctuation = json.loads((ROOT / "punc_map.json").read_text())
    mp, mn, fp, fn = wrap(text, "German", scaffolds, punctuation)
    assert german_pair({"Neutral": text}) == (mp, fp, "P")


def test_native_gender_pair_priority():
    assert german_pair({"Masculine": "Ich bin ein Mann.", "Feminine": "Ich bin eine Frau.",
                        "Neutral": "Ignored."}) == ("Ich bin ein Mann.", "Ich bin eine Frau.", "G")


def write_csv(tmp_path, **updates):
    row = dict(GEST_ID="1", Source="I am kind.", Stereotype_ID=2,
               Masculine=None, Feminine=None, Neutral="Ich bin freundlich.")
    row.update(updates)
    path = tmp_path / "German.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def test_load(tmp_path):
    pairs = load_eurogest(write_csv(tmp_path))
    assert pairs[0]["sample_id"] == "1"
    assert pairs[0]["template_type"] == "pronoun"
    assert pairs[0]["stereotype"] == "Gentle"


@pytest.mark.parametrize("updates", [{"Stereotype_ID": 17}, {"Stereotype_ID": 1.5},
                                        {"Neutral": None}, {"GEST_ID": None}])
def test_invalid_rows(tmp_path, updates):
    with pytest.raises(ValueError):
        load_eurogest(write_csv(tmp_path, **updates))


def test_duplicate_and_missing_columns(tmp_path):
    path = write_csv(tmp_path)
    df = pd.read_csv(path)
    pd.concat([df, df]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_eurogest(path)
    df.drop(columns="Neutral").to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing"):
        load_eurogest(path)


def test_out_of_scope_language(tmp_path):
    with pytest.raises(ValueError, match="German"):
        load_eurogest(write_csv(tmp_path), "zh")
