import math

import pandas as pd
import pytest

from eurogest_mvp import LABELS
from eurogest_mvp.metrics import aggregate
from upstream_oracle import module


def test_matches_upstream_aggregation_baseline_and_gs(tmp_path):
    # Uneven category sizes catch accidental sample-weighted baselines.
    rows = [{"stereotype_id": sid, "gender_preference_score": .1 + sid / 25 + j / 1000}
            for sid in range(1, 17) for j in range(10 + sid)]
    official = pd.DataFrame({
        "Stereotype_ID": [r["stereotype_id"] for r in rows],
        "masc_prob": [r["gender_preference_score"] * .01 for r in rows],
        "fem_prob": [(1 - r["gender_preference_score"]) * .01 for r in rows],
    })
    official.to_csv(tmp_path / "german.csv", index=False)
    upstream = module("scoring")
    reference = upstream.load_and_aggregate_results(tmp_path, ["German"])
    inc = upstream.calculate_inclinations(reference, {str(k): v for k, v in LABELS.items()})
    gs = upstream.calculate_gs_scores(reference).loc["German", "g_s_score"]
    aggregates, summary = aggregate(rows)
    for row in aggregates:
        assert row["mean_preference"] == pytest.approx(reference.loc[row["stereotype_id"], "German"])
        assert row["inclination"] == pytest.approx(inc.loc[row["stereotype"], "German"])
        assert row["rank"] == 17 - row["stereotype_id"]
    assert summary["g_s_score"] == pytest.approx(gs)
    assert summary["default_masculine_preference"] == pytest.approx(reference.loc[1:14, "German"].mean())


def test_under_ten_and_incomplete_suppress_global_metrics(tmp_path):
    rows = [{"stereotype_id": sid, "gender_preference_score": .5}
            for sid in range(1, 17) for _ in range(9 if sid == 4 else 10)]
    aggregates, summary = aggregate(rows)
    assert aggregates[3]["mean_preference"] is None
    assert not summary["metrics_complete"]
    assert summary["stereotypes_below_minimum_10"] == [4]
    assert summary["g_s_score"] is None
    assert summary["default_masculine_preference"] is None
    assert all(r["rank"] is None and r["inclination"] is None for r in aggregates)


def test_neutral_and_tie_break():
    rows = [{"stereotype_id": sid, "gender_preference_score": .5}
            for sid in range(1, 17) for _ in range(10)]
    aggregates, summary = aggregate(rows)
    assert summary["g_s_score"] == 1
    assert summary["default_masculine_preference"] == .5
    assert [r["rank"] for r in aggregates] == list(range(1, 17))
    assert all(r["inclination"] == 0 for r in aggregates)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1, 2])
def test_invalid_score(score):
    with pytest.raises(ValueError):
        aggregate([{"stereotype_id": 1, "gender_preference_score": score}])
