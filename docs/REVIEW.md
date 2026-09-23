# Colleague review checklist

1. Read `METHODOLOGY.md`, especially the paper/code sign discrepancy and the default suffix scope. Confirm the intended research target is the pinned repository behavior.
2. Inspect `vendor/eurogest/PROVENANCE.json`; files are unchanged copies of the listed upstream commit under Apache-2.0. Compare `evaluate_sentence`, prompt construction and aggregation with the implementation.
3. Run basic tests and `RUN_MODEL_TESTS=1 python -m pytest -q` after caching the model. Check that reference tests execute upstream bodies and only omit text generation after probability computation.
4. Verify the user-supplied German.csv checksum and run the full command from the README. Check all available rows were scored, no duplicates were introduced, and all 16 categories meet the upstream minimum of 10.
5. Inspect both native gender-marked (G) and neutral/pronoun (P) examples. Confirm punctuation and suffix-start alignment. Inspect start=-1 cases separately.
6. Verify `r = sigmoid(L_m - L_f)` from sentence output; recompute each category mean, baseline over IDs 1-14, and `mean(q_8..q_16)/mean(q_1..q_7)`.
7. Check model/data revisions, hashes, source-code fingerprints and runtime versions in `summary.json`. The user-supplied CSV was matched to the pinned HF revision through public Git blob metadata. Check that evidence in `German.metadata.json`. Other manual downloads remain unverified unless similarly checked.
8. Treat this model run as pipeline validation. It provides no evidence of an adaptation or mitigation effect, because there is only one model and one language.

No communication, sharing or publishing to colleagues is performed automatically. Share the source and permitted results yourself. The owner requested inclusion of the downloaded German dataset; its attribution and license are retained.
