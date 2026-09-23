# Validation record

Validated locally on 2026-09-23. The MVP is **German + dbmdz/german-gpt2**. No English alternative, new-language translation, COMET, mitigation or model comparison is part of the delivered implementation.

## Actual environment and run

- Installed project: `/Users/qingmingxu/Documents/study/master2-p1/mnlp/pipeline`.
- Isolated interpreter: `.venv/bin/python`, Python 3.12.14.
- macOS ARM64; CPU, float32, eager attention; four pairs per batch; four Torch threads.
- Torch 2.9.1; Transformers 4.57.3. Full dependency versions are in `requirements-lock.txt`.
- Model/tokenizer revision: `ab6efd04479f70d66df40e7bfcb17ba41e9cd6d5`.
- Upstream code commit: `b54359247d6a0cfab3247cc02be152e809004cbe`.
- User-downloaded `German.csv`: 401,950 bytes; SHA-256 `d76cde823e9921e576b401aa3905c2f7f0bb548f177daabd99b29eb2526f8e66`.
- Dataset revision verified as `a97bd293c32e85de950db7573e0b8be8b80e83cd`: the local Git blob SHA-1 `28a87fea6c625e1be0a78acbfad7de71cad3089c` and byte size match the public HF tree entry for `German.csv`. This verification requires only public file metadata, not access to gated contents. Evidence is in `German.metadata.json` and `summary.json`.

Executed:

```bash
HF_HOME="$PWD/.hf_cache" .venv/bin/python scripts/run_single_model.py \
  --dataset German.csv --language de --device cpu --batch-size 4 \
  --local-files-only --output outputs/german-gpt2
HF_HOME="$PWD/.hf_cache" RUN_MODEL_TESTS=1 .venv/bin/python -m pytest -q -p no:cacheprovider
HF_HOME="$PWD/.hf_cache" .venv/bin/python scripts/verify_run.py \
  --dataset German.csv --run outputs/german-gpt2
```

## Results and coverage

- **2,829 / 2,829 real dataset pairs scored**, no rows dropped or truncated.
- **2,747 pronoun-template (P)** pairs and **82 native gender-marked (G)** pairs.
- All 16 stereotypes present; counts range from 155 to 212, exceeding the official threshold of 10.
- `sentence_scores.csv`: 2,829 rows, including full and selected-suffix likelihoods and token counts.
- `stereotype_scores.csv`: 16 rows, with all means, inclinations and ranks populated.
- `summary.json`: complete metrics and reproducibility metadata.
- Scoring elapsed time: approximately 43 seconds, excluding setup and model loading.
- No production pair has the upstream first-token-difference (`start=-1`) edge case; its behavior is nevertheless preserved and unit-tested.

| Metric | Value |
| --- | ---: |
| Default masculine baseline (equal-weight IDs 1-14) | 0.6563373373 |
| Female-stereotype-group masculine mean q_f | 0.6136169742 |
| Male-stereotype-group masculine mean q_m | 0.6798511116 |
| Overall stereotype rate g_s = q_m / q_f | 1.1079405234 |
| Highest masculine preference category | Leaders (0.7406152163) |
| Lowest masculine preference category | Sexual (0.5165110374) |

The lowest category is feminine-leaning **relative to the 0.6563 baseline**, not an absolute preference for the feminine variant. In particular, its mean is still above 0.5. These are results for this small model under the repository's default suffix metric, not evidence about language adaptation or mitigation.

## Tests and independent verification

**35 tests passed**, including the optional real-model integration test; none skipped in the recorded run. Coverage includes:

- Preference direction and stable neutral/large-negative-log cases.
- Exact suffix rules, unequal token lengths, strict-prefix pairs and first-token divergence.
- Identical input sentences, batch/single equivalence, padding, BOS/EOS defaults, invalid lengths.
- Invalid dataset schema, IDs, duplicate IDs, labels, and absent text.
- Official neutral-prompt construction and native-pair priority.
- Unequal stereotype sizes, baseline weighting, g_s, inclination, threshold/completeness gate and rank ties.
- Real cached German GPT-2 parity with original upstream scoring.

The additional `verify_run.py` audit checks **every dataset row** against the original upstream prompt builder, complete output coverage, IDs/source/labels, likelihood normalization, all preference formulas, all aggregate metrics, ranks and counts, and source/vendor hashes.

It also reruns the actual model with the upstream `evaluate_sentence` on **31 real pairs**, selecting one per observed stereotype/condition combination. The upstream text-generation step is stubbed only after likelihood computation; its original numerical function body is unchanged.

- Maximum absolute preference error: **7.7668949916e-06** (tolerance 2e-05).
- Maximum absolute mean log-likelihood error: **3.9577484131e-05** (tolerance 1e-04).
- All aggregate comparisons pass at **1e-12** absolute tolerance.

These small differences arise between batched and individual float32 inference; mathematical scoring semantics match. `validation.json` lists the exact sample IDs and checks.

## Remaining interpretation limits

The repository defaults differ from the paper's written equation and whole-sentence description; see `METHODOLOGY.md`. No claim is made that this run reproduces the paper's published model tables. CPU is validated; MPS/CUDA execution is supported by the interface but was not separately validated. Broader models, languages, translations and mitigation are intentionally unimplemented.
