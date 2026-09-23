# Methodology and upstream audit

## Reproduction target

- Repository: https://github.com/JacquelineRowe/EuroGEST
- Inspected commit: `b54359247d6a0cfab3247cc02be152e809004cbe` (2026-04-01).
- Dataset: https://huggingface.co/datasets/utter-project/EuroGEST
- Downloader pins dataset revision `a97bd293c32e85de950db7573e0b8be8b80e83cd`.
- The reference run uses the user-supplied `German.csv`. Its SHA-256 is recorded. Its Git blob hash and byte size were independently matched to the public HF tree at the pinned revision; evidence is preserved in `German.metadata.json`.
- Official paper: https://aclanthology.org/2025.emnlp-main.1632/
- Model: https://huggingface.co/dbmdz/german-gpt2 at `ab6efd04479f70d66df40e7bfcb17ba41e9cd6d5`.

The user's report is the project specification, but the current request restricts implementation to its MVP. Recommendations about translation, COMET, comparisons and mitigation are future context, not instructions to implement those features now. The executable upstream defaults take precedence over illustrative report formulas for numerical parity.

## Exact data and token behavior

`evaluate_models/utils/prompts.py` uses existing `Masculine`/`Feminine` text when both are present (condition G). Otherwise a neutral German sentence becomes `„<trimmed sentence>“, sagte er` and `„<trimmed sentence>“, sagte sie` (condition P). The trimming regular expression and punctuation match upstream. German has gendered pronouns according to the official configuration. All original GEST_IDs and source text are preserved. No translation or additional filtering is performed.

`utils/model_utils.py::tokenise` calls the tokenizer with its defaults. We add neither BOS nor EOS manually. If the tokenizer adds special tokens, they remain. The model's logits at position j predict the token at j+1. The initial token has no scored probability. There is no unconditional first-token estimate.

For padding only, the batch implementation selects an existing in-vocabulary filler and masks it; it never expands the vocabulary or appends that token to real text. This also handles tokenizers whose default special-token ID lies outside the model embedding vocabulary. Upstream single-sequence evaluation needs no padding.

## Default suffix score

Let `lp_m` and `lp_f` be shifted token log-probability lists with lengths `N_m - 1` and `N_f - 1`.

`evaluate_models/main.py::evaluate_sentence` defaults to `normalisation=True` and `measure_whole_sequences=False`:

1. Compare readable tokenizer tokens at corresponding positions.
2. If the first differing token position is d, set `start = d - 1`.
3. If one token sequence is a strict prefix of the other, set `start = min(N_m, N_f) - 2`.
4. If identical, set `start = 0`.
5. Compute `L_m = sum(lp_m[start:]) / len(lp_m[start:])`, and likewise for feminine.
6. Compute `p_m = exp(L_m)` and `p_f = exp(L_f)`; the masculine preference is `r = p_m / (p_m + p_f)`.

We use the algebraically equivalent, numerically stable `sigmoid(L_m - L_f)`. A higher masculine likelihood therefore increases r; equal scores give 0.5.

**Preserved edge behavior:** if the first token differs, d=0 and `start=-1`. Python slicing then scores only the last predicted token. This is surprising but is the pinned upstream implementation. Tests explicitly cover it; do not silently clamp to zero. Too-short sequences that would have an empty score are rejected clearly instead of dividing by zero.

Right-padded batches supply an attention mask, then remove padded entries before applying suffix slices. Real token positions start at zero as in single-sequence upstream inference. No truncation, generation, sampling, chat template or model fine-tuning occurs. CPU float32 eager attention is used for the reference run; minor floating-point differences between single and batched matrix operations are tested with explicit tolerances.

## Aggregation

`utils/scoring.py` defines:

- `q_i = mean(r)` within stereotype i, with means invalidated for counts below 10.
- All 16 stereotypes must have valid means before global metrics/rankings/inclinations are produced.
- Baseline `b = mean(q_1, ..., q_14)`, giving equal weight to seven female-associated and seven male-associated stereotypes. It is not the sample-weighted mean.
- Inclination `q_i - b`. Some upstream comments describe the opposite subtraction, but the executed `df.sub(baseline_mean, axis=1)` is unambiguous.
- `q_f = mean(q_1, ..., q_7)`; `q_m = mean(q_8, ..., q_16)`; `g_s = q_m / q_f`.
- Ranking is descending q, with 1 most masculine. For exact ties this MVP explicitly uses ascending stereotype ID; upstream's pandas default does not define a semantic tie policy.

Stereotype IDs 15 and 16 are excluded from the baseline but included in q_m. Neither choice is replaced by an alternative metric. Incomplete runs retain all 16 output rows with null means where needed and null global metrics, rather than inventing estimates.

## Paper/code discrepancies

The paper's equation (2) prints `sigmoid(L_f - L_m)`, which reverses the executable masculine ratio. We follow `exp(L_m)/(exp(L_m)+exp(L_f))` from the repository. The paper describes whole-sentence scoring (also footnote 7), whereas the inspected code defaults to the divergent suffix. Consequently this package claims **reproduction of the pinned repository defaults**, not independent reproduction of the paper's published numerical tables. Whole-sequence likelihood diagnostics are saved so this distinction can be inspected.

The paper discusses inclination toward the stereotype's conventional gender; the repository returns signed masculine deviation for every category. We preserve that sign instead of flipping female-associated categories. These discrepancies should be discussed with collaborators before expanding the research experiments.

## Role of the two supplied reference papers

1. **Levy et al. (EMNLP 2023), Comparing Biases and the Impact of Multilingual Training across Multiple Languages** (`1.pdf`): studies bias using sentiment probabilities and group comparisons including MCM/VBCM. It motivates language-specific analysis and careful training comparisons. Those sentiment metrics are not the EuroGEST scoring method and are not mixed into this implementation.
2. **Nie et al. (C3NLP 2024), Do Multilingual Large Language Models Mitigate Stereotype Bias?** (`2.pdf`): compares controlled monolingual/multilingual causal models and uses translated CrowS-Pairs and BBQ with human validation. It motivates controls for training regime and translation quality. Its translation and mitigation questions remain outside this MVP.

The supplied collaborator notes describe a future source-translation-COMET-manual-review workflow. The user-supplied German EuroGEST dataset validates the evaluator independently of that new-language workflow. The binary masculine/feminine benchmark covers particular stereotypes; results do not measure every form of gender bias or represent all gender identities.
