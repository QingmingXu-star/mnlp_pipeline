# EuroGEST MVP: German + German GPT-2

A local evaluation pipeline for **one existing EuroGEST language (German)** and **one Hugging Face causal LM (`dbmdz/german-gpt2`)**. It scores existing sentence pairs without generating text. Chinese/Indonesian translation, COMET, mitigation, comparisons and multilingual grids are outside this MVP.

It reproduces the **default executable scoring behavior** at EuroGEST commit `b54359247d6a0cfab3247cc02be152e809004cbe`. Read [the methodology audit](docs/METHODOLOGY.md): the paper/report and repository differ in scoring scope and sign.

## Quick start for colleagues

This repository includes the source, tests, `German.csv`, its provenance record, and the completed reference results in `outputs/german-gpt2/`. The two reference PDFs and the original project brief are excluded at the owner's request. Upstream code and dataset licensing notices are retained.

From your cloned repository directory, using Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps --no-build-isolation -e .
export HF_HOME="$PWD/.hf_cache"
python -m pytest -q
python scripts/run_single_model.py --dataset German.csv --language de \
  --device cpu --batch-size 4 --output outputs/my-run
RUN_MODEL_TESTS=1 python -m pytest -q
python scripts/verify_run.py --dataset German.csv --run outputs/my-run
```

The evaluation command downloads the pinned model on first use; internet access and sufficient disk space are required. Later runs can use `--local-files-only`. The default test run skips the real-model test until explicitly enabled above. The reference results can be inspected without downloading a model. For Windows, activate with `.venv\Scripts\Activate.ps1` and set `$env:HF_HOME="$PWD\.hf_cache"` in PowerShell; the Python commands are unchanged. The recorded validation environment is macOS ARM64; other platforms have not been independently tested.

See [GitHub handoff notes](docs/GITHUB.md) for file inclusion and publishing instructions.

## Run on this machine

The project, isolated `.venv`, and German model cache have been installed in the requested directory. The user supplied `German.csv` contains 2,829 samples covering all 16 stereotypes.

```bash
cd /Users/qingmingxu/Documents/study/master2-p1/mnlp/pipeline
source .venv/bin/activate
export HF_HOME="$PWD/.hf_cache"
python scripts/run_single_model.py \
  --language de --dataset German.csv --batch-size 4 --device cpu \
  --local-files-only --output outputs/german-gpt2-rerun
```

The completed reference run is in `outputs/german-gpt2/`. Choose a new output directory to rerun; nonempty directories are not overwritten. Each batch size unit is a **pair**, i.e. two sequences. CPU float32 eager attention is the reference configuration. `--device mps` is available on Apple Silicon and `--device cuda` on NVIDIA hardware; only the CPU configuration is validated here. Reduce `--batch-size` if memory is limited. No automatic truncation or silent dropping occurs.

For a quick check, add `--limit 16 --output outputs/smoke`. A limited run is marked in `summary.json`; if any stereotype has fewer than ten samples, global metrics remain `null`, matching the official completeness gate.

## Setup on another machine

Use Python 3.11+ (tested with Python 3.12 on Apple Silicon):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps --no-build-isolation -e .
export HF_HOME="$PWD/.hf_cache"
```

`requirements-lock.txt` records the validated environment; `pyproject.toml` pins direct dependencies. The CLI intentionally fixes the model and its revision. This small German GPT-2 model is chosen for local validation; it provides no evidence about Llama/LeoLM or effects of language adaptation.

Obtain `German.csv` from [the official EuroGEST dataset](https://huggingface.co/datasets/utter-project/EuroGEST) after accepting its access conditions. Save it in the project root. Alternatively, authenticate locally and use the pinned downloader:

```bash
hf auth login
python -m eurogest_mvp.dataset --output German.csv
```

The downloader uses revision `a97bd293c32e85de950db7573e0b8be8b80e83cd` and writes a SHA-256 provenance sidecar. A manually downloaded file is accepted and hashed, but its revision is marked unverified without a matching download sidecar. For this reference run, the manually downloaded file was subsequently verified by matching its Git blob hash and size to public HF metadata at the pinned revision; the evidence is saved in `German.metadata.json`. No translation service is required. Do not put tokens in source code.

Run the evaluation command above without `--local-files-only` on the first run to download the public model. Once cached, offline evaluation is supported.

## Outputs

| File | Contents |
| --- | --- |
| `sentence_scores.csv` | GEST IDs, source, stereotype, exact M/F inputs, condition, token counts, suffix start, summed/averaged suffix likelihoods, whole-sequence diagnostics, masculine preference |
| `stereotype_scores.csv` | All 16 IDs/labels, sample counts, mean preference, inclination, masculine rank |
| `summary.json` | Completeness, baseline, q_f, q_m, g_s, strongest associations, model/data/code provenance, runtime versions, device/dtype and timestamp |

Unqualified `*_log_likelihood` fields refer to the **selected suffix** used by official default scoring. `*_whole_*` fields are separate diagnostics and never feed preference/summary metrics. Token count includes tokenizer-added special tokens; predicted count excludes the first token; scored count is the selected suffix length.

Higher preference means more masculine. `inclination = stereotype mean - baseline`, so positive means more masculine relative to baseline. `g_s > 1` indicates the stereotypical direction; `< 1` indicates the anti-stereotypical direction. This ratio is not a probability or a general social-fairness measure.

## Tests and colleague review

```bash
python -m pytest -q
# Adds comparison of the actual cached model with the original EuroGEST function:
RUN_MODEL_TESTS=1 python -m pytest -q
# Recompute aggregates and compare stratified real-data samples with upstream:
python scripts/verify_run.py --dataset German.csv --run outputs/german-gpt2
```

Basic tests run offline using a tiny random HF causal model as a fixture. The optional integration test uses the actual pinned German GPT-2. Original reference functions and their Apache license are in `vendor/eurogest/`; parity tests execute upstream bodies, not a second hand-written formula. Test fixtures are not presented as research data.

See [REVIEW.md](docs/REVIEW.md) for a review checklist and [VALIDATION.md](docs/VALIDATION.md) for actual run evidence. The dataset copy is included as requested by the project owner; its Apache-2.0 notice is retained in `data/eurogest/LICENSE`. No credentials or local runtime caches are included.

## Layout

```text
src/eurogest_mvp/    data loading, scoring, aggregation and CLI
scripts/            single-model entry point and independent run audit
tests/             offline tests and real-model parity
vendor/eurogest/    unchanged upstream reference files and provenance
docs/              methodological audit and review evidence
German.csv          user-supplied official German data (included for review)
outputs/            separate run directories
```
