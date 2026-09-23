# GitHub handoff

## Included

- `src/`, `scripts/`, `tests/`: implementation, CLI, tests, and independent result audit.
- `German.csv`, `German.metadata.json`: owner-provided dataset and verified revision/checksum.
- `outputs/german-gpt2/`: sentence scores, stereotype scores, summary, and validation evidence.
- `README.md`, `docs/`, `pyproject.toml`, `requirements-lock.txt`, `.gitignore`: setup, methodology, reproducibility, and review instructions.
- `vendor/eurogest/` and `data/eurogest/LICENSE`: upstream source/configuration, attribution, provenance, and licensing notices.

## Excluded

Exactly these reference documents are excluded at the owner's request: `1.pdf`, `2.pdf`, and `Gender Bias Evaluation Pipeline — Research & Implementation Brief.md`. The PDFs remain referenced as background in the methodology; they are not required to run the code.

Machine-specific `.venv/`, `.hf_cache/`, Python bytecode, and package build metadata are also excluded. Recreate the environment and download the model using the README commands. Existing reviewed results and German.csv are intentionally included, unlike the initial development ignore rules.

## Upload

Unzip the supplied GitHub package into a local clone of the intended repository, keeping the source files and the folder structure. If the repository already has content, review overlaps before replacing files. Do not upload the zip as a substitute for the browsable project files.

For a new, empty repository, after configuring your Git identity and authentication:

```bash
git init -b main
git add .
git status --short
# Confirm the three excluded documents and runtime caches are absent.
git commit -m "Add German EuroGEST evaluation MVP and validated reference results"
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

The repository URL and visibility are chosen by the owner. No remote upload is implied by preparing this package. For an existing repository, use its existing branch/workflow instead of blindly reinitializing or force-pushing.

Start review at `docs/METHODOLOGY.md`, then `docs/VALIDATION.md` and `docs/REVIEW.md`. The completed run covers 2,829 pairs and all 16 stereotypes. Read the paper/code discrepancy notes before interpreting the metric scientifically.
