"""Command-line adapter and backwards-compatible evaluate_model function."""

import argparse
from pathlib import Path

from . import MODEL_ID, MODEL_REVISION
from .config import EvaluationConfig, ModelConfig
from .pipeline import run_evaluation


def evaluate_model(dataset, output, batch_size=4, device="cpu", limit=None, local_files_only=False,
                   *, model_config=None, language="de", threads=4, seed=42):
    """Preserve the original arguments and summary return value."""
    config = EvaluationConfig(dataset=dataset, output=output, batch_size=batch_size, device=device,
                              limit=limit, local_files_only=local_files_only, language=language,
                              threads=threads, seed=seed, model=model_config or ModelConfig())

    def report(completed, total):
        if completed == min(batch_size, total) or completed == total or completed % (batch_size * 25) == 0:
            print(f"Scored {completed}/{total} pairs", flush=True)

    result = run_evaluation(config, progress=report)
    print(f"Saved {len(result.sentence_scores)} pairs to {output}; complete metrics: {result.summary['metrics_complete']}")
    return result.summary


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate German EuroGEST with a pinned HF causal model")
    parser.add_argument("--dataset", type=Path, default=Path("German.csv"))
    parser.add_argument("--language", choices=["de"], default="de")
    parser.add_argument("--output", type=Path, default=Path("outputs/german-gpt2"))
    parser.add_argument("--batch-size", type=int, default=4, help="Pairs per batch; two sequences per pair")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="cpu")
    parser.add_argument("--limit", type=int, help="First N pairs, for smoke testing only")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", help="Full HF commit SHA; required when changing the default model")
    parser.add_argument("--tokenizer", help="Defaults to the model repository")
    parser.add_argument("--tokenizer-revision", help="Full HF commit SHA for a separate tokenizer")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.model != MODEL_ID and args.model_revision is None:
        parser.error("--model-revision is required when changing --model")
    try:
        model_config = ModelConfig(args.model, args.model_revision or MODEL_REVISION,
                                   args.tokenizer, args.tokenizer_revision)
        evaluate_model(args.dataset, args.output, args.batch_size, args.device, args.limit,
                       args.local_files_only, model_config=model_config, language=args.language,
                       threads=args.threads, seed=args.seed)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
