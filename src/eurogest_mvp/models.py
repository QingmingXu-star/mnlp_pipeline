"""The only module that loads Hugging Face models or performs Torch inference."""

import importlib.metadata
import math
import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import EvaluationConfig
from .contracts import SequenceLogProbs


class HuggingFaceBackend:
    def __init__(self, model, tokenizer, device="cpu", identity=None):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.identity = dict(identity or {"model": "injected", "model_revision": None,
                                          "tokenizer": "injected", "tokenizer_revision": None})

    def metadata(self):
        return {
            **self.identity, "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype).removeprefix("torch."),
            "attention_implementation": self.model.config._attn_implementation,
            "tokenizer_class": type(self.tokenizer).__name__,
            "bos_token_id": self.tokenizer.bos_token_id, "eos_token_id": self.tokenizer.eos_token_id,
            "torch_version": torch.__version__,
            "transformers_version": importlib.metadata.version("transformers"),
        }

    def sequence_log_probs(self, texts):
        # Same tokenizer defaults as upstream: no manual BOS/EOS, no truncation.
        ids = [self.tokenizer(t)["input_ids"] for t in texts]
        limit = getattr(self.model.config, "max_position_embeddings", None)
        for sequence in ids:
            if len(sequence) < 2:
                raise ValueError("At least two tokens are needed for causal scoring")
            if limit and len(sequence) > limit:
                raise ValueError(f"Sequence exceeds model context ({len(sequence)} > {limit}); no truncation")
        if not ids:
            return []
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None or not 0 <= pad < self.model.get_input_embeddings().num_embeddings:
            pad = 0  # Masked padding only; never add a vocabulary item.
        width = max(map(len, ids))
        input_ids = torch.full((len(ids), width), pad, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros_like(input_ids)
        for i, sequence in enumerate(ids):
            input_ids[i, :len(sequence)] = torch.tensor(sequence, device=self.device)
            attention_mask[i, :len(sequence)] = 1
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            shifted = torch.log_softmax(logits[:, :-1, :], dim=-1)
            lp = shifted.gather(-1, input_ids[:, 1:, None]).squeeze(-1)
        result = []
        for i, sequence in enumerate(ids):
            # Strip padding before suffix slicing (especially for start=-1).
            values = lp[i, :len(sequence) - 1].cpu().tolist()
            if not all(math.isfinite(x) for x in values):
                raise ValueError("Model returned nonfinite token log probabilities")
            result.append(SequenceLogProbs(self.tokenizer.convert_ids_to_tokens(sequence), values))
        return result


def load_backend(config: EvaluationConfig):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    device = config.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    settings = config.model
    tokenizer = AutoTokenizer.from_pretrained(
        settings.resolved_tokenizer_id, revision=settings.resolved_tokenizer_revision,
        local_files_only=config.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_id, revision=settings.revision, local_files_only=config.local_files_only,
        torch_dtype=torch.float32, attn_implementation="eager", use_safetensors=True,
    )
    return HuggingFaceBackend(model, tokenizer, device, {
        "model": settings.model_id, "model_revision": settings.revision,
        "tokenizer": settings.resolved_tokenizer_id,
        "tokenizer_revision": settings.resolved_tokenizer_revision,
    })
