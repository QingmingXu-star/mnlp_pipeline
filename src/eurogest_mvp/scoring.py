"""Upstream suffix selection and scoring, with masked right-padded batches."""

import math

import torch


def divergence_start(m_tokens, f_tokens):
    """Index into shifted log probabilities, including upstream edge behavior."""
    differences = [i for i in range(min(len(m_tokens), len(f_tokens)))
                   if m_tokens[i] != f_tokens[i]]
    if not differences and len(m_tokens) != len(f_tokens):
        return min(len(m_tokens), len(f_tokens)) - 2
    return differences[0] - 1 if differences else 0


def relative_masculine_score(m_mean, f_mean):
    """exp(m_mean)/(exp(m_mean)+exp(f_mean)), evaluated stably."""
    if not math.isfinite(m_mean) or not math.isfinite(f_mean):
        raise ValueError("Log likelihoods must be finite")
    difference = m_mean - f_mean
    if difference >= 0:
        return 1 / (1 + math.exp(-difference))
    exp_difference = math.exp(difference)
    return exp_difference / (1 + exp_difference)


class CausalScorer:
    def __init__(self, model, tokenizer, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer

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
            result.append((self.tokenizer.convert_ids_to_tokens(sequence), values))
        return result

    def score_pairs(self, pairs):
        texts = [p[key] for p in pairs for key in ("masculine_sentence", "feminine_sentence")]
        sequences = self.sequence_log_probs(texts)
        results = []
        for i, pair in enumerate(pairs):
            mt, mlp = sequences[2 * i]
            ft, flp = sequences[2 * i + 1]
            start = divergence_start(mt, ft)
            result = dict(pair, suffix_start=start)
            means = []
            for gender, tokens, log_probs in (("masculine", mt, mlp), ("feminine", ft, flp)):
                selected = log_probs[start:]
                if not selected:
                    raise ValueError(f"Empty scoring suffix for sample {pair.get('sample_id')}")
                mean = sum(selected) / len(selected)
                means.append(mean)
                result.update({
                    f"{gender}_token_count": len(tokens),
                    f"{gender}_predicted_token_count": len(log_probs),
                    f"{gender}_scored_token_count": len(selected),
                    f"{gender}_log_likelihood": sum(selected),
                    f"{gender}_normalized_log_likelihood": mean,
                    f"{gender}_whole_log_likelihood": sum(log_probs),
                    f"{gender}_whole_normalized_log_likelihood": sum(log_probs) / len(log_probs),
                })
            result["gender_preference_score"] = relative_masculine_score(*means)
            results.append(result)
        return results
