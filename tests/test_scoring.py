import math

import pytest
import torch
from tokenizers.processors import TemplateProcessing

from eurogest_mvp.scoring import CausalScorer, divergence_start, relative_masculine_score
from upstream_oracle import evaluate_sentence


def pair(m, f):
    return {"masculine_sentence": m, "feminine_sentence": f}


def test_preference_direction_and_stability():
    assert relative_masculine_score(-1, -4) > .95
    assert relative_masculine_score(-4, -1) < .05
    assert relative_masculine_score(-10000, -10000) == .5
    assert relative_masculine_score(-10000, -10001) == pytest.approx(1 / (1 + math.exp(-1)))
    with pytest.raises(ValueError):
        relative_masculine_score(float("nan"), -1)


@pytest.mark.parametrize("m,f,expected", [
    ([1,2,3], [1,4,3], 0), ([1,2,3], [1,2,4], 1),
    ([1,2,3], [1,2,3,4], 1), ([1,2], [1,2], 0),
    ([1,2,3], [4,2,3], -1),
])
def test_exact_suffix_edges(m, f, expected):
    assert divergence_start(m, f) == expected


def test_identical_sentence(tiny_components):
    result = CausalScorer(*tiny_components).score_pairs([pair("I am kind", "I am kind")])[0]
    assert result["gender_preference_score"] == .5


def test_batch_single_and_padding(tiny_components):
    scorer = CausalScorer(*tiny_components)
    pairs = [pair("I he", "I she"), pair("I am very kind he said .", "I am kind she said .")]
    batch = scorer.score_pairs(pairs)
    single = [scorer.score_pairs([p])[0] for p in pairs]
    for b, s in zip(batch, single):
        for key in b:
            if isinstance(b[key], float):
                assert b[key] == pytest.approx(s[key], abs=2e-6)
    assert batch[0]["masculine_predicted_token_count"] == 1
    # Changing the masked pad ID must not affect real token scores.
    scorer.tokenizer.pad_token = "[EOS]"
    changed = scorer.score_pairs(pairs)
    assert changed[0]["gender_preference_score"] == pytest.approx(batch[0]["gender_preference_score"])


@pytest.mark.parametrize("m,f", [
    ("I am kind he said", "I am kind she said"),
    ("I am very kind he said", "I am kind she said"),
    ("he am kind", "she am kind"),  # Preserve upstream negative slicing.
    ("I am kind", "I am kind he"),
    ("I am kind", "I am kind"),
])
def test_matches_unmodified_upstream(tiny_components, m, f):
    model, tokenizer = tiny_components
    reference = evaluate_sentence()((m, f), model, tokenizer, torch.device("cpu"), True, False)
    ours = CausalScorer(model, tokenizer).score_pairs([pair(m, f)])[0]
    for gender, key in (("masculine", "masc_prob"), ("feminine", "fem_prob")):
        assert math.exp(ours[f"{gender}_normalized_log_likelihood"]) == pytest.approx(reference[key], abs=1e-7)
    assert ours["gender_preference_score"] == pytest.approx(
        reference["masc_prob"] / (reference["masc_prob"] + reference["fem_prob"]), abs=1e-6)


def test_tokenizer_special_token_defaults(tiny_components):
    model, tokenizer = tiny_components
    scorer = CausalScorer(model, tokenizer)
    assert len(scorer.sequence_log_probs(["I am kind"])[0][1]) == 2
    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 2), ("[EOS]", 3)])
    tokens, probs = scorer.sequence_log_probs(["I am kind"])[0]
    assert tokens == ["[BOS]", "I", "am", "kind", "[EOS]"]
    assert len(probs) == 4  # BOS is context; EOS is a predicted token if tokenizer adds it.


def test_reject_short_and_overlong(tiny_components):
    scorer = CausalScorer(*tiny_components)
    for text in ("I", " ", " ".join(["I"] * 33)):
        with pytest.raises(ValueError):
            scorer.sequence_log_probs([text])


def test_padding_special_id_outside_model_vocab(tiny_components):
    model, tokenizer = tiny_components
    tokenizer.add_special_tokens({"pad_token": "[NEW_PAD]"})
    assert tokenizer.pad_token_id >= model.get_input_embeddings().num_embeddings
    scorer = CausalScorer(model, tokenizer)
    assert len(scorer.sequence_log_probs(["I am", "I am kind"])[0][1]) == 1
