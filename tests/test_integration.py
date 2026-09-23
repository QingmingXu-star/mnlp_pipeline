"""Real-model parity on handmade German fixtures, not EuroGEST research data."""

import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eurogest_mvp import MODEL_ID, MODEL_REVISION
from eurogest_mvp.scoring import CausalScorer
from upstream_oracle import evaluate_sentence


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="Set RUN_MODEL_TESTS=1 with cached model")
def test_real_german_gpt2_matches_official_and_batch():
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                               local_files_only=True, attn_implementation="eager")
    scorer = CausalScorer(model, tokenizer)
    pairs = [
        {"masculine_sentence": "„Ich bin freundlich“, sagte er", "feminine_sentence": "„Ich bin freundlich“, sagte sie"},
        {"masculine_sentence": "Ich bin ein guter Lehrer.", "feminine_sentence": "Ich bin eine gute Lehrerin."},
        {"masculine_sentence": "Er ist stark.", "feminine_sentence": "Sie ist stark."},
        {"masculine_sentence": "Ich helfe meinen Freunden sehr gerne.", "feminine_sentence": "Ich helfe meinen Freunden sehr gerne."},
    ]
    actual = scorer.score_pairs(pairs)
    oracle = evaluate_sentence()
    for pair, row in zip(pairs, actual):
        result = oracle(tuple(pair.values()), model, tokenizer, torch.device("cpu"), True, False)
        expected = result["masc_prob"] / (result["masc_prob"] + result["fem_prob"])
        assert row["gender_preference_score"] == pytest.approx(expected, abs=2e-5)
        single = scorer.score_pairs([pair])[0]
        assert row["masculine_normalized_log_likelihood"] == pytest.approx(
            single["masculine_normalized_log_likelihood"], abs=1e-4)
    assert tokenizer("Hello world")["input_ids"][0] != tokenizer.bos_token_id
