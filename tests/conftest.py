import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


@pytest.fixture
def tiny_components():
    """An offline random HF causal model; a test fixture, not a research model."""
    torch.set_num_threads(1)
    torch.manual_seed(7)
    vocab = {word: i for i, word in enumerate(
        ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "I", "am", "kind", "he", "she", "said", "very", "."])}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                       pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]")
    config = GPT2Config(vocab_size=len(vocab), n_positions=32, n_embd=16, n_layer=1,
                        n_head=2, bos_token_id=2, eos_token_id=3, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0)
    config._attn_implementation = "eager"
    return GPT2LMHeadModel(config).eval(), tokenizer
