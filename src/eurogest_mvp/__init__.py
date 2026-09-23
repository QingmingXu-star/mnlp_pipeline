"""Minimal, auditable reproduction of EuroGEST's default evaluation."""

UPSTREAM_COMMIT = "b54359247d6a0cfab3247cc02be152e809004cbe"
DATASET_ID = "utter-project/EuroGEST"
DATASET_REVISION = "a97bd293c32e85de950db7573e0b8be8b80e83cd"
MODEL_ID = "dbmdz/german-gpt2"
MODEL_REVISION = "ab6efd04479f70d66df40e7bfcb17ba41e9cd6d5"

LABELS = dict(enumerate([
    "Emotional", "Gentle", "Empathetic", "Neat", "Social", "Weak", "Beautiful",
    "Tough", "Self-confident", "Professional", "Rational", "Providers", "Leaders",
    "Childish", "Sexual", "Strong",
], start=1))
