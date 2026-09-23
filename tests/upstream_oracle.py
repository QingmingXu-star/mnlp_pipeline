"""Execute unchanged upstream function bodies without its CLI/login/generation side effects."""

import ast
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "vendor" / "eurogest"


def module(name):
    spec = importlib.util.spec_from_file_location(f"reference_{name}", ROOT / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def function(filename, name, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / filename), "exec"), namespace)
    return namespace[name]


def evaluate_sentence():
    utils = module("model_utils")
    return function("main.py", "evaluate_sentence", {
        "np": np, "tokenise": utils.tokenise, "get_sequence_log_probs": utils.get_sequence_log_probs,
        # Upstream generates text AFTER computing probabilities. Only that unrelated step is stubbed.
        "generate_new_tokens": lambda *args, **kwargs: "",
    })
