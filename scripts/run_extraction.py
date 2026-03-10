"""Main script: load model, extract activations for both concepts."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_loader import load_model_and_tokenizer
from src.prompt_runner import extract_activations_for_concept

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def main():
    model, tokenizer = load_model_and_tokenizer()

    print("\n=== Extracting Type Hints activations ===")
    extract_activations_for_concept(
        "type_hints",
        PROMPTS_DIR / "type_hints_pairs.jsonl",
        model,
        tokenizer,
        RESULTS_DIR,
    )

    print("\n=== Extracting Eval Awareness activations ===")
    extract_activations_for_concept(
        "eval_awareness",
        PROMPTS_DIR / "eval_awareness_pairs.jsonl",
        model,
        tokenizer,
        RESULTS_DIR,
    )

    print("\nAll activations extracted successfully!")


if __name__ == "__main__":
    main()
