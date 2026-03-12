"""Run matched extraction: generate code under eval/deploy and capture activations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_loader import load_model_and_tokenizer
from src.matched_extraction import run_matched_extraction

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PAIRS_PATH = PROJECT_ROOT / "prompts" / "matched_pairs.jsonl"


def main():
    model, tokenizer = load_model_and_tokenizer()

    print("\n=== Matched Extraction: 20 pairs x 2 conditions = 40 generations ===")
    run_matched_extraction(model, tokenizer, PAIRS_PATH, RESULTS_DIR)
    print("\nExtraction complete!")


if __name__ == "__main__":
    main()
