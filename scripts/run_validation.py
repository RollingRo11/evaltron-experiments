"""Run behavioral validation: generate code and check EA direction predicts type hints."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_loader import load_model_and_tokenizer
from src.behavioral_validation import run_validation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PROMPTS_PATH = PROJECT_ROOT / "prompts" / "validation_prompts.jsonl"
SYSTEM_PROMPTS_PATH = PROJECT_ROOT / "prompts" / "validation_system_prompts.jsonl"


def main():
    model, tokenizer = load_model_and_tokenizer()

    print("\n=== Running Behavioral Validation ===")
    print(f"10 prompts x 6 system prompts = 60 generations")
    records = run_validation(
        model, tokenizer, RESULTS_DIR,
        PROMPTS_PATH, SYSTEM_PROMPTS_PATH,
    )
    print(f"\nDone. {len(records)} generations completed.")


if __name__ == "__main__":
    main()
