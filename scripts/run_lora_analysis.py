"""Run LoRA subspace analysis — CPU only, no model needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lora_subspace_analysis import run_lora_subspace_analysis

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

if __name__ == "__main__":
    run_lora_subspace_analysis(RESULTS_DIR)
