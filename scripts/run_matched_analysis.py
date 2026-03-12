"""Run analysis on matched extraction results — CPU only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.matched_analysis import run_matched_analysis

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

if __name__ == "__main__":
    run_matched_analysis(RESULTS_DIR)
