"""Run deconfounded analysis — CPU only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.deconfound_analysis import run_deconfound_analysis

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

if __name__ == "__main__":
    run_deconfound_analysis(RESULTS_DIR)
