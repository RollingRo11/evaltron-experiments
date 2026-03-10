"""Offline analysis of saved activations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.direction_analysis import run_full_analysis

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    run_full_analysis(RESULTS_DIR)


if __name__ == "__main__":
    main()
