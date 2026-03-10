#!/bin/bash
#SBATCH --job-name=evaltron-analyze
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=results/analyze_%j.log

cd $SLURM_SUBMIT_DIR/..

uv run python scripts/run_analysis.py
