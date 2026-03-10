#!/bin/bash
#SBATCH --job-name=evaltron-validate
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --output=results/validate_%j.log

cd $SLURM_SUBMIT_DIR/..

uv run python scripts/run_validation.py
