"""Submit the LoRA subspace analysis SLURM job."""

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SBATCH_FILE = PROJECT_ROOT / "slurm" / "lora_analysis.sbatch"

(PROJECT_ROOT / "results").mkdir(exist_ok=True)

result = subprocess.run(
    ["sbatch", str(SBATCH_FILE)],
    cwd=PROJECT_ROOT / "slurm",
    capture_output=True,
    text=True,
)

if result.returncode == 0:
    print(result.stdout.strip())
    job_id = result.stdout.strip().split()[-1]
    print(f"Monitor with: squeue -j {job_id}")
    print(f"Logs at: results/lora_analysis_{job_id}.log")
else:
    print(f"Submission failed:\n{result.stderr}")
    raise SystemExit(1)
