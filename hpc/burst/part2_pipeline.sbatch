#!/usr/bin/env bash
#SBATCH --job-name=a4-t5-baseline
#SBATCH --account=ds_ga_1011-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/yc7009/projects/assignment4/part2/slurm/slurm-%j.log

# Share one allocation while keeping smoke and baseline artifacts separate.
set -euo pipefail
release=${1:?Pass the absolute immutable release directory}
runner="$release/hpc/burst/run_part2_gpu.sh"
project=/scratch/yc7009/projects/assignment4
printf 'Starting GPU smoke gate for job %s\n' "$SLURM_JOB_ID"
timeout --signal=TERM --kill-after=60s 20m \
  bash "$release/hpc/burst/part2.sbatch" smoke "$release"

# Require a complete, finite smoke report before spending time on full training.
bash "$runner" python -B - "$project/part2/runs/smoke_${SLURM_JOB_ID}" <<'PY'
import json, math, sys
from pathlib import Path
root = Path(sys.argv[1])
info = json.loads((root / 'run_info.json').read_text())
report = json.loads((root / 'reports/smoke.json').read_text())
assert info['status'] == 'completed' and info['process_exit_code'] == 0
assert report['checkpoint_round_trip'] and report['optimizer_steps'] == 2
assert math.isfinite(report['train_loss'])
assert all(math.isfinite(value) for value in report['dev_metrics'].values())
assert report['peak_gpu_memory_bytes'] < info['gpu_memory_bytes']
print('SMOKE_GATE_PASSED', json.dumps(report['dev_metrics']))
PY

printf 'Starting independent pretrained baseline for job %s\n' "$SLURM_JOB_ID"
bash "$release/hpc/burst/part2.sbatch" baseline "$release"
