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

# Share one allocation while keeping smoke and training artifacts separate
set -euo pipefail
source_root=${1:?Pass the absolute synced project or source directory}
experiment=${2:-baseline_v1}
sql_preprocessing=${3:-none}
runner="$source_root/hpc/burst/run_part2_gpu.sh"
project=/scratch/yc7009/projects/assignment4

# SQL processing: retain the old baseline command while supporting a named experiment
[[ "$experiment" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || exit 2
case "$sql_preprocessing" in none|operators_v1) ;; *) exit 2 ;; esac
training_mode=train
if [ "$experiment" = baseline_v1 ] && [ "$sql_preprocessing" = none ]; then training_mode=baseline; fi

printf 'Starting GPU smoke gate for job %s\n' "$SLURM_JOB_ID"
timeout --signal=TERM --kill-after=60s 20m \
  bash "$source_root/hpc/burst/part2.sbatch" smoke "$source_root" "$experiment" "$sql_preprocessing"

# Require the matching preprocessing mode and a complete GPU smoke report
bash "$runner" python -B - "$project/part2/runs/smoke_${SLURM_JOB_ID}" "$sql_preprocessing" <<'PY'
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
assert report['sql_preprocessing'] == info['sql_preprocessing'] == sys.argv[2]
if sys.argv[2] == 'operators_v1':
    for checks in report['sql_processing_checks'].values():
        assert all(checks[name] == 0 for name in ['conversion_mismatches', 'tokenizer_roundtrip_mismatches', 'examples_with_unk', 'examples_above_generation_limit'])
print('SMOKE_GATE_PASSED', json.dumps(report['dev_metrics']))
PY

printf 'Starting independent pretrained experiment %s for job %s\n' "$experiment" "$SLURM_JOB_ID"
bash "$source_root/hpc/burst/part2.sbatch" "$training_mode" "$source_root" "$experiment" "$sql_preprocessing"
