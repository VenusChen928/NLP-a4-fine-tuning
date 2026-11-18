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
max_n_epochs=${4:-20}
learning_rate=${5:-3e-4}
post_training_dev=${6:-none}

# Keep the original command valid; scratch settings are explicit trailing options
model_mode=${7:-finetune}
min_n_epochs=${8:-1}
patience_epochs=${9:-5}

runner="$source_root/hpc/burst/run_part2_gpu.sh"
project=/scratch/yc7009/projects/assignment4

# Keep baseline defaults while allowing a named experiment and epoch budget
[[ "$experiment" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || exit 2
[[ "$max_n_epochs" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$min_n_epochs" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$patience_epochs" =~ ^[1-9][0-9]*$ ]] || exit 2
(( min_n_epochs <= max_n_epochs )) || exit 2
case "$model_mode" in finetune|scratch) ;; *) exit 2 ;; esac
python3 - "$learning_rate" <<'PY'
import math, sys
value = float(sys.argv[1])
assert math.isfinite(value) and value > 0, 'Learning rate must be positive and finite'
PY
case "$post_training_dev" in none|beam4_valid) ;; *) exit 2 ;; esac
case "$sql_preprocessing" in none|operators_v1) ;; *) exit 2 ;; esac
training_mode=train
if [ "$experiment" = baseline_v1 ] && [ "$sql_preprocessing" = none ] && [ "$model_mode" = finetune ]; then training_mode=baseline; fi
smoke_prefix=smoke
if [ "$model_mode" = scratch ]; then smoke_prefix=scratch_smoke; fi

printf 'Starting GPU smoke gate for job %s\n' "$SLURM_JOB_ID"
timeout --signal=TERM --kill-after=60s 20m \
  bash "$source_root/hpc/burst/part2.sbatch" smoke "$source_root" "$experiment" "$sql_preprocessing" "$max_n_epochs" "$learning_rate" none "$model_mode" "$min_n_epochs" "$patience_epochs"

# Require the matching preprocessing mode and a complete GPU smoke report
bash "$runner" python -B - "$project/part2/runs/${smoke_prefix}_${SLURM_JOB_ID}" "$sql_preprocessing" "$model_mode" <<'PY'
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
assert info['model_mode'] == sys.argv[3]
assert report['initialization'] == info['initialization'] == ('pretrained' if sys.argv[3] == 'finetune' else 'random')
if sys.argv[2] == 'operators_v1':
    for checks in report['sql_processing_checks'].values():
        assert all(checks[name] == 0 for name in ['conversion_mismatches', 'tokenizer_roundtrip_mismatches', 'examples_with_unk', 'examples_above_generation_limit'])
print('SMOKE_GATE_PASSED', json.dumps(report['dev_metrics']))
PY

printf 'Starting independent %s experiment %s for job %s\n' "$model_mode" "$experiment" "$SLURM_JOB_ID"
bash "$source_root/hpc/burst/part2.sbatch" "$training_mode" "$source_root" "$experiment" "$sql_preprocessing" "$max_n_epochs" "$learning_rate" "$post_training_dev" "$model_mode" "$min_n_epochs" "$patience_epochs"
