#!/usr/bin/env bash
#SBATCH --job-name=a4-q2-typo040
#SBATCH --account=ds_ga_1011-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/yc7009/projects/assignment4/runs/slurm-%j.log

set -euo pipefail
PROJECT=/scratch/yc7009/projects/assignment4
RUN_DIR="$PROJECT/runs/q2_typo_p040_${SLURM_JOB_ID}"
MODEL_DIR="$PROJECT/runs/q1_25970/out"
RUNNER=/scratch/yc7009/a4_env/run_part1_gpu.sh
mkdir "$RUN_DIR"
cd "$RUN_DIR"
exec > eval.log 2>&1
finish() {
  status=$?
  printf 'Process exit code: %s\n' "$status"
  # Slurm creates this before the script starts; preserve it if it contains errors.
  slurm_log="$PROJECT/runs/slurm-${SLURM_JOB_ID}.log"
  if [ -f "$slurm_log" ] && [ ! -s "$slurm_log" ]; then
    rm -- "$slurm_log"
  fi
}
trap finish EXIT
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
test -s "$MODEL_DIR/pytorch_model.bin"
test -s "$MODEL_DIR/config.json"

# Check the debugged p=0.40 source; do not duplicate code or environments in runs.
check_source() {
  printf '%s  %s\n' \
    00236c763967f90972eed123de60b358cb693a360f575a401a779c4551ef8431 "$PROJECT/part1/main.py" \
    9883a560434c7d1742cbe49bb7c1dde35c35448d556ce21744d18f0f2bc9687c "$PROJECT/part1/utils.py" | sha256sum -c -
}
check_source
bash "$RUNNER" python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU", torch.cuda.get_device_name(0)); print("PyTorch", torch.__version__)'

# This is evaluation only. The Q1 full-run checkpoint is never retrained.
bash "$RUNNER" python -u "$PROJECT/part1/main.py" \
  --eval_transformed --model_dir "$MODEL_DIR" --batch_size 8
check_source

python3 - <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path

baseline = Path('/scratch/yc7009/projects/assignment4/runs/q1_25970/out_original.txt')
output = Path('out_transformed.txt')
def read_predictions(path):
    values = [int(x) for x in path.read_text().splitlines()]
    assert len(values) == 50000, (str(path), len(values))
    assert set(values) <= {0, 1}, str(path)
    return values[::2], values[1::2]
original, labels = read_predictions(baseline)
transformed, transformed_labels = read_predictions(output)
assert labels == transformed_labels, 'Test label order differs from Q1'
correct_original = sum(p == y for p, y in zip(original, labels))
correct_transformed = sum(p == y for p, y in zip(transformed, labels))
project = Path('/scratch/yc7009/projects/assignment4')
metrics = {
    'test_examples': len(labels),
    'baseline_correct': correct_original,
    'transformed_correct': correct_transformed,
    'baseline_accuracy': correct_original / len(labels),
    'transformed_accuracy': correct_transformed / len(labels),
    'accuracy_drop_percentage_points': 100 * (correct_original - correct_transformed) / len(labels),
    'drop_strictly_more_than_4_points': (correct_original - correct_transformed) > 0.04 * len(labels),
    'label_order_matches_q1': True,
}
run_info = {
    'task': 'Q2 transformed-test evaluation',
    'execution': 'Burst HPC',
    'job_id': int(os.environ['SLURM_JOB_ID']),
    'process_exit_code': 0,
    'parameters': {'batch_size': 8, 'typo_probability': 0.40, 'transform_seed': 0},
    'model_checkpoint': '../q1_25970/out/',
    'source_commit': subprocess.check_output(['git', '-C', str(project), 'rev-parse', 'HEAD'], text=True).strip(),
    'source_has_uncommitted_changes': bool(subprocess.check_output(['git', '-C', str(project), 'diff', 'HEAD', '--', 'part1/main.py', 'part1/utils.py'])),
    'source_sha256': {name: hashlib.sha256((project / 'part1' / name).read_bytes()).hexdigest() for name in ['main.py', 'utils.py']},
    'results': metrics,
    'output_sha256': {'out_transformed.txt': hashlib.sha256(output.read_bytes()).hexdigest()},
}
Path('run_info.json').write_text(json.dumps(run_info, indent=2) + '\n')
print(json.dumps(run_info, indent=2))
PY
