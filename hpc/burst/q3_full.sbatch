#!/usr/bin/env bash
#SBATCH --job-name=a4-q3-typo040
#SBATCH --account=ds_ga_1011-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/yc7009/projects/assignment4/runs/slurm-%j.log

set -euo pipefail
: "${A4_SOURCE_COMMIT:?Set A4_SOURCE_COMMIT to the verified uploaded source commit}"
export PROJECT=/scratch/yc7009/projects/assignment4
export RUN_DIR="$PROJECT/runs/q3_typo_p040_${SLURM_JOB_ID}"
RUNNER="$PROJECT/hpc/burst/run_part1_gpu.sh"
mkdir "$RUN_DIR"
cd "$RUN_DIR"
exec > train_eval.log 2>&1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

finish() {
  status=$?
  trap - EXIT
  set +e
  printf 'Process exit code: %s\n' "$status"
  python3 - "$status" <<'PY'
import hashlib, json, sys
from pathlib import Path
p = Path('run_info.json')
record = json.loads(p.read_text()) if p.exists() else {}
record['process_exit_code'] = int(sys.argv[1])
record['pipeline_status'] = 'succeeded' if int(sys.argv[1]) == 0 else 'failed'
names = ['out_augmented/config.json', 'out_augmented/pytorch_model.bin',
         'out_augmented_original.txt', 'out_augmented_transformed.txt', 'train_eval.log']
record['output_sha256'] = {n: hashlib.sha256(Path(n).read_bytes()).hexdigest()
                           for n in names if Path(n).is_file()}
p.write_text(json.dumps(record, indent=2) + '\n')
PY
  slurm_log="$PROJECT/runs/slurm-${SLURM_JOB_ID}.log"
  if [ -f "$slurm_log" ] && [ ! -s "$slurm_log" ]; then rm -- "$slurm_log"; fi
  exit "$status"
}
trap finish EXIT

check_source() {
  printf '%s  %s\n' \
    00236c763967f90972eed123de60b358cb693a360f575a401a779c4551ef8431 "$PROJECT/part1/main.py" \
    9883a560434c7d1742cbe49bb7c1dde35c35448d556ce21744d18f0f2bc9687c "$PROJECT/part1/utils.py" | sha256sum -c -
}
check_source
python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
p = Path(os.environ['PROJECT'])
record = {
    'task': 'Q3 augmented training and original/transformed test evaluation',
    'execution': 'Burst HPC', 'job_id': int(os.environ['SLURM_JOB_ID']),
    'pipeline_status': 'running', 'source_commit': os.environ['A4_SOURCE_COMMIT'],
    'source_sha256': {n: hashlib.sha256((p / 'part1' / n).read_bytes()).hexdigest()
                      for n in ['main.py', 'utils.py']},
    'parameters': {'model': 'bert-base-cased', 'learning_rate': 5e-5,
                   'num_epochs': 3, 'batch_size': 8, 'training_seed': 0,
                   'augmentation_sample_seed': 42, 'typo_probability': 0.40,
                   'transform_seed': 0, 'original_train_examples': 25000,
                   'augmented_examples': 5000, 'total_train_examples': 30000},
    'checkpoint': 'out_augmented/',
    'baselines': {'q1_run': '../q1_25970/', 'q2_run': '../q2_typo_p040_28708/'},
}
Path('run_info.json').write_text(json.dumps(record, indent=2) + '\n')
PY

bash "$RUNNER" python -B -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU", torch.cuda.get_device_name(0)); print("PyTorch", torch.__version__)'

# main.py saves out_augmented before loading it for each evaluation.
# Any training exception stops this pipeline before evaluation.
bash "$RUNNER" python -B -u "$PROJECT/part1/main.py" \
  --train_augmented --eval --eval_transformed \
  --learning_rate 5e-5 --num_epochs 3 --batch_size 8
check_source
test -s out_augmented/pytorch_model.bin
test -s out_augmented/config.json

python3 - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ['PROJECT'])
def predictions(path):
    values = [int(v) for v in path.read_text().splitlines()]
    assert len(values) == 50000 and set(values) <= {0, 1}, str(path)
    return values[::2], values[1::2]
q1, labels = predictions(p / 'runs/q1_25970/out_original.txt')
q2, q2_labels = predictions(p / 'runs/q2_typo_p040_28708/out_transformed.txt')
original, original_labels = predictions(Path('out_augmented_original.txt'))
transformed, transformed_labels = predictions(Path('out_augmented_transformed.txt'))
assert labels == q2_labels == original_labels == transformed_labels
def accuracy(values):
    return sum(a == b for a, b in zip(values, labels)) / len(labels)
record = json.loads(Path('run_info.json').read_text())
record['results'] = {
    'test_examples_per_evaluation': len(labels), 'label_order_matches_baselines': True,
    'original_accuracy': accuracy(original), 'transformed_accuracy': accuracy(transformed),
    'q1_original_accuracy': accuracy(q1), 'q2_transformed_accuracy': accuracy(q2),
    'original_gain_percentage_points': 100 * (accuracy(original) - accuracy(q1)),
    'transformed_gain_percentage_points': 100 * (accuracy(transformed) - accuracy(q2)),
}
Path('run_info.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record['results'], indent=2))
PY
