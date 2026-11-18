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

set -euo pipefail
mode=${1:?Pass smoke, baseline or train}
source_root=${2:?Pass the absolute synced project or source directory}
experiment=${3:-baseline_v1}
sql_preprocessing=${4:-none}
max_n_epochs=${5:-20}
learning_rate=${6:-3e-4}
post_training_dev=${7:-none}

# Append initialization and stopping options without changing existing arguments
model_mode=${8:-finetune}
min_n_epochs=${9:-1}
patience_epochs=${10:-5}

# Share experiment settings and keep the baseline epoch budget as the default
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
case "$mode" in
  smoke|baseline) run_prefix="$mode" ;;
  train) run_prefix="$experiment" ;;
  *) exit 2 ;;
esac
if [ "$model_mode" = scratch ]; then run_prefix="scratch_${run_prefix}"; fi

project=/scratch/yc7009/projects/assignment4
run_dir="$project/part2/runs/${run_prefix}_${SLURM_JOB_ID}"
runner="$source_root/hpc/burst/run_part2_gpu.sh"
mkdir "$run_dir"
cd "$run_dir"
exec > >(tee -a train_eval.log) 2>&1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export A4_RUN_KIND="$mode"
mkdir source records results reports
cp "$source_root"/part2/*.py "$source_root/part2/requirements.txt" source/
cp "$source_root"/part2/tools/*.py source/
cp "$source_root/release_manifest.json" release_manifest.json
cp "$source_root/hpc/burst/"* source/
ln -s "$project/part2/data" data
cp "$project/part2/records/ground_truth_dev.pkl" records/

finish() {
  status=$?
  trap - EXIT
  set +e
  bash "$runner" python -B source/experiment_report.py finalize --exit-code "$status" --experiment-name "$experiment"
  report_status=$?
  if [ "$status" -eq 0 ] && [ "$report_status" -ne 0 ]; then status=$report_status; fi
  printf 'Pipeline exit code: %s\n' "$status"
  exit "$status"
}
trap finish EXIT

bash "$runner" python -B source/experiment_report.py prepare --experiment-name "$experiment" --sql-preprocessing "$sql_preprocessing" --model-mode "$model_mode"
if [ "$mode" = smoke ]; then
  bash "$runner" python -B source/baseline_smoke.py --sql_preprocessing "$sql_preprocessing" "--$model_mode"
else
  # Vary the training budget while retaining the other baseline hyperparameters
  command=(python -B -u source/train_t5.py "--$model_mode" --experiment_name "$experiment" --sql_preprocessing "$sql_preprocessing"
    --learning_rate "$learning_rate" --weight_decay 0 --batch_size 16 --test_batch_size 16
    --scheduler_type linear --num_warmup_epochs 1 --max_n_epochs "$max_n_epochs"
    --min_n_epochs "$min_n_epochs" --patience_epochs "$patience_epochs" --max_new_tokens 512 --num_beams 1 --seed 42)
  printf '%q ' "${command[@]}" > command.txt
  printf '\n' >> command.txt
  bash "$runner" "${command[@]}"
  if [ "$post_training_dev" = beam4_valid ]; then
    printf 'Starting best-checkpoint beam4_valid dev evaluation\n'
    bash "$runner" python -B -u source/export_test.py \
      --run-dir "$run_dir" --output-dir "$run_dir/dev/beam4_valid" \
      --split dev --num-beams 4 --valid --query-timeout 5 \
      --batch-size 16 --threads 4 2>&1 | tee dev_beam4_valid.log
  fi
fi
