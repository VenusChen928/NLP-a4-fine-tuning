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

# SQL processing: share one experiment configuration across all entry points
[[ "$experiment" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || exit 2
case "$sql_preprocessing" in none|operators_v1) ;; *) exit 2 ;; esac
case "$mode" in
  smoke|baseline) run_prefix="$mode" ;;
  train) run_prefix="$experiment" ;;
  *) exit 2 ;;
esac

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

bash "$runner" python -B source/experiment_report.py prepare --experiment-name "$experiment" --sql-preprocessing "$sql_preprocessing"
if [ "$mode" = smoke ]; then
  bash "$runner" python -B source/baseline_smoke.py --sql_preprocessing "$sql_preprocessing"
else
  # SQL processing: retain every baseline training and decoding hyperparameter
  command=(python -B -u source/train_t5.py --finetune --experiment_name "$experiment" --sql_preprocessing "$sql_preprocessing"
    --learning_rate 3e-4 --weight_decay 0 --batch_size 16 --test_batch_size 16
    --scheduler_type linear --num_warmup_epochs 1 --max_n_epochs 20
    --patience_epochs 5 --max_new_tokens 512 --num_beams 1 --seed 42)
  printf '%q ' "${command[@]}" > command.txt
  printf '\n' >> command.txt
  bash "$runner" "${command[@]}"
fi
