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
mode=${1:?Pass smoke or baseline}
release=${2:?Pass the absolute immutable release directory}
case "$mode" in smoke|baseline) ;; *) exit 2 ;; esac
project=/scratch/yc7009/projects/assignment4
run_dir="$project/part2/runs/${mode}_${SLURM_JOB_ID}"
runner="$release/hpc/burst/run_part2_gpu.sh"
mkdir "$run_dir"
cd "$run_dir"
exec > >(tee -a train_eval.log) 2>&1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export A4_RUN_KIND="$mode"
mkdir source records results reports
cp "$release"/part2/*.py "$release/part2/requirements.txt" source/
cp "$release"/part2/tools/*.py source/
cp "$release/release_manifest.json" release_manifest.json
cp "$release/hpc/burst/"* source/
ln -s "$project/part2/data" data
cp "$project/part2/records/ground_truth_dev.pkl" records/

finish() {
  status=$?
  trap - EXIT
  set +e
  bash "$runner" python -B source/experiment_report.py finalize --exit-code "$status"
  report_status=$?
  if [ "$status" -eq 0 ] && [ "$report_status" -ne 0 ]; then status=$report_status; fi
  printf 'Pipeline exit code: %s\n' "$status"
  exit "$status"
}
trap finish EXIT

bash "$runner" python -B source/experiment_report.py prepare
if [ "$mode" = smoke ]; then
  bash "$runner" python -B source/baseline_smoke.py
else
  command=(python -B -u source/train_t5.py --finetune --experiment_name baseline_v1
    --learning_rate 3e-4 --weight_decay 0 --batch_size 16 --test_batch_size 16
    --scheduler_type linear --num_warmup_epochs 1 --max_n_epochs 20
    --patience_epochs 5 --max_new_tokens 512 --num_beams 1 --seed 42)
  printf '%q ' "${command[@]}" > command.txt
  printf '\n' >> command.txt
  bash "$runner" "${command[@]}"
fi
