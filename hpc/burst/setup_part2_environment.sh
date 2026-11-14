#!/usr/bin/env bash
# Run on a Burst CPU session before GPU submission. Never edit an active overlay.
set -euo pipefail
release=${1:?Pass the absolute extracted Part 2 release directory}
environment_dir=/scratch/yc7009/a4_env
overlay=${A4_PART2_OVERLAY:-$environment_dir/part2.ext3}
image=${A4_PART2_IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}
test -f "$release/part2/requirements.txt"
if [ -e "$overlay" ]; then
  printf 'Refusing to overwrite existing environment: %s\n' "$overlay" >&2
  exit 1
fi
# A separate copy preserves the completed Part 1 environment and experiments.
cp --sparse=always "$environment_dir/part1.ext3" "$overlay"
singularity exec --bind /scratch --overlay "$overlay" "$image" /bin/bash -c '
set -euo pipefail
unset PYTHONPATH
source /ext3/env.sh
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/scratch/yc7009/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
# SciPy is inherited from the base environment; pin a version compatible with
# the assignment-required numpy==1.26.0 without changing assignment pins.
python -m pip install --disable-pip-version-check --no-cache-dir -r "$1/part2/requirements.txt" scipy==1.11.4
python -m pip check
python -m pip freeze > /scratch/yc7009/a4_env/part2-installed.txt
python - <<"PY"
from transformers import T5TokenizerFast, T5ForConditionalGeneration
tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
print("PRETRAINED_READY", sum(p.numel() for p in model.parameters()), model.config._commit_hash)
PY
' a4-part2-setup "$release"
printf 'PART2_ENV_READY\n'
