#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/yc7009/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
image=${A4_PART2_IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}
overlay=${A4_PART2_OVERLAY:-/scratch/yc7009/a4_env/part2.ext3}
exec singularity exec --nv --bind /scratch --overlay "$overlay:ro" "$image" \
  /bin/bash -c 'set -euo pipefail; unset PYTHONPATH; source /ext3/env.sh; exec "$@"' a4-part2 "$@"
