#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
exec singularity exec --nv --bind /scratch \
  --overlay /scratch/yc7009/a4_env/part1.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c '
set -e
unset PYTHONPATH
source /ext3/env.sh
exec "$@"
' a4-part1 "$@"
