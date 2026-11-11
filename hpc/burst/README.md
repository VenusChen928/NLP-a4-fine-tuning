# Assignment 4 Part 1 on Burst

Use the course Burst OOD site in Safari: https://ood.burst.hpc.nyu.edu/.
Burst is separate from Torch. On 2026-11-11, a CPU JupyterLab session supplied
the working terminal; Files > Open in Terminal was unavailable.

## Files to keep

- `run_part1_gpu.sh`: shared GPU/container launcher using the existing Part 1
  overlay. Deploy to `/scratch/yc7009/a4_env/run_part1_gpu.sh` on Burst.
- `q2_eval.sbatch`: evaluates the Q1 checkpoint with `TYPO_PROB=0.40`; requests
  one A100, 4 CPUs, 32 GB RAM and 30 minutes. It does not retrain the model.
- This README: submission commands and the location of saved results.

The launcher and evaluation script disable Python bytecode generation.
Model weights, container images and environment copies do not belong here.

## Submit and inspect a deliberately requested Q2 run

After syncing the matching source and scripts, run inside the Burst terminal:

```bash
cd /scratch/yc7009/projects/assignment4
sbatch hpc/burst/q2_eval.sbatch
```

The script checks the exact hashes of `part1/main.py` and `part1/utils.py` before
and after evaluation. Keep them unchanged while a job is queued or running;
update the checks deliberately when changing the source. The current source
has only comment whitespace cleaned up since job 28708, with identical Python
AST and transformation behavior. Historical run records retain their original
source hashes.

To inspect the already completed p=0.40 run without resubmitting it:

```bash
squeue -j 28708
sacct -j 28708 --format=JobID,JobName,State,ExitCode,Elapsed
tail -n 25 /scratch/yc7009/projects/assignment4/runs/q2_typo_p040_28708/eval.log
```

The batch job runs independently of Jupyter and the browser. It uses
`/scratch/yc7009/projects/assignment4/runs/q1_25970/out/` as the model checkpoint.

## Saved results

Local records are in `part1/runs/`; remote records are in
`/scratch/yc7009/projects/assignment4/runs/`. The local archives contain the
outputs and one `run_info.json`, with no repeated source or environment copies.

| Run | Accuracy | Drop from Q1 | Slurm result |
| --- | --- | --- | --- |
| Q1 `25970` | 93.072% | — | COMPLETED, 0:0 |
| Q2 p=0.30 `28674` | 89.788% | 3.284 points | COMPLETED, 0:0 |
| Q2 p=0.40 `28708` | 88.036% | 5.036 points | COMPLETED, 0:0 |

The p=0.40 run took 00:05:24 and exceeds Q2b's greater-than-4-point criterion.
Its local folder `q2_typo_p040_28708/` contains `out_transformed.txt`, `eval.log`,
`debug_cli.log` and `run_info.json`. Downloaded outputs match HPC checksums;
all 25,000 predictions were checked against the Q1 label order. Q1's `out/`
contains the checkpoint needed for subsequent comparisons. The separate local
Q1 debug checkpoint is not the full-run checkpoint.

## Local transformation debug

From the project root:

```bash
cd part1
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  /opt/anaconda3/envs/nlp-a4-p1/bin/python -B main.py \
  --eval_transformed --debug_transformation
```

This prints five pairs and exits before model evaluation. Both p=0.30 and p=0.40
full debug CLI runs passed. Detailed checks are in their existing run records.
Use the same final p=0.40 transformation for subsequent Q3 comparisons.
