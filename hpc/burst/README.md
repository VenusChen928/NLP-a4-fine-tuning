# Assignment 4 Part 1 on Burst

Use the course Burst OOD site: https://ood.burst.hpc.nyu.edu/.
Burst is separate from Torch. On 2026-11-11, a CPU JupyterLab session supplied
the working terminal; Files > Open in Terminal was unavailable.

## Files to keep

- `run_part1_gpu.sh`: shared GPU/container launcher using the existing Part 1
  overlay. Deploy to `/scratch/yc7009/a4_env/run_part1_gpu.sh` on Burst.
- `q2_eval.sbatch`: evaluates the Q1 checkpoint with `TYPO_PROB=0.40`; requests
  one A100, 4 CPUs, 32 GB RAM and 30 minutes. It does not retrain the model.
- `q3_full.sbatch`: trains on 25,000 original plus 5,000 transformed training
  examples, then evaluates both full test sets in the same GPU job.
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
includes the Q3 augmentation loader. Formatting changes in `utils.py` preserve
its Python AST and Q2 transformation behavior. Historical run records retain
their original source hashes.

To inspect the already completed p=0.40 run without resubmitting it:

```bash
squeue -j 28708
sacct -j 28708 --format=JobID,JobName,State,ExitCode,Elapsed
tail -n 25 /scratch/yc7009/projects/assignment4/runs/q2_typo_p040_28708/eval.log
```

The batch job runs independently of Jupyter and the browser. It uses
`/scratch/yc7009/projects/assignment4/runs/q1_25970/out/` as the model checkpoint.

## Q3 training and both evaluations

The Q3 script requests one A100, 4 CPUs, 32 GB RAM and a 2-hour limit. It uses
the repo's `hpc/burst/run_part1_gpu.sh`, starts from `bert-base-cased`, and runs:

```bash
python -B main.py --train_augmented --eval --eval_transformed \
  --learning_rate 5e-5 --num_epochs 3 --batch_size 8
```

The program saves `out_augmented/` before loading it for the two evaluations.
An exception stops the pipeline. Submit only after deploying the checked code,
setting `A4_SOURCE_COMMIT` to that code's actual Git commit:

```bash
sbatch --export=ALL,A4_SOURCE_COMMIT=<verified-source-commit> hpc/burst/q3_full.sbatch
```

The output directory is `runs/q3_typo_p040_<jobid>/`. It retains the checkpoint,
`out_augmented_original.txt`, `out_augmented_transformed.txt`, `train_eval.log`
and one `run_info.json`. The script checks output sizes and label order against
Q1/Q2, and records accuracies and output hashes on completion. `pipeline_status`
describes the script's outcome; use `sacct` to verify final Slurm status later.
Do not edit the remote source while the job is queued or running.

## Saved results

Local records are in `part1/runs/`; remote records are in
`/scratch/yc7009/projects/assignment4/runs/`. The local archives contain the
outputs and one `run_info.json`, with no repeated source or environment copies.

| Run | Accuracy | Comparison | Slurm result |
| --- | --- | --- | --- |
| Q1 `25970` | 93.072% | — | COMPLETED, 0:0 |
| Q2 p=0.30 `28674` | 89.788% | Q1 minus 3.284 points | COMPLETED, 0:0 |
| Q2 p=0.40 `28708` | 88.036% | Q1 minus 5.036 points | COMPLETED, 0:0 |
| Q3 original `29911` | 92.912% | Q1 minus 0.160 points | COMPLETED, 0:0 |
| Q3 transformed `29911` | 89.048% | Q2 p=0.40 plus 1.012 points | COMPLETED, 0:0 |

The p=0.40 run took 00:05:24 and exceeds Q2b's greater-than-4-point criterion.
Its local folder `q2_typo_p040_28708/` contains `out_transformed.txt`, `eval.log`,
`debug_cli.log` and `run_info.json`. Downloaded outputs match HPC checksums;
all 25,000 predictions were checked against the Q1 label order. Q1's `out/`
contains the checkpoint needed for subsequent comparisons. The separate local
Q1 debug checkpoint is not the full-run checkpoint.

Q3 `29911` completed training and both evaluations in 00:43:09. Its checkpoint,
logs and run record are preserved locally and on Burst. Git includes the four
final prediction files and a [results index](../../part1/runs/README.md), while
checkpoints, logs and earlier/debug runs remain local. Cloning the repo does not
download the model weights required for evaluation-only commands.

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
