# Part 2 baseline experiments on Burst

Use the course Burst Open OnDemand portal, not Torch. Local source is the
source of truth. Sync local scripts directly to the matching remote project paths. Each job
keeps its own source copy and a new directory named with its Slurm job ID. Preserve successful and failed
run records; never reuse a run directory for another experiment.

## Storage

The tree below is the raw **remote job working layout**. Completed local
experiments use the compact layout in [the Part 2 README](../../part2/README.md).
Fine-tuned local archives retain the selected checkpoint and one copy of each output;
source and logs are compressed, and metadata are consolidated in `run.json`.
Scratch archives retain native checkpoint/output paths and `.log.gz` logs.
The helper scripts live in `part2/tools/` in the repository and are copied into
the flat `source/` folder for an HPC job.

Remote project root: `/scratch/yc7009/projects/assignment4`.
Local project root: `/Users/venus/Desktop/NLP/Assignments/assignment4`.

```text
part2/
  data/                              shared supplied data and database
  records/ground_truth_dev.pkl       shared supplied dev records
  runs/smoke_<jobid>/                interface validation, not a baseline score
  runs/baseline_<jobid>/             one full experiment
    source/                         exact Python and batch scripts used
    release_manifest.json           source/data hashes and Git provenance
    command.txt                     exact training command
    run_info.json                   hardware, versions, status, output hashes
    train_eval.log                  training and evaluation stdout/stderr
    checkpoints/ft_experiments/baseline_v1/
      run_config.json               actual training/model configuration
      history.jsonl                 one metric row per epoch (1-based)
      best_metrics.json             epoch selected by dev Record F1
      dev_metrics.json              final evaluation of that checkpoint
      best/                         selected model weights and config
      last/                         final epoch weights and config
      tokenizer files               tokenizer used by this run
    results/                        final best-checkpoint dev SQL
    records/                        generated records and copied dev gold
    reports/                        table evidence and per-example dev output
  slurm/slurm-<jobid>.log            scheduler stdout/stderr
```

Hugging Face cache is shared at `/scratch/yc7009/hf_cache`; it does not belong
in Git or each run. Part 2 uses a separate `/scratch/yc7009/a4_env/part2.ext3`
environment, mounted read-only by jobs. Do not modify the Part 1 environment.
The environment and pretrained model cache must be prepared before submitting
the offline GPU runner. `best/` and `last/` contain model weights, not optimizer
or RNG state: they support evaluation, not exact interrupted-training resume.

`setup_part2_environment.sh <extracted-release>` documents the CPU preparation
recipe and refuses to overwrite an existing overlay. It installs the supplied
requirements plus SciPy 1.11.4: the inherited SciPy 1.17.1 was incompatible with
the assignment's NumPy 1.26.0 pin. Keep installation logs and the resulting
`part2-installed.txt`; every GPU run also records its installed package versions.

## Baseline definition

The first experiment fully fine-tunes `google-t5/t5-small` using its default
tokenizer. Text processing only strips surrounding whitespace. There is no
prefix, filtering, truncation or tokenizer extension. This default tokenizer
emits UNK for some SQL characters (including `<`), a documented baseline
limitation to consider in later dev-only experiments.

Training uses AdamW, LR 3e-4, weight decay 0, batch size 16, linear decay with
one warmup epoch, at most 20 epochs, patience 5 and seed 42. All parameters
are trainable, FP32, with one optimizer step per batch. Dev generation is
greedy, batch size 16, at most 512 new tokens. The checkpoint with greatest
dev Record F1 is selected; ties retain the earlier epoch. The final dev report
reloads that checkpoint. Local JSON logs are sufficient for this baseline;
W&B is optional and disabled in the batch command.

## Script synchronization and submission

Sync `part2/*.py`, `part2/requirements.txt`, `part2/tools/*.py` and the Part 2
batch/runner scripts in `hpc/burst/` directly to their matching remote paths.
No compressed release upload is required. Write `release_manifest.json` at the
remote project root with local source/data hashes for the existing preflight
checks. The filename is retained for compatibility with experiment reporting.
Each job copies the synchronized scripts and manifest into its run directory.
Do not edit the synchronized files between submission and job startup.

```bash
cd /scratch/yc7009/projects/assignment4
source_root=$PWD
# Avoid inheriting Jupyter job environment retrieval flags
unset SLURM_GET_USER_ENV SLURM_EXPORT_ENV SBATCH_GET_USER_ENV SBATCH_EXPORT
mkdir -p part2/runs part2/slurm
```

First inspect current course partition limits and the prepared Part 2
environment. The default full-training allocation is one A100, four CPU threads,
32 GB RAM and four hours. Override the time limit to 20 minutes for smoke.
These are requested limits, not an estimate of elapsed training time.

The preferred first-baseline command shares one A100 allocation between a
20-minute smoke gate and independent full training, avoiding a second cloud
node startup. It writes `runs/smoke_<jobid>/` and `runs/baseline_<jobid>/`.
Any failed smoke exit, non-finite metric, failed checkpoint round trip or
incomplete report prevents baseline training. The four-hour allocation limit
covers both stages; the smoke's internal limit excludes Slurm queue time.

```bash
sbatch "$source_root/hpc/burst/part2_pipeline.sbatch" "$source_root"
```

For the SQL operator experiment, use the same pipeline and allocation:

```bash
sbatch --export=ALL --job-name=a4-t5-sql-ops "$source_root/hpc/burst/part2_pipeline.sbatch" "$source_root" sql_ops_v1 operators_v1
```

This writes `runs/smoke_<jobid>/` and `runs/sql_ops_v1_<jobid>/`. The same
`operators_v1` mode is passed to preprocessing checks, GPU smoke, training and
reporting. Only the experiment name and SQL representation differ from the
baseline command. The smoke gate verifies the saved mode and lossless target
round trips before starting an independent pretrained T5-small training run.
The existing no-argument baseline mode remains supported.

The optional fourth pipeline argument sets the maximum epoch budget, defaulting
to 20. The 30-epoch experiment reuses the same scripts and pretrained model:

```bash
sbatch --export=ALL --job-name=a4-sql-ops-e30 "$source_root/hpc/burst/part2_pipeline.sbatch" "$source_root" sql_ops_e30_lr3e4 operators_v1 30
```

It keeps learning rate `3e-4`, patience 5 and all other training settings.
Increasing the maximum epochs also stretches the existing linear learning-rate
decay. The run is archived as `runs/sql_ops_e30_lr3e4_<jobid>/`; its greedy dev
outputs and subsequent `dev/beam4_valid/` evaluation share this training root.

The fifth pipeline argument sets the learning rate (default `3e-4`). The sixth
selects an automatic post-training dev evaluation (`none` by default, or
`beam4_valid`). For the learning-rate comparison:

```bash
sbatch --export=ALL --job-name=a4-e30-lr1e4 "$source_root/hpc/burst/part2_pipeline.sbatch" "$source_root" sql_ops_e30_lr1e4 operators_v1 30 1e-4 beam4_valid
```

The checkpoint is still selected by greedy dev Record F1. On successful
training, the same allocation loads that checkpoint and writes beam-4 +
validity predictions and metrics to `dev/beam4_valid/`, with a five-second
validity timeout and batch size 16. `dev_beam4_valid.log` and the final run
report retain the evaluation evidence. This mode never generates test output.
For standalone `part2.sbatch`, the corresponding optional arguments are
maximum epochs, learning rate and post-training dev mode after preprocessing.

### Scratch initialization

The final scratch run used the same architecture with random weights:

```bash
sbatch --export=ALL --time=02:00:00 --job-name=a4-scr-e25 \
  "$source_root/hpc/burst/part2_pipeline.sbatch" "$source_root" \
  ec_e25_lr3e4 operators_v1 25 3e-4 none scratch 20 20
```

The trailing pipeline arguments are initialization (`finetune` or `scratch`),
minimum epochs and patience. Defaults remain `finetune 1 5`. Scratch uses a
separate smoke gate and `checkpoints/scr_experiments/` output tree; full training
starts independently after smoke. Both modes use the existing default tokenizer.
The epoch budget changes the horizon of linear learning-rate decay. Saved weights
support inference, not an exact optimizer/RNG resume.

All planned runs are complete. See [the final experiment summary](../../part2/EXPERIMENTS.md)
for verified metrics, status qualifications and selected outputs. Scratch job
35002 selected epoch 23; dev job 35419 and test job 36063 used beam-4 + validity.
The original per-run status snapshots and source archives remain available locally.

The standalone commands below remain useful for later debugging or experiments:

```bash
cd /scratch/yc7009/projects/assignment4
mkdir -p part2/runs part2/slurm
source_root=/scratch/yc7009/projects/assignment4
sbatch --job-name=a4-t5-smoke --time=00:20:00 "$source_root/hpc/burst/part2.sbatch" smoke "$source_root"
# Inspect smoke JSON, log, CUDA memory, and sacct exit status before baseline
sbatch "$source_root/hpc/burst/part2.sbatch" baseline "$source_root"
```

The smoke job performs two updates on 32 training examples with long SQL
targets, reloads its checkpoint, and generates/evaluates eight dev queries.
It starts from pretrained weights independently of the full baseline.
Never initialize baseline training from the smoke checkpoint.

Use `squeue -j <jobid>` while running and
`sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS` afterward.
Archive the accounting output alongside the run. After completion download
the run and scheduler log, verify checksums, and retain at least the selected
checkpoint locally. The Slurm job continues independently of Jupyter/browser.

## Assignment evidence and test policy

| Requirement | Saved evidence |
| --- | --- |
| Q4, Tables 1 and 2 | `reports/q4_statistics.json`; train/dev counts before/after preprocessing |
| Q5, Table 3 | `run_config.json`, saved model config, package versions, source manifest, command |
| Q6, Table 4 | `history.jsonl`, `best_metrics.json`, `dev_metrics.json`, `run_info.json` |
| Q6, Table 5 | `reports/dev_evidence.jsonl`; manually inspect actual failures, define categories and count them |
| Q7 | Generate final test SQL/records only after choosing the final configuration using dev |

Q4 token lengths exclude added EOS, decoder start and batch padding; observed
vocabulary counts are distinct T5 token IDs, including any UNK. Raw and
processed numbers may match because the current tokenizer already normalizes
the whitespace stripped by the loader. Record the convention in the write-up.
This first baseline supplies one Table 4 row, not all comparison experiments.

Training does not generate test predictions or submit to Gradescope. After
selecting a final compact run on dev, use `part2/tools/export_test.py` with
`--run-dir <run>` and a new `--output-dir <directory>`. This loads the saved
model, tokenizer and decoding settings, then calls the existing
`train_t5.test_inference()` without running training or dev evaluation.
The original `train_t5.py --eval_only --run_test` workflow remains unchanged.
See the Part 2 README for the full export command.
Validate 432 SQL lines and 432 record/error entries in original test order.
The required output basenames are:

- `t5_ft_experiment_test.sql`
- `t5_ft_experiment_test.pkl` (the supplied helper's `(records, error_msgs)` tuple)

Keep each exported SQL/PKL pair in its experiment's `test/` directory and record
its paths and hashes in `run.json`. For the selected submission only, copy the
pair to `part2/results/` and `part2/records/` under the required filenames and
record the source run in `part2/results/README.md`. Replace both submission
copies together while preserving every experiment's archived pair.
Use the test leaderboard sparingly as required by A4; do not tune using
repeated test scores.
The write-up also needs the repository link and the final checkpoint download
link. Neither a submission nor checkpoint publication is part of baseline
training itself.
