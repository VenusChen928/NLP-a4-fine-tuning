# Part 2 baseline experiments on Burst

Use the course Burst Open OnDemand portal, not Torch. Local source is the
source of truth. Each submitted job receives a frozen source release and a
new directory named with its Slurm job ID. Preserve successful and failed
run records; never reuse a run directory for another experiment.

## Storage

Remote project root: `/scratch/yc7009/projects/assignment4`.
Local project root: `/Users/venus/Desktop/NLP/Assignments/assignment4`.

```text
part2/
  data/                              shared supplied data and database
  records/ground_truth_dev.pkl       shared supplied dev records
  releases/<timestamp>_<hash>/       frozen source and release manifest
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

## Release and submission

Build a new local release after validating changes:

```bash
python -B hpc/burst/package_part2.py
```

Upload its archive through Burst OOD into `part2/releases/`, verify the archive
SHA256 printed by the packager, and extract it there. Do not edit a release
after submission. The job verifies the supplied data against local checksums.

First inspect current course partition limits and the prepared Part 2
environment. The planned formal allocation is one A100, four CPU threads,
32 GB RAM and four hours. Override the time limit to 20 minutes for smoke.
These are requested limits, not an estimate of elapsed training time.

The preferred first-baseline command shares one A100 allocation between a
20-minute smoke gate and independent full training, avoiding a second cloud
node startup. It writes `runs/smoke_<jobid>/` and `runs/baseline_<jobid>/`.
Any failed smoke exit, non-finite metric, failed checkpoint round trip or
incomplete report prevents baseline training. The four-hour allocation limit
covers both stages; the smoke's internal limit excludes Slurm queue time.

```bash
sbatch "$release/hpc/burst/part2_pipeline.sbatch" "$release"
```

The standalone commands below remain useful for later debugging or experiments:

```bash
cd /scratch/yc7009/projects/assignment4
mkdir -p part2/releases part2/runs part2/slurm
release=/scratch/yc7009/projects/assignment4/part2/releases/<release-id>
sbatch --job-name=a4-t5-smoke --time=00:20:00 "$release/hpc/burst/part2.sbatch" smoke "$release"
# Inspect smoke JSON, log, CUDA memory, and sacct exit status before baseline.
sbatch "$release/hpc/burst/part2.sbatch" baseline "$release"
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
| Q4, Tables 1 and 2 | `reports/q4_statistics.csv` and JSON; train/dev counts before/after preprocessing |
| Q5, Table 3 | `run_config.json`, saved model config, package versions, source manifest, command |
| Q6, Table 4 | `history.jsonl`, `best_metrics.json`, `dev_metrics.json`, `reports/baseline_summary.json` |
| Q6, Table 5 | `reports/dev_evidence.jsonl`; manually inspect actual failures, define categories and count them |
| Q7 | Generate final test SQL/records only after choosing the final configuration using dev |

Q4 token lengths exclude added EOS, decoder start and batch padding; observed
vocabulary counts are distinct T5 token IDs, including any UNK. Raw and
processed numbers may match because the current tokenizer already normalizes
the whitespace stripped by the loader. Record the convention in the write-up.
This first baseline supplies one Table 4 row, not all comparison experiments.

Training does not generate test predictions or submit to Gradescope. After
selecting a final run, generate its test files from a separate export working
directory using the frozen source, saved best checkpoint, and the same
tokenizer and decoding configuration. Pass `--eval_only --run_test` and the
original experiment name. Validate 432 SQL lines and 432 record/error entries
in original test order. The required output basenames are:

- `t5_ft_experiment_test.sql`
- `t5_ft_experiment_test.pkl` (the supplied helper's `(records, error_msgs)` tuple)

Keep that export and its configuration tied to the selected run. Use the test
leaderboard sparingly as required by A4; do not tune using repeated test scores.
The write-up also needs the repository link and the final checkpoint download
link. Neither a submission nor checkpoint publication is part of baseline
training itself.
