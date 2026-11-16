# DS-GA 1011: Homework 4 Part 2

> ### Please start early.

The reproducible baseline workflow, Burst submission steps, artifact layout,
and assignment-table evidence are documented in
[the Part 2 experiment guide](../hpc/burst/PART2.md).

## Code and experiment layout

The baseline framework stays in `load_data.py`, `train_t5.py`, `t5_utils.py`,
`utils.py` and `evaluate.py`. `sql_processing.py` contains the reversible SQL
operator conversion. Model architecture and training are shared by all current
experiments; beam width and validity filtering are inference settings.

There are three supporting scripts, shared across experiments:

| Script | Purpose |
| --- | --- |
| `tools/export_test.py` | Load a saved checkpoint for dev evaluation or test export |
| `tools/baseline_smoke.py` | Short GPU/environment check used by the Burst launcher |
| `tools/experiment_report.py` | Training metadata, Q4 statistics and Q6 error evidence |

Experiment names belong in `runs/`, rather than in separate Python scripts:

| Run | Model source | SQL processing | Decoding | Status |
| --- | --- | --- | --- | --- |
| `baseline_31836` | Pretrained T5-small, fine-tuned | None | Greedy | Dev F1 74.5086%, epoch 19 |
| `sql_ops_v1_32887` | Pretrained T5-small, fine-tuned | `operators_v1` | Greedy | Dev F1 76.1490%, epoch 20 |
| `sql_ops_v1_beam4` | Checkpoint from `sql_ops_v1_32887` | `operators_v1` | Beam-4 | Dev completed, job 33306 |
| `sql_ops_v1_beam4_valid` | Checkpoint from `sql_ops_v1_32887` | `operators_v1` | Beam-4 + validity | Dev 33307, test 33657; current best |

A completed training run keeps its selected `checkpoint/`, `run.json`,
`history.jsonl`, `dev/`, `test/`, `analysis/`, `source.zip` and compressed log.
The historical scheduler/status details remain in that run's `run.json`.
A decoding experiment follows the same one-directory-per-experiment layout:
`dev/` holds `predictions.sql` and `records.pkl`, and `test/` holds that version's
required SQL/PKL pair when exported. The root `run.json` records the shared
settings, `dev_evaluation` and `test_export`; `source.zip`,
`dev_inference.log.gz` and `test_inference.log.gz` retain the source and logs.
The checkpoint is referenced from the training run without another model copy.
The paired comparison is in
[`runs/decoding_comparison.json`](runs/decoding_comparison.json).

### Dev decoding experiments

Use the existing export entry point with `--split dev`. The command below uses
the completed beam-4 + validity experiment's settings and the current directory
layout; choose a new output directory to repeat it:

```bash
python -B part2/tools/export_test.py \
  --run-dir part2/runs/sql_ops_v1_32887 \
  --output-dir part2/runs/sql_ops_v1_beam4_valid/dev \
  --split dev --num-beams 4 --valid --batch-size 16 --threads 4
```

For each question, beam search returns four candidates in sequence-score order.
`--valid` selects the first candidate that executes successfully as a read-only
SELECT/WITH query. Empty results are valid. Each check has its own five-second
budget, adjustable with `--query-timeout`; if every candidate fails, the first
candidate is kept. The selected SQL is evaluated with the original record
executor and metric functions, using the same evaluation policy as the baseline.

Without `--valid`, the same script runs ordinary beam search and keeps its
first candidate. This was run as the ablation in `sql_ops_v1_beam4/dev/`.
Both jobs used batch size 16, seed 42, a 512-token generation limit and an A100
40GB GPU. The source checkpoint supplies the tokenizer and preprocessing mode.
The exporter accepts both the compact local archive and the original Burst
training directory. CUDA is used when available, otherwise CPU.

### Dev decoding results

All three rows use the epoch-20 checkpoint from `sql_ops_v1_32887` and the same
466 dev examples. Greedy results are the existing saved outputs; no training
or greedy generation was repeated.

| Decoding | Record F1 (%) | Record EM (%) | SQL EM (%) | SQL errors | Timeouts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 76.1490 | 71.8884 | 57.0815 | 47 | 3 |
| Beam-4 | 78.6717 | 74.4635 | 57.7253 | 35 | 2 |
| Beam-4 + validity | 79.7226 | 75.1073 | 57.9399 | 14 | 1 |

Beam-4 gains 2.5227 F1 percentage points over the saved greedy result. Adding
validity filtering gains a further 1.0508 points and resolves 21 execution
errors without introducing any new ones. Filtering changes 25 SQL predictions:
10 examples improve in Record F1, 3 worsen and 12 keep the same score.

`beam4_valid` is the preferred configuration on this dev set. Its generation
and filtering loop took 432 seconds, versus 199 seconds for plain beam-4
(about 2.17 times as long). These times exclude final SQL scoring and model
loading. Total batch runtimes were 9m32s and 5m38s, respectively. Both jobs
completed with exit code 0. The earlier one-hour requests, 33301 and 33302,
were canceled before execution and replaced with 20-minute requests for
backfill scheduling.

The comparison retains the original evaluation policy. At dev line 353,
greedy and beam-4 generated identical SQL, but the saved greedy execution
timed out and the beam-4 execution succeeded. This accounts for about 0.2146
points of the observed greedy-to-beam gain; about 2.3081 points are associated
with changed SQL predictions. There are no record differences for unchanged
SQL between the two beam variants.

Validity does not guarantee semantic correctness. Two of its three F1
regressions, lines 130 and 134, lose the original metric's credit for an SQL
error returning an empty list that happened to match empty gold records.
Line 188 replaces a previously correct result with a worse candidate.
Candidate rejection reasons were not logged; the five-second check can reject
SQL that succeeds under the longer final scoring budget.

### Next experiments

The selected dev output has 116 Record EM failures: 104 queries execute but
return the wrong records, and 12 fail execution. There are 14 execution errors
in total; two receive Record EM credit because empty predictions match empty
gold records under the supplied metric. The longest processed gold SQL is 503
tokens including special tokens, below the current 512-token generation limit.
These checks favor improving the learned mapping from questions to SQL.

The current training run reaches its best dev F1 and lowest dev loss at epoch
20, the configured limit. The next training comparison is proposed below;
neither run has been launched:

| Proposed configuration | Maximum epochs | Learning rate | Purpose |
| --- | ---: | ---: | --- |
| `sql_ops_e30_lr3e4` | 30 | 3e-4 | Test a longer training and linear-decay budget |
| `sql_ops_e30_lr1e4` | 30 | 1e-4 | Isolate the learning-rate change against the first run |

Both start from the same pretrained T5-small and retain `operators_v1`, seed
42, batch size 16, weight decay 0, one warmup epoch, linear scheduling and
early-stopping patience 5. Keep the existing greedy dev checkpoint-selection
rule, then evaluate each selected checkpoint with the same `beam4_valid`
settings. Extending the epoch budget also stretches linear decay, so the first
comparison tests that combined training schedule rather than epochs alone.
See the [Transformers scheduler documentation](https://huggingface.co/docs/transformers/v4.40.0/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup).

A single `beam8_valid` dev run on the current checkpoint is an optional cheap
search-width check. Further data processing should follow inspection of the
remaining executable mistakes, focusing on entity linking, time conditions
and schema relations. Any normalization must apply consistently to train/dev
and test, with reversible SQL values and no rules fitted to test answers.
Architecture changes and incremental grammar constraints such as
[PICARD](https://aclanthology.org/2021.emnlp-main.779/) are later experiments;
the present errors are predominantly executable queries with wrong results.
Use dev for selection, and repeat the winning training configuration with an
additional seed if its gain is small before another final test export.

### Test export

The original export command still works after selecting a final run:

```bash
python -B part2/tools/export_test.py \
  --run-dir part2/runs/SELECTED_TRAINING_RUN \
  --output-dir part2/runs/SELECTED_TRAINING_RUN/test
```

The default split is test. The script writes the required
`t5_ft_experiment_test.sql` and `t5_ft_experiment_test.pkl` pair. To export a
selected decoding variant, pass its `--num-beams` and `--valid` settings again,
using its training run as `--run-dir` and a new output directory. The output
directory must always be new, so archived predictions cannot be overwritten.

Both existing trained runs already have archived test pairs. The SQL processing
run's pair contains 432 predictions and 46 execution errors. These historical
pairs stay in their own run directories.

The dev-selected `beam4_valid` configuration keeps both dev and test outputs in
[`runs/sql_ops_v1_beam4_valid/`](runs/sql_ops_v1_beam4_valid/).
Burst job **33657** completed with exit code 0 in **7m57s** on an A100 40GB.
The existing exporter and epoch-20 checkpoint were reused. The command with
the current directory layout is:

```bash
python -B part2/tools/export_test.py \
  --run-dir part2/runs/sql_ops_v1_32887 \
  --output-dir part2/runs/sql_ops_v1_beam4_valid/test \
  --split test --num-beams 4 --valid --query-timeout 5 \
  --batch-size 16 --threads 4
```

The archive contains 432 SQL predictions and 432 paired records/error messages
in test-input order, with **14 SQL errors and 0 timeouts**. The saved greedy
test outputs have 46 errors. There are 72 changed SQL predictions, 33 resolved
errors and 1 newly introduced error; unchanged SQL has identical saved records.
These execution checks are distinct from the Gradescope accuracy scores below.
Plain beam-4 was evaluated on dev for the ablation and was not
exported on test; the test configuration was selected using dev results.

The archived pair stays in this experiment's `test/` directory. The current
best pair is also copied to `results/t5_ft_experiment_test.sql` and
`records/t5_ft_experiment_test.pkl`, with matching hashes recorded in `run.json`
and its source identified in `results/README.md`. Upload these two files
directly; no submission ZIP is needed. The user supplied this version's
Gradescope result on November 15, 2026. The original Burst commands remain in the corresponding
`dev_evaluation` and `test_export` metadata for provenance.

| Test configuration | Record F1 (%) | Record EM (%) | SQL errors / 432 |
| --- | ---: | ---: | ---: |
| Baseline + greedy | 70.68 | 67.36 | 63 |
| SQL operators + greedy | 73.45 | 69.21 | 46 |
| SQL operators + beam-4 + validity | **77.66** | **72.92** | **14** |

The latest screenshot reports 25/25 performance points and SQL Query EM 0.00.
It improves Record F1 by 4.21 percentage points and Record EM by 3.71 points
over the SQL-operator greedy submission. Values are rounded as displayed by
Gradescope. The screenshot is retained in the selected run's `analysis/`
directory, with its result recorded in `run.json`; it does not show a
submission ID or verify uploaded-file hashes. SQL Query EM remains 0.00
despite nonzero local dev SQL EM; the reason is unresolved. Do not treat that
display alone as evidence of zero semantic accuracy or tune against it.

### Reproducing Table 1

From the repository root, print statistics without creating another report:

```bash
python -B part2/tools/experiment_report.py stats \
  --data-folder part2/data \
  --tokenizer part2/runs/baseline_31836/checkpoint
```

The script counts lines in `train.nl`, `train.sql`, `dev.nl`, and `dev.sql`,
tokenizes without truncation/padding/added special tokens, computes mean token
counts, and counts distinct observed token IDs separately for each split and
text type. UNK is included. `--output <path.json>` optionally saves the result.
The baseline's JSON is already in `analysis/q4_statistics.json`. Training
hyperparameters do not affect Table 1 when the raw files and tokenizer stay fixed.
For the SQL processing run, `analysis/q4_statistics.json` also records the
`operators_v1` processed stage for Table 2. To reproduce it, use that run's
checkpoint and add `--sql-preprocessing operators_v1` to the command above.

### Baseline submission result

Gradescope submission `448354102` displays test Record F1 **70.68**, Record EM
**67.36**, SQL Query EM **0.00**, 63 SQL errors out of 432 queries, and performance
points **25/25**. These are the displayed, rounded values; final grading is by the
course staff. `run.json` records the submission URL and this observation.
The uploaded SQL file was downloaded and matched the local SHA-256 and 432-line
count. The reason for zero test SQL EM remains unresolved; do not replace it
with an inferred score or change predictions just to improve this display.

### Retention rules

Keep one directory per experiment, with `dev/` and `test/` as subdirectories;
do not create separate experiment roots for evaluation and test. Preserve the
baseline Q7 pair in
`runs/baseline_31836/test/`, matching the SQL processing run's layout. Every
experiment's `test/` directory owns its archived SQL/PKL pair, with paths and
hashes in `run.json`; do not overwrite earlier predictions. Failed submission
attempts can be recorded in the actual experiment's metadata without a separate
run directory.

The top-level `results/` and `records/` hold the current best run's test pair,
under the course-required filenames. They currently contain
`sql_ops_v1_beam4_valid`. When selecting another best run, copy both files from
that run together,
verify both hashes and update `results/README.md` with its source.
Keep the archived pair in that run. `records/ground_truth_dev.pkl` remains the
shared, course-supplied dev reference; dev predictions belong in each run's
`dev/` directory.

Do not create submission ZIPs. Do not keep whole-A4 submission bundles, extra
test-export copies, copied input answers, intermediate status downloads, or
setup/preflight directories inside
`runs/`. Useful preflight facts have been merged into `run.json`. Q4's JSON
already includes all CSV rows and the counting convention. Q6's per-example
analysis is retained alongside the raw dev outputs. A compressed source
snapshot is intentional: future code edits must not change a past run's code.
Direct script synchronization is the default; a local `releases/` directory is
not required. Remove unused packaging artifacts after confirming that actual
runs retain their own source/provenance. Keep each completed run's `source.zip`.

## Environment

It's highly recommended to use a virtual environment (e.g. conda, venv) for this assignment. Note that Part 2 pins different package versions than Part 1, so use a separate environment from Part 1.

Example of virtual environment creation using conda:
```bash
conda create -n dsga1011-hw4-part2 python=3.11
conda activate dsga1011-hw4-part2
python -m pip install -r requirements.txt
```

You can refer to the [HPC Tutorials](https://github.com/Athul-R/NYU-HPC-Tutorials) for more information on how to use the NYU HPC.

## Evaluation commands

From `part2/`, evaluate the saved baseline dev outputs without regenerating SQL:
```bash
python evaluate.py \
  --predicted_sql runs/baseline_31836/dev/predictions.sql \
  --predicted_records runs/baseline_31836/dev/records.pkl \
  --development_sql data/dev.sql \
  --development_records records/ground_truth_dev.pkl
```

## Submission

You need to submit your test SQL queries and their associated SQL records. Please only submit your final files corresponding to the test set.

For SQL queries, ensure that the name of the submission files (in the `results/` subfolder) are:
- `t5_ft_experiment_test.sql` (for extra credit `t5_ft_experiment_ec_test.sql`)

For database records, ensure that the name of the submission files (in the `records/` subfolder) are:
- `t5_ft_experiment_test.pkl` (for extra credit `t5_ft_experiment_ec_test.pkl`)

⚠️ Note that the predictions in each line of the .sql file or in each index of the list within the .pkl file must match each natural language query in `data/test.nl` in the order they appear.
