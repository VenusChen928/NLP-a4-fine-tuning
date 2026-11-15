# DS-GA 1011: Homework 4 Part 2

> ### Please start early.

The reproducible baseline workflow, Burst submission steps, artifact layout,
and assignment-table evidence are documented in
[the Part 2 experiment guide](../hpc/burst/PART2.md).

## Workspace layout

`data/`, `records/`, and `results/` come from the course repository.
`tools/` and `runs/` are our additions. Keep the assignment implementation
(`load_data.py`, `train_t5.py`, `t5_utils.py`, `utils.py`, `evaluate.py`) here.

```text
part2/
  data/                              supplied train/dev/test inputs and database
  records/
    ground_truth_dev.pkl             supplied dev answers
    t5_ft_experiment_test.pkl         current Q7 database records
  results/
    t5_ft_experiment_test.sql         current Q7 SQL queries
  tools/                             supporting scripts, not model implementation
    baseline_smoke.py                small GPU interface/checkpoint check
    experiment_report.py            run metadata, Q4 statistics, Q6 dev evidence
    export_test.py                  load a saved run and call test_inference
  runs/
    baseline_31836/
      checkpoint/                    epoch-19 model and its saved tokenizer
      run.json                       configuration, scores, environment, provenance
      history.jsonl                  all 20 epochs for learning-curve analysis
      dev/                           predictions.sql and records.pkl
      analysis/                      q4_statistics.json and dev_evidence.jsonl
      source.zip                     exact training/export source snapshots
      train_eval.log.gz              complete original log, compressed
```

The baseline's best dev Record F1 is **74.5086%** (epoch 19). The unselected
last checkpoint is not needed for this result or Q7; it did not include
optimizer/RNG state for exact training resumption. Only the selected model
is retained locally. The remote original run was not changed.

`run.json` records the actual configuration, selected-epoch metrics, final dev
metrics, package versions, source/data/artifact hashes, and test-export settings.
The original remote status was stale (`running`); scheduler completion was
observed separately. Both facts are retained, without inventing a process exit code.

### Supporting scripts and test export

- `baseline_smoke.py`: a small GPU training/checkpoint check.
- `experiment_report.py`: run metadata, Q4 statistics, and Q6 dev evidence.
- `export_test.py`: loads the compact run's saved model, tokenizer and decoding
  settings, then calls the existing `train_t5.test_inference()` directly.

The exporter reuses `T5Dataset.process_data()` and `test_collate_fn()`; it has no
separate generation loop or SQL execution implementation. The original
`train_t5.py`, `load_data.py`, and `utils.py` implementations are preserved.
There is no `--test_only` mode in `train_t5.py`.

For the first test export of a newly selected compact run, from the repo root:

```bash
python -B part2/tools/export_test.py \
  --run-dir part2/runs/SELECTED_RUN \
  --output-dir part2/runs/SELECTED_RUN/test
```

The output directory must be new. Only the required SQL/PKL pair is written.
The existing baseline has already been exported and submitted; no regeneration
is needed for this code cleanup. Its files and original export provenance remain
unchanged, including the historical exporter in `source.zip`.

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
The processed stage currently applies the baseline's whitespace stripping; update
that stage if a later experiment changes preprocessing before filling Table 2.

### Baseline submission result

Gradescope submission `448354102` displays test Record F1 **70.68**, Record EM
**67.36**, SQL Query EM **0.00**, 63 SQL errors out of 432 queries, and performance
points **25/25**. These are the displayed, rounded values; final grading is by the
course staff. `run.json` records the submission URL and this observation.
The uploaded SQL file was downloaded and matched the local SHA-256 and 432-line
count. The reason for zero test SQL EM remains unresolved; do not replace it
with an inferred score or change predictions just to improve this display.

### Retention rules

Keep one directory per actual experiment. The current Q7 outputs live only in
`results/` and `records/`, with their hashes and generating run recorded in
`run.json`. Before replacing them with a different selected model, move the
old pair into that old run's `test/` directory and update its recorded paths.

Do not keep whole-A4 submission bundles, duplicate test exports, copied input
answers, intermediate status downloads, or setup/preflight directories inside
`runs/`. Useful preflight facts have been merged into `run.json`. Q4's JSON
already includes all CSV rows and the counting convention. Q6's per-example
analysis is retained alongside the raw dev outputs. A compressed source
snapshot is intentional: future code edits must not change a past run's code.
Deployment releases are temporary upload artifacts; remove local copies after
verified deployment and run archival.

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
