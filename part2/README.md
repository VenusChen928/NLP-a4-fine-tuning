# DS-GA 1011: Homework 4 Part 2

All planned fine-tuning and scratch experiments are complete. The final fine-tuned
model is epoch 29 of the 30-epoch, LR `3e-4` run. The final scratch model is epoch 23
of the 25-epoch, LR `3e-4` run. Both use beam-4 with execution validity filtering.

- [Experiment results and decisions](EXPERIMENTS.md)
- [Selected submission files](results/README.md)
- [Burst execution guide](../hpc/burst/PART2.md)
- [Machine-readable results and hashes](reports/final_results.json)
- [Scratch error analysis](reports/scratch_error_analysis.json)
- [Final repository, Gradescope and Drive audit](reports/submission_audit.json)

## Code and storage

| Path | Purpose |
| --- | --- |
| `load_data.py`, `train_t5.py`, `t5_utils.py` | Shared data, training and initialization code |
| `sql_processing.py` | Reversible SQL operator representation |
| `utils.py`, `evaluate.py` | Supplied record metrics and saved-output evaluation |
| `tools/export_test.py` | Saved-checkpoint dev evaluation or test export |
| `tools/baseline_smoke.py` | Short GPU smoke check before full training |
| `tools/experiment_report.py` | Configuration, data statistics and per-example evidence |
| `tools/check_training_modes.py` | Regression checks for initialization, selection and archive paths |
| `data/`, `records/ground_truth_dev.pkl` | Shared assignment data and dev references |
| `runs/<experiment>/` | Local checkpoints, histories, outputs, logs and source snapshots; ignored by Git |
| `reports/` | Small final summaries and analysis retained in Git |
| `results/`, `records/` | Selected test SQL and PKL copies under required filenames |

Fine-tuned archives use `checkpoint/`, `dev/`, optional `test/`, `run.json`,
`history.jsonl`, `analysis/`, `source.zip` and compressed logs. Decoding-only runs
reference their training checkpoint without copying weights.

Scratch archives preserve the native Burst layout:
`checkpoints/scr_experiments/<name>/best/` holds weights; its parent holds the
training configuration, history and tokenizer. `results/` and `records/` hold
greedy dev outputs; `dev/beam4_valid/` and `test/` hold subsequent evaluations.
The exporter accepts both layouts, including native runs with a root `run.json`.

Keep each run's test pair in its own `test/` directory. Promote both selected
files together; never overwrite the archived pair. Historical status snapshots
and recommendations are preserved in each run, while `EXPERIMENTS.md` records
the final selection. Raw scratch logs were losslessly compressed to `.log.gz`.
The cleanup inventory and superseded documentation are in
`runs/_maintenance_20261118/`. Checkpoints, histories and source archives remain intact.

## Environment and validation

Use a separate Python 3.11 environment for Part 2; its pins differ from Part 1.
From this directory:

```bash
python -m pip install -r requirements.txt
```

From the repository root, in the Part 2 environment:

```bash
python -B -m pytest -q -p no:cacheprovider part2/tools/check_training_modes.py
```

These local checks use small models and temporary files. The Burst pipeline
separately checks the allocated GPU and actual T5-small checkpoint. Training
uses either `--finetune` (pretrained weights) or `--scratch` (random weights with
the same T5-small architecture). Scratch retains the existing T5 tokenizer.

## Saved-output evaluation

From `part2/`, this computes the selected scratch dev F1 from the archived SQL
and records without generating or executing SQL again:

```bash
python -B evaluate.py \
  --predicted_sql runs/scratch_ec_e25_lr3e4_35002/dev/beam4_valid/predictions.sql \
  --predicted_records runs/scratch_ec_e25_lr3e4_35002/dev/beam4_valid/records.pkl \
  --development_sql data/dev.sql \
  --development_records records/ground_truth_dev.pkl
```

The expected F1 is `0.618842533473483`. Execution errors are saved alongside
records; empty predictions can receive credit when gold records are also empty.

## Checkpoint export

Export only after choosing a checkpoint and decoding settings on dev. From the
repository root, use a **new** output directory:

```bash
python -B part2/tools/export_test.py \
  --run-dir part2/runs/scratch_ec_e25_lr3e4_35002 \
  --output-dir part2/runs/scratch_ec_e25_lr3e4_35002/test/new_export \
  --split test --num-beams 4 --valid --query-timeout 5 \
  --batch-size 16 --threads 4
```

Use `--split dev` for dev evaluation. Validity filtering checks candidates in
sequence-score order as read-only SELECT/WITH queries. Each candidate has a
five-second budget; empty results are valid. If every candidate fails, the
highest-scoring candidate is retained. Final scoring uses the original shared
batch deadline, so validity checks and final execution can disagree.

The exporter loads the saved tokenizer and processing mode. It writes
`t5_ft_experiment_test.sql/.pkl` for fine-tuning or
`t5_ft_experiment_ec_test.sql/.pkl` for scratch. Each pair must contain 432
examples in `data/test.nl` order. Existing final exports are already archived;
rerunning is unnecessary for submission.
