# DS-GA 1011 Assignment 4: Fine-tuning Language Models

Completed implementation and experiment records for Assignment 4. See the course
handout for the full problem statement.

Part 1 (BERT fine-tuning for sentiment classification) and Part 2 (T5 fine-tuning for text-to-SQL)
pin different, incompatible package versions, so set up a **separate environment for each part** —
see `part1/README.md` and `part2/README.md` for setup instructions specific to each part.

You will require GPUs for this homework. You can either use the HPC server (see documentation
[here](https://github.com/Athul-R/NYU-HPC-Tutorials)) or use an API service like
[lightning.ai](https://lightning.ai/) or other cloud providers.

## Layout

- `part1/`: completed BERT training, typo transformation and data augmentation
  (`main.py`, `utils.py`). See the [Part 1 guide](part1/README.md) and
  [verified Q1–Q3 submission outputs](part1/runs/README.md).
- `part2/`: completed T5 fine-tuning and training from scratch for text-to-SQL.
  See the [Part 2 guide](part2/README.md), [experiment record](part2/EXPERIMENTS.md),
  and [final submission audit](part2/reports/submission_audit.json).
  Selected SQL and record outputs are under `results/` and `records/`.
  Large run archives and weights remain local or in the linked Google Drive folders.
