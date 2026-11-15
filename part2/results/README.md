# Part 2 submission outputs

`t5_ft_experiment_test.sql` contains the baseline's 432 test predictions in
the order of `data/test.nl`. Its companion database records are saved as
`../records/t5_ft_experiment_test.pkl`, using the supplied `(records, error_msgs)`
tuple format. Both are ordinary files ready to upload to Gradescope.

The generating model is job 31836's selected checkpoint (epoch 19).
`../runs/baseline_31836/run.json` records the export settings and file hashes;
`../runs/baseline_31836/checkpoint/` contains the model and tokenizer.
There is no second copy of these outputs inside the run directory.

Before replacing this submission with another dev-selected model, move the old
SQL and PKL pair into that run's `test/` folder and update its paths in `run.json`.
