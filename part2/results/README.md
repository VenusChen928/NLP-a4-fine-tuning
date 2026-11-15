# Part 2 current submission

The current submission copy is **baseline_31836**, selected epoch 19, already
submitted to Gradescope as submission `448354102`.

| Submission copy | Permanent experiment archive |
| --- | --- |
| `t5_ft_experiment_test.sql` | `../runs/baseline_31836/test/t5_ft_experiment_test.sql` |
| `../records/t5_ft_experiment_test.pkl` | `../runs/baseline_31836/test/t5_ft_experiment_test.pkl` |

Both pairs have identical SHA256 hashes recorded in
`../runs/baseline_31836/run.json`. They contain 432 predictions in `data/test.nl`
order; the PKL uses the supplied `(records, error_msgs)` tuple format.

Archive every experiment's outputs in its own `runs/<run>/test/` directory.
To stage a different submission, copy its SQL and PKL here and into `../records/`
together, verify both hashes, and update this source note. Keep all archived
experiment outputs. The newer `sql_ops_v1_32887` pair is archived in that run's
`test/` directory and has not replaced the current submission copy.
