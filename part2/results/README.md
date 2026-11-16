# Part 2 current submission

The current best submission copy is **sql_ops_v1_beam4_valid**, selected by dev
Record F1 (79.7226%). It reuses the epoch-20 checkpoint from `sql_ops_v1_32887`
with beam width 4 and validity filtering. Test export job 33657 completed
successfully. The user's November 15, 2026 Gradescope screenshot reports
Record F1 **77.66**, Record EM **72.92**, SQL Query EM **0.00**, 14/432 SQL
errors and **25/25** performance points. The screenshot and displayed results
are retained in the experiment archive; no submission ID was supplied.

| Submission copy | Permanent experiment archive |
| --- | --- |
| `t5_ft_experiment_test.sql` | `../runs/sql_ops_v1_beam4_valid/test/t5_ft_experiment_test.sql` |
| `../records/t5_ft_experiment_test.pkl` | `../runs/sql_ops_v1_beam4_valid/test/t5_ft_experiment_test.pkl` |

Both pairs have identical SHA256 hashes recorded in
`../runs/sql_ops_v1_beam4_valid/run.json`. They contain 432 predictions in
`data/test.nl` order; the PKL uses the supplied `(records, error_msgs)` tuple
format, with 14 SQL errors and no timeouts. Upload the two files directly;
no submission ZIP is needed.

Keep each experiment's dev/test outputs under the same `runs/<run>/` directory.
When selecting a new best version, copy its SQL and PKL here and into
`../records/` together, verify both hashes, and update this source note.
Historical pairs remain in `baseline_31836/test/` and `sql_ops_v1_32887/test/`.
Baseline submission `448354102` remains recorded in its own run metadata.
