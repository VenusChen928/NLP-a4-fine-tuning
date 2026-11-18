# Selected Part 2 test submissions

Updated 2026-11-18. Only these two final SQL/PKL pairs are selected. Each contains
432 predictions in the original test order. Every top-level file was checked
byte-for-byte against its archived source.

| Submission | Run / selected epoch | Decoding | Test F1 | Test EM |
| --- | --- | --- | ---: | ---: |
| Q7 | `sql_ops_e30_lr3e4_33756`, epoch 29 | Beam-4 + valid | 83.37% | 80.79% |
| Extra credit | `scratch_ec_e25_lr3e4_35002`, epoch 23 | Beam-4 + valid | 53.39% | 50.23% |

Both reported test SQL EM scores are 0.00%. Gradescope scores are user-reported;
local verification checks saved file counts, execution errors and hashes.

- Q7 SQL: `results/t5_ft_experiment_test.sql`
- Q7 records: `records/t5_ft_experiment_test.pkl`
- Extra-credit SQL: `results/t5_ft_experiment_ec_test.sql`
- Extra-credit records: `records/t5_ft_experiment_ec_test.pkl`

Paths are relative to `part2/`. The Q7 source pair is under the selected run's
`test/`; the extra-credit source pair is under `test/beam4_valid/` (job 36063).
The historical epoch-20 scratch pair remains under
`runs/scratch_ec_e20_lr3e4_34933/test/` and is no longer the selected submission.

[Exact source paths and SHA-256 hashes](../reports/final_results.json) are recorded
with the final results. The selected files have 16 Q7 execution errors and 8
extra-credit errors, including two extra-credit timeouts.

- [Fine-tuned checkpoint](https://drive.google.com/drive/folders/1QnYX8-S_jJCA4kNl6YQQJ9IN0Y2IZDas)
- [Scratch epoch-23 checkpoint](https://drive.google.com/drive/folders/1GtJ2IgVI3lc4vqJf8JoopUNmcgOVoSiR)

The scratch folder contains all Hugging Face shards and their index, tokenizer,
configuration and loading instructions. It can be loaded from one local folder
with `from_pretrained`. Only the models used for the submitted outputs are shared;
other experiments remain in the local archives.
