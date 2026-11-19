# Selected Part 2 test submissions

Updated 2026-11-18. Only these two final SQL/PKL pairs are selected. Each contains
432 predictions in the original test order. Every top-level file was checked
byte-for-byte against its archived source.

| Submission | Run / selected epoch | Decoding | Test F1 | Test EM |
| --- | --- | --- | ---: | ---: |
| Q7 | `sql_ops_e30_lr3e4_33756`, epoch 29 | Beam-4 + valid | 83.37% | 80.79% |
| Extra credit | `scratch_ec_e25_lr3e4_35002`, epoch 23 | Beam-4 + valid | 53.39% | 50.23% |

Both reported test SQL EM scores are 0.00%. The live Gradescope results and all
eight files in coding submission `449497755` were verified on November 18.
The downloaded submission files match the selected local files byte-for-byte.

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

- [Fine-tuned epoch-29 archive](https://drive.google.com/drive/folders/1biyHXenO-PRGijD6IdsZVrojRKwCT1XP)
- [Scratch epoch-23 checkpoint](https://drive.google.com/drive/folders/1GtJ2IgVI3lc4vqJf8JoopUNmcgOVoSiR)

The scratch folder contains all Hugging Face shards and their index, tokenizer,
configuration and loading instructions. It can be loaded from one local folder
with `from_pretrained`. Only the models used for the submitted outputs are shared;
other experiments remain in the local archives.

The Q7 archive stores the model inside its `checkpoint/` subfolder. Downloaded
Drive files were checked against the original checkpoints. All 134 tensors in
the downloaded scratch model exactly match the original epoch-23 model.
See [the final submission audit](../reports/submission_audit.json) for evidence.
