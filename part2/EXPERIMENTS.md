# Final Part 2 experiments

Verified on 2026-11-18. All planned training and decoding experiments have finished.
No additional training is planned. Dev metrics below were recomputed from saved
SQL and record files using the supplied scoring functions. Test scores are
Gradescope results reported by the user; test references are unavailable locally.
Full-precision metrics, artifact paths and SHA-256 hashes are in
[final_results.json](reports/final_results.json).

## Fine-tuning

All runs start independently from pretrained T5-small. `Operators` uses the
reversible `<`/`<=` representation; other data and architecture settings are shared.

| Training run | Selected epoch | Decoding | SQL EM % | Record EM % | Record F1 % |
| --- | ---: | --- | ---: | ---: | ---: |
| Baseline, 20 epochs, 3e-4 (`31836`) | 19 | Greedy | 55.36 | 70.17 | 74.51 |
| Operators, 20 epochs, 3e-4 (`32887`) | 20 | Greedy | 57.08 | 71.89 | 76.15 |
| Same checkpoint | 20 | Beam-4 | 57.73 | 74.46 | 78.67 |
| Same checkpoint | 20 | Beam-4 + valid | 57.94 | 75.11 | 79.72 |
| Same checkpoint | 20 | Beam-8 + valid | 57.94 | 74.68 | 79.15 |
| Operators, 30 epochs, 1e-4 (`33954`) | 30 | Greedy | 46.14 | 62.45 | 70.20 |
| Same checkpoint | 30 | Beam-4 + valid | 48.71 | 66.09 | 75.02 |
| Operators, 30 epochs, 3e-4 (`33756`) | 29 | Greedy | 68.03 | 82.83 | 84.47 |
| **Selected checkpoint** | **29** | **Beam-4 + valid** | **69.10** | **84.55** | **86.46** |

The 30-epoch budget also extends linear learning-rate decay; this comparison
cannot isolate the effect of extra epochs. Beam-8 did not improve the 20-epoch
checkpoint, and reducing the learning rate to 1e-4 underperformed at 30 epochs.
The selected model's test F1 is **83.37%**, Record EM **80.79%**, SQL EM **0.00%**,
with 16 execution errors among 432 saved predictions.

## Training from scratch

These runs use random T5-small weights, the default T5 tokenizer, all 4,225
training pairs, operator processing, batch size 16, FP32, AdamW with zero weight
decay, one warmup epoch and linear decay. All parameters are trained. Seed 42
and the architecture are unchanged. Each run starts independently; the 25-epoch
run does not resume the 20-epoch checkpoint.

| Training run | Selected epoch | Decoding | SQL EM % | Record EM % | Record F1 % | SQL errors |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Pilot, 10 epochs, 1e-3 (`34503`) | 1 | Greedy | 0.00 | 11.80 | 11.80 | 466 |
| 20 epochs, 3e-4 (`34933`) | 20 | Greedy | 41.85 | 55.36 | 58.61 | 33 |
| Same checkpoint | 20 | Beam-4 + valid | 43.99 | 56.87 | 60.14 | 10 |
| 25 epochs, 3e-4 (`35002`) | 23 | Greedy | 44.42 | 58.37 | 61.69 | 31 |
| **Selected checkpoint** | **23** | **Beam-4 + valid** | **45.06** | **57.94** | **61.88** | **7** |

The pilot completed ten epochs but generated invalid SQL for every dev example.
Its 11.80% F1 reflects 55 empty gold results, not successful queries. Reducing LR
and extending training improved generation, though both changed together.
The 20- and 25-epoch runs use minimum 20 epochs and patience 20, allowing time
for SQL generation to emerge. Checkpoints are selected by greedy dev Record F1,
retaining the earliest exact tie. All 25 epochs finished; epoch 23 outperformed
epoch 25 on F1 despite the later epoch's lower dev loss.

Beam-4 + validity on epoch 23 raised F1 by 0.1960 percentage points and reduced
saved execution errors from 31 to 7. Record EM fell from 58.37% to 57.94%, so the
gain is specific to the selection metric. Fine-tuning remains 24.57 F1 points
higher with the same final decoding. Different training budgets prevent treating
this as a fully controlled estimate of pretraining alone.

| Scratch test export | Record F1 % | Record EM % | SQL EM % | SQL errors |
| --- | ---: | ---: | ---: | ---: |
| Epoch 20, greedy, job 34979 (historical) | 51.60 | 47.45 | 0.00 | 31/432 |
| **Epoch 23, beam-4 + valid, job 36063 (selected)** | **53.39** | **50.23** | **0.00** | **8/432** |

The latest test score was supplied by the user on November 18. Its eight errors
include two saved timeouts. Both test versions remain in their original run
archives. The selected SQL/PKL pair is also copied to the required top-level paths.
The reported 0.00 SQL EM is retained as reported; dev results and record metrics
should not be replaced by an inferred test SQL score.

## Error analysis and provenance

[Scratch error analysis](reports/scratch_error_analysis.json) records the protocol,
category indices and representative examples. All 196 Record EM failures and two
additional execution failures with empty-gold credit were reviewed. Three
executable ASD reference anomalies are isolated. The remaining 195 cases have one
primary label: conditions/logical structure 116, temporal interpretation 31,
entity/type mapping 21, intent/projection/aggregation 20, and execution 7.
Semantic symptoms can overlap; counts use all 466 dev examples. Causes are
hypotheses. The full annotated SQL differences remain in the selected run's
`analysis/beam4_valid_error_analysis.json`.

The fine-tuned model's corresponding full analysis remains in
`runs/sql_ops_e30_lr3e4_33756/analysis/beam4_valid_error_analysis.json`.

Job 34933 completed all 20 training epochs, but its repeated final evaluation and
reporting were interrupted by infrastructure/time-limit errors. Later dev and
test exports succeeded. Its original `run_info.json` is a historical snapshot;
this does not imply training is still running. Job 35002 and follow-up jobs
35419 (dev) and 36063 (test) completed with exit code 0.

The scratch tokenizer files omitted from local transfers were restored from the
identical saved default T5 tokenizer. The pilot and 25-epoch run's original
manifests confirm matching hashes; each run has `tokenizer_restoration.json`.
The shared [scratch checkpoint folder](https://drive.google.com/drive/folders/1GtJ2IgVI3lc4vqJf8JoopUNmcgOVoSiR)
contains standard Hugging Face shards, tokenizer, configuration, loading notes
and hashes. Every reloaded weight tensor equals the original epoch-23 checkpoint.

## Final review

The exporter now distinguishes compact archives from native Burst archives that
also contain `run.json`. Eighteen regression checks passed, including this case,
initialization separation, minimum-epoch selection and required output filenames.
The selected scratch checkpoint and restored tokenizer loaded offline and passed
a finite-loss forward check. No training, generation or SQL execution was repeated.
Local checks used Python 3.12, PyTorch 2.11.0, Transformers 4.40.0 and W&B 0.16.6;
W&B differs from the course pin because the older dependency cannot install under
Python 3.12. Archived GPU package records retain the actual training environment.
