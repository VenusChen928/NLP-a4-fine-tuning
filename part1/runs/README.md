# Part 1 verified results

The four linked files are the final Part 1 programming submissions. Each contains
25,000 prediction/label pairs (50,000 lines). Labels and their order match across
all four files. Accuracies below were recomputed directly from these files.

| Task / Burst job | Submission file | Correct / total | Accuracy |
| --- | --- | --- | --- |
| Q1 / `25970` | [out_original.txt](q1_25970/out_original.txt) | 23,268 / 25,000 | 93.072% |
| Q2, p=0.40 / `28708` | [out_transformed.txt](q2_typo_p040_28708/out_transformed.txt) | 22,009 / 25,000 | 88.036% |
| Q3, original / `29911` | [out_augmented_original.txt](q3_typo_p040_29911/out_augmented_original.txt) | 23,228 / 25,000 | 92.912% |
| Q3, transformed / `29911` | [out_augmented_transformed.txt](q3_typo_p040_29911/out_augmented_transformed.txt) | 22,262 / 25,000 | 89.048% |

Q1 exceeds 91%. Q2 decreases accuracy by **5.036 percentage points**, exceeding
the required four-point drop. Q3 improves transformed-test accuracy by **1.012
percentage points** over Q2 and decreases original-test accuracy by **0.160
percentage points** relative to Q1. Q3 asks for both evaluations and analysis;
it specifies no minimum improvement threshold. These are single-run comparisons,
not a statistical significance claim.

## Run provenance

All three formal jobs completed on Burst with Slurm `COMPLETED` and exit code
`0:0`. Q1 and Q3 started from `bert-base-cased`, using learning rate `5e-5`,
3 epochs, batch size 8 and training seed 0. Q2 evaluated the saved Q1 model.
Q2/Q3 use typo probability 0.40 and transform seed 0. Q3 appends 5,000 transformed
training examples sampled with seed 42, giving 30,000 examples and 11,250
optimizer steps, versus Q1's 9,375 steps.

| Job | Source provenance | Elapsed |
| --- | --- | --- |
| Q1 `25970` | `7a4e18756e6409718b47b6290b7a715ebfbcf995` | 00:34:32 |
| Q2 `28708` | Remote HEAD was `7a4e187` with uncommitted transformation changes; exact source hashes are below. | 00:05:24 |
| Q3 `29911` | `f54ecc22f0b3668ab3f6992e970fea275fc0becb` | 00:43:09 |

The Q2 source hashes recorded during evaluation were:

```text
main.py  f31a03b1c9ca79e7954f4f2535656d76e0f5fc66d05fae978d6143a0acf035d5
utils.py ec017074afa2efd0cde86e26be9b05fbd52f52bfce39297a1b59a2800ed818c7
```

Q3's source hashes, also used by the current Burst script guards, were:

```text
main.py  00236c763967f90972eed123de60b358cb693a360f575a401a779c4551ef8431
utils.py 9883a560434c7d1742cbe49bb7c1dde35c35448d556ce21744d18f0f2bc9687c
```

The current Q1 training/evaluation functions have the same Python AST as Q1's
source. The current transformation has the same Python AST as the Q2
implementation committed in `dc3cb5b`; later formatting changes did not alter it.

Earlier local experiments remain preserved: Q1's MPS debug run achieved
887/1,000 (88.7%), and Q2 p=0.30 achieved 89.788% (a 3.284-point drop).
They are not the final submission files. Both p=0.30 and p=0.40 transformation
debug commands printed five example pairs successfully.

## File integrity

SHA-256 checksums of the submitted files:

```text
59a4e340eed7357c18af6211485b317e34951319600e1c09b0ac9d5ab3cedddc  q1_25970/out_original.txt
d46dcc912a478f7ba8f1696c384cd3ef493d754d35cb1e75258ebadceec9a561  q2_typo_p040_28708/out_transformed.txt
b46a62b117bf53cdaf81bcced82124860d9cf394b318d642a16e57f97640982c  q3_typo_p040_29911/out_augmented_original.txt
87c7df9fde2a906fa727c273555cc4ee5555b0740311f3723ec547cf2823fb7f  q3_typo_p040_29911/out_augmented_transformed.txt
```

The local run directories retain checkpoints, logs and detailed `run_info.json`
records. These remain excluded from Git, together with earlier/debug runs.
The corresponding formal-run archives are also on Burst under
`/scratch/yc7009/projects/assignment4/runs/`.
