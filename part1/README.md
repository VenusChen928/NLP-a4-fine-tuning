# DS-GA 1011: Homework 4 Part 1

Part 1 fine-tunes `bert-base-cased` for IMDB sentiment classification, evaluates
it on reviews with keyboard typos, and trains a second model with data augmentation.

| Question | Implementation | Output |
| --- | --- | --- |
| Q1 | `main.py`: `do_train` | `out_original.txt` |
| Q2 | `utils.py`: `custom_transform` | `out_transformed.txt` |
| Q3 | `main.py`: `create_augmented_dataloader` | `out_augmented_original.txt`, `out_augmented_transformed.txt` |

The [saved results](runs/README.md) link to the four final submission files and
record their measured accuracies and provenance. Each file alternates one
prediction and its gold label, preserving the original test order.

## Environment

Use a separate environment from Part 2. From this directory:

```bash
conda create -n dsga1011-hw4-part1 python=3.11
conda activate dsga1011-hw4-part1
pip install -r requirements.txt
```

The final typo transformation uses no NLTK corpora. The starter's optional
synonym approach may require `wordnet` and `punkt` downloads.

## Training and evaluation

Run from `part1/`. Full training uses CUDA when available; the script otherwise
uses CPU. Defaults are learning rate `5e-5`, 3 epochs and batch size 8, with
training seed 0.

```bash
# Q1 debug: 4,000 training and 1,000 test examples.
python -B main.py --train --eval --debug_train

# Q1 full run: 25,000 training and 25,000 test examples.
python -B main.py --train --eval

# Q2: inspect five transformed examples, then evaluate the full Q1 checkpoint.
python -B main.py --eval_transformed --debug_transformation
python -B main.py --eval_transformed --model_dir out

# Q3: train from pretrained BERT, then evaluate the saved model on both test sets.
python -B main.py --train_augmented --eval --eval_transformed
```

The Q1 debug and full commands both write `out/` and `out_original.txt` in the
working directory. Preserve completed runs before rerunning them. The Burst
scripts create a separate directory for each job; see the
[Burst workflow](../hpc/burst/README.md).

To evaluate an existing Q3 checkpoint separately:

```bash
python -B main.py --eval --eval_transformed --model_dir out_augmented
```

## Transformation and augmentation

`TYPO_PROB = 0.40` selects each eligible word with probability 0.40. Eligible
words have at least four letters, contain no apostrophe, are not `never`,
`neither`, `cannot` or `without`, and have an internal vowel in `NEARBY_KEYS`.
One internal vowel is replaced by a configured neighboring keyboard letter;
case, word boundaries, punctuation and spacing are preserved. The first and
last characters of each word are unchanged. Labels are copied unchanged;
this does not guarantee that every transformed review has identical human meaning.

Each review has its own deterministic random generator, seeded from the first
eight bytes of SHA-256 of UTF-8-encoded `"0\0" + original_text`, interpreted as a
big-endian integer. Thus Q2 and Q3 use the same transformed test inputs independently of
processing order. The probability applies to eligible words, not all words or
all characters.

Q3 samples 5,000 training examples without replacement using seed 42, transforms
copies, and appends them to the original 25,000 examples. It retains their labels,
tokenizes with a maximum length of 512, and shuffles training batches. Both
evaluations load the resulting `out_augmented/` checkpoint. Test examples are
never added to the training set.

## Local artifacts and Git

Completed experiments, debug records, logs and model checkpoints remain in
`runs/` locally. Git tracks only the four final prediction files and their
results index. Checkpoints are not included in a fresh clone; reproduce Q1/Q3
or restore the corresponding local/HPC checkpoint before evaluating it.
