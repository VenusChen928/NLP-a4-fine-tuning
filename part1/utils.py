import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

import hashlib
import re

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


TYPO_PROB = 0.40   # Probability for random transformation

TRANSFORM_SEED = 0

# Vowels can be replaced by Adjacent keyboard letters:
NEARBY_KEYS = {
    "a" : "qwsz",
    "e" : "wsdr",
    "o" : "iklp",
    "i" : "ujko",
    "u" : "yhji"
    }

# Protect expressions that can affect Negative meanings:
PROTECTED_WORDS = {"never", "neither", "cannot", "without"}

# Identify expressions with apostrophes such as "don't" as complete words:
WORD_PATTERN = re.compile(r"\b[A-Za-z]+(?:['’][A-Za-z]+)*\b")


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]   # Store the original text

    seed_input = f"{TRANSFORM_SEED}\0{text}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_input).digest()[:8], "big")
    rng = random.Random(seed)


    def replace_word(match):
        word = match.group(0)

        # Skip short words, abbreviations,
        # possessives and protected words:
        if (
            len(word) < 4
            or "'" in word
            or "’" in word
            or word.lower() in PROTECTED_WORDS
        ):
            return word

        # Only replace vowels within words (find positions)
        # while keeping the front and end:
        positions = [i for i in range(1, len(word) - 1) if word[i].lower() in NEARBY_KEYS]

        # No replacement position or not selected:
        if not positions or rng.random() >= TYPO_PROB:
            return word

        i = rng.choice(positions)
        replacement = rng.choice(NEARBY_KEYS[word[i].lower()])

        if word[i].isupper():
            replacement = replacement.upper()

        return word[:i] + replacement + word[i + 1:]

    example["text"] = WORD_PATTERN.sub(replace_word, text)

    return example
