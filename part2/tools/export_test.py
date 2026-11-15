"""Load a compact saved run and export Q7 through the existing test_inference."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

PART2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART2))

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from load_data import T5Dataset, test_collate_fn
from train_t5 import DEVICE, test_inference
from utils import set_random_seeds


class SavedTokenizerTestDataset(T5Dataset):
    """Reuse the original dataset processing with the run's saved tokenizer."""

    def __init__(self, data_folder, tokenizer):
        self.tokenizer = tokenizer
        self.encoder_ids, self.target_ids = self.process_data(data_folder, "test", tokenizer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        parser.error("--output-dir must be new; existing files will not be overwritten")

    config = json.loads((run / "run.json").read_text())["configuration"]
    checkpoint = run / "checkpoint"
    torch.set_num_threads(args.threads)
    set_random_seeds(config["seed"])
    tokenizer = T5TokenizerFast.from_pretrained(checkpoint, local_files_only=True)
    dataset = SavedTokenizerTestDataset(PART2 / "data", tokenizer)
    loader = DataLoader(dataset, batch_size=config["test_batch_size"],
                        shuffle=False, collate_fn=test_collate_fn)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True).to(DEVICE)
    output.mkdir(parents=True, exist_ok=False)

    # The unchanged SQL helper resolves data/flight_database.db from the cwd.
    previous_directory = Path.cwd()
    try:
        os.chdir(PART2)
        test_inference(SimpleNamespace(**config), model, loader,
                       output / "t5_ft_experiment_test.sql",
                       output / "t5_ft_experiment_test.pkl")
    finally:
        os.chdir(previous_directory)
    print(f"Saved Q7 files to {output}")


if __name__ == "__main__":
    main()
