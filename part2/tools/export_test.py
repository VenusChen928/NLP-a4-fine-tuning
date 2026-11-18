"""
Load a saved checkpoint for dev evaluation or test export
"""

import argparse
import json
import math
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
from utils import set_random_seeds, compute_metrics
from sql_processing import resolve_sql_preprocessing


class SavedTokenizerDataset(T5Dataset):
    """
    Reuse the original dataset processing with the run's saved tokenizer
    """

    def __init__(self, data_folder, tokenizer, split = "test", sql_preprocessing = "none"):
        self.tokenizer = tokenizer
        self.sql_preprocessing = sql_preprocessing

        # Keep only encoder inputs so inference never receives dev SQL targets
        self.encoder_ids, _ = self.process_data(data_folder, split, tokenizer, sql_preprocessing = sql_preprocessing)
        self.target_ids = None


def main():
    # Read export options and validate the destination
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--run-dir", type = Path, required = True)
    parser.add_argument("--output-dir", type = Path, required = True)
    parser.add_argument("--threads", type = int, default = 4)
    parser.add_argument("--split", choices = ["dev", "test"], default = "test")
    parser.add_argument("--num-beams", type = int, help = "Override the saved beam width")
    parser.add_argument("--batch-size", type = int, help = "Override the saved inference batch size")
    parser.add_argument("--valid", action = "store_true", help = "Select the first executable beam candidate")
    parser.add_argument("--query-timeout", type = float, default = 5, help = "Seconds allowed for each validity check")
    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads must be positive")
    if any(value is not None and value < 1 for value in [args.num_beams, args.batch_size]):
        parser.error("Beam width and batch size must be positive")
    if not math.isfinite(args.query_timeout) or args.query_timeout <= 0:
        parser.error("--query-timeout must be positive and finite")

    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        parser.error("--output-dir must be new; existing files will not be overwritten")

    # Load either the compact local archive or the original Burst training directory
    # Native Burst archives can also have run.json audit metadata. Their
    # authoritative training configuration still lives under checkpoints/.
    if (run / "checkpoint").is_dir() and (run / "run.json").is_file():
        config = json.loads((run / "run.json").read_text())["configuration"]
        checkpoint = tokenizer_path = run / "checkpoint"
    else:
        configs = list(run.glob("checkpoints/ft_experiments/*/run_config.json"))
        configs += list(run.glob("checkpoints/scr_experiments/*/run_config.json"))
        if len(configs) != 1:
            parser.error("Expected one saved training configuration in --run-dir")
        config = json.loads(configs[0].read_text())
        tokenizer_path = configs[0].parent
        checkpoint = tokenizer_path / "best"

    # Apply only the requested decoding changes
    config["sql_preprocessing"] = resolve_sql_preprocessing(config)
    config.update({"split": args.split, "valid": args.valid, "query_timeout": args.query_timeout})
    if args.num_beams is not None:
        config["num_beams"] = args.num_beams
    if args.batch_size is not None:
        config["test_batch_size"] = args.batch_size
    if args.valid and config["num_beams"] < 2:
        parser.error("--valid requires at least two beams")

    torch.set_num_threads(args.threads)
    set_random_seeds(config["seed"])

    # Load the saved tokenizer, selected split and model
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, local_files_only = True)
    dataset = SavedTokenizerDataset(PART2 / "data", tokenizer, split = args.split, sql_preprocessing = config["sql_preprocessing"])
    loader = DataLoader(dataset, batch_size = config["test_batch_size"], shuffle = False, collate_fn = test_collate_fn)

    model = T5ForConditionalGeneration.from_pretrained(checkpoint, local_files_only = True).to(DEVICE)
    output.mkdir(parents = True, exist_ok = False)

    # Use the original inference and scoring functions for both splits
    # Older fine-tuning archives may not include an explicit initialization mode
    test_name = "t5_ft_experiment_test" if config.get("finetune", True) else "t5_ft_experiment_ec_test"
    sql_path = output / ("predictions.sql" if args.split == "dev" else f"{test_name}.sql")
    record_path = output / ("records.pkl" if args.split == "dev" else f"{test_name}.pkl")
    previous_directory = Path.cwd()
    try:
        os.chdir(PART2)
        test_inference(SimpleNamespace(**config), model, loader, sql_path, record_path)

        if args.split == "dev":
            sql_em, record_em, record_f1, errors = compute_metrics(
                "data/dev.sql", sql_path, "records/ground_truth_dev.pkl", record_path
            )
            metrics = {"record_f1": float(record_f1), "record_em": record_em, "sql_em": sql_em,
                       "sql_errors": sum(bool(error) for error in errors),
                       "sql_timeouts": sum(error == "Query timed out" for error in errors)}
            report = {"source_run": str(run), "checkpoint": str(checkpoint), "examples": len(dataset),
                      "configuration": {name: config[name] for name in
                                        ["split", "seed", "sql_preprocessing", "num_beams", "valid", "query_timeout", "max_new_tokens", "test_batch_size"]},
                      "metrics": metrics}
            (output / "run.json").write_text(json.dumps(report, indent = 2) + "\n")
            print(json.dumps(metrics, indent = 2))
    finally:
        os.chdir(previous_directory)

    print(f"Saved {args.split} outputs to {output}")


if __name__ == "__main__":
    main()
