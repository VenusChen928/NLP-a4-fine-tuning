import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb
import json
import time

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records, set_random_seeds, select_valid_sql

from sql_processing import SQL_PREPROCESSING_MODES, decode_sql, resolve_sql_preprocessing, validate_sql_preprocessing

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters:
    model_group = parser.add_mutually_exclusive_group(required = True)
    model_group.add_argument('--finetune', action = 'store_true', help = "Finetune pretrained T5-small")
    model_group.add_argument('--scratch', action = 'store_true', help = "Train T5-small from random weights")

    # Training hyperparameters:
    parser.add_argument('--optimizer_type', type = str, default = "AdamW", choices = ["AdamW"],
                        help = "What optimizer to use")

    parser.add_argument('--learning_rate', type = float, default = 3e-4)

    parser.add_argument('--weight_decay', type = float, default = 0)

    parser.add_argument('--scheduler_type', type = str, default = "linear", choices = ["none", "cosine", "linear"],
                        help = "Whether to use a LR scheduler and what type to use if so?")

    parser.add_argument('--num_warmup_epochs', type = int, default = 1,
                        help = "How many epochs to warm up the learning rate for if using a scheduler?")

    parser.add_argument('--max_n_epochs', type = int, default = 20,
                        help = "How many epochs to train the model for?")

    parser.add_argument('--patience_epochs', type = int, default = 5,
                        help = "If validation performance stops improving, how many epochs should we wait before stopping?")

    # Scratch may need time to learn SQL before dev F1 starts improving:
    # The default preserves the existing fine-tuning stopping rule
    parser.add_argument('--min_n_epochs', type = int, default = 1,
                        help = "Minimum number of epochs before early stopping is allowed")

    parser.add_argument('--use_wandb', action = 'store_true',
                        help = "If set, we will use wandb to keep track of experiments")

    parser.add_argument('--experiment_name', type = str, default = 'experiment',
                        help = "How should we name this experiment?")

    # Data hyperparameters:
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--test_batch_size', type = int, default = 16)

    # SQL processing:
    parser.add_argument('--sql_preprocessing', choices = SQL_PREPROCESSING_MODES, default = None)

    # Reproducibility:
    parser.add_argument("--seed", type = int, default = 42)

    # Generation hyperparameters:
    parser.add_argument("--max_new_tokens", type = int, default = 512)
    parser.add_argument("--num_beams", type = int, default = 1)

    # Execution options:
    parser.add_argument("--eval_only", action = "store_true",
                        help = "Load the best checkpoint and evaluate without training.")

    parser.add_argument("--run_test", action = "store_true",
                        help = "Generate test submission files from the best checkpoint.")

    args = parser.parse_args()

    if not args.eval_only and args.max_n_epochs < 1:
        parser.error("--max_n_epochs must be at least 1 for training!")

    if args.patience_epochs < 1:
        parser.error("--patience_epochs must be at least 1!")

    if args.min_n_epochs < 1 or (not args.eval_only and args.min_n_epochs > args.max_n_epochs):
        parser.error("--min_n_epochs must be positive and cannot exceed --max_n_epochs for training!")

    if args.max_new_tokens < 1 or args.num_beams < 1:
        parser.error("--max_new_tokens and --num_beams must be positive!")

    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{args.experiment_name}_dev.pkl')
    for epoch in range(args.max_n_epochs):
        epoch_started = time.monotonic()
        learning_rate_start = optimizer.param_groups[0]['lr']
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': float(tr_loss),
            'dev_loss': float(eval_loss),
            'record_f1': float(record_f1),
            'record_em': float(record_em),
            'sql_em': float(sql_em),
            'error_rate': float(error_rate),
            'learning_rate_start': learning_rate_start,
            'learning_rate_end': optimizer.param_groups[0]['lr'],
            'elapsed_seconds': time.monotonic() - epoch_started,
            'is_best': epochs_since_improvement == 0,
        }
        with open(os.path.join(checkpoint_dir, 'history.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(epoch_metrics) + '\n')
        if epoch_metrics['is_best']:
            with open(os.path.join(checkpoint_dir, 'best_metrics.json'), 'w', encoding='utf-8') as f:
                json.dump(epoch_metrics, f, indent=2)

        # Keep selecting the best checkpoint throughout the minimum training period
        if epoch + 1 >= args.min_n_epochs and epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
            use_cache=False,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens


def generate_sql(args, model, tokenizer, encoder_input, encoder_mask, initial_decoder_inputs):
    # Return multiple candidates only when validity filtering is requested
    valid = getattr(args, "valid", False)
    num_return_sequences = args.num_beams if valid else 1

    generated_ids = model.generate(
        input_ids = encoder_input.to(DEVICE),
        attention_mask = encoder_mask.to(DEVICE),
        decoder_input_ids = initial_decoder_inputs.to(DEVICE),
        max_new_tokens = args.max_new_tokens,
        num_beams = args.num_beams,
        num_return_sequences = num_return_sequences,
        do_sample = False
    )

    queries = tokenizer.batch_decode(
        generated_ids.cpu(),
        skip_special_tokens = True,
        clean_up_tokenization_spaces = False
    )

    # Restore SQL operators and keep one query per line
    queries = [decode_sql(query, args.sql_preprocessing) for query in queries]
    queries = [query.replace("\r", " ").replace("\n", " ").strip() for query in queries]

    if len(queries) != encoder_input.shape[0] * num_return_sequences:
        raise ValueError("Generated candidates do not match the input batch!")

    # Beam candidates are grouped by question in descending sequence-score order
    if valid:
        queries = [select_valid_sql(queries[start:start + num_return_sequences], args.query_timeout)
                   for start in range(0, len(queries), num_return_sequences)]

    return queries


@torch.no_grad()
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance
    '''
    # TODO
    model.eval()
    tokenizer = dev_loader.dataset.tokenizer

    criterion = nn.CrossEntropyLoss(reduction = "sum")
    total_loss = 0.0
    total_tokens = 0
    predicted_queries = []

    for batch in tqdm(dev_loader, desc = "Dev"):
        (
            encoder_input,
            encoder_mask,
            decoder_input,
            decoder_targets,
            initial_decoder_inputs
        ) = batch

        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        # Teacher-forced loss on the gold SQL on dev set:
        logits = model(
            input_ids = encoder_input,
            attention_mask = encoder_mask,
            decoder_input_ids = decoder_input,
            use_cache = False
        ).logits

        non_pad = decoder_targets != PAD_IDX

        total_loss += criterion(logits[non_pad], decoder_targets[non_pad]).item()
        total_tokens += non_pad.sum().item()

        del logits

        # Free generation (no gold targets) for SQL/record metrics:
        predicted_queries.extend(
            generate_sql(
                args,
                model,
                tokenizer,
                encoder_input,
                encoder_mask,
                initial_decoder_inputs
            )
        )

    if total_tokens == 0:
        raise ValueError("The dev set contains no target tokens!")

    if len(predicted_queries) != len(dev_loader.dataset):
        raise ValueError("The number of dev predictions is incorrect!")

    os.makedirs(os.path.dirname(model_sql_path) or ".", exist_ok = True)
    os.makedirs(os.path.dirname(model_record_path) or ".", exist_ok = True)

    save_queries_and_records(predicted_queries, model_sql_path, model_record_path)

    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth,
        model_sql_path,
        gt_record_path,
        model_record_path
    )

    error_rate = sum(bool(message) for message in error_msgs) / len(error_msgs)
    average_loss = total_loss / total_tokens

    return average_loss, record_f1, record_em, sql_em, error_rate

@torch.no_grad()
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated
    database records. Implementation should be very similar to eval_epoch
    '''
    model.eval()
    tokenizer = test_loader.dataset.tokenizer
    predicted_queries = []

    # Reuse the same input-only inference loop for dev and test exports
    for batch in tqdm(test_loader, desc = getattr(args, "split", "test").capitalize()):
        encoder_input, encoder_mask, initial_decoder_inputs = batch

        predicted_queries.extend(
            generate_sql(
                args,
                model,
                tokenizer,
                encoder_input,
                encoder_mask,
                initial_decoder_inputs
            )
        )

    if len(predicted_queries) != len(test_loader.dataset):
        raise ValueError("The number of predictions is incorrect!")

    os.makedirs(os.path.dirname(model_sql_path) or ".", exist_ok = True)
    os.makedirs(os.path.dirname(model_record_path) or ".", exist_ok = True)

    save_queries_and_records(predicted_queries, model_sql_path, model_record_path)

def main():
    # Get key arguments
    args = get_args()

    # Keep scratch checkpoints and dev outputs separate from fine-tuning runs
    model_type = "ft" if args.finetune else "scr"
    checkpoint_dir = os.path.join("checkpoints", f"{model_type}_experiments", args.experiment_name)

    if args.eval_only:
        with open(os.path.join(checkpoint_dir, "run_config.json"), encoding = "utf-8") as f:
            saved_config = json.load(f)
        args.sql_preprocessing = resolve_sql_preprocessing(saved_config, requested = args.sql_preprocessing)
    else:
        args.sql_preprocessing = validate_sql_preprocessing(args.sql_preprocessing or "none")

    set_random_seeds(args.seed)

    # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
    wandb_run = None
    if args.use_wandb and not args.eval_only:
        wandb_run = setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size, sql_preprocessing = args.sql_preprocessing)

    if not args.eval_only:
        # A new training run must not overwrite an existing experiment
        os.makedirs(checkpoint_dir, exist_ok=False)

        model = initialize_model(args)
        # Save the actual model and optimization configuration for Q5
        run_config = {
            **vars(args),
            "base_model_name": "google-t5/t5-small",
            "initialization": "pretrained" if args.finetune else "random",
            "base_model_revision": getattr(model.config, "_commit_hash", None),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model_dtype": str(next(model.parameters()).dtype),
            "gradient_accumulation_steps": 1,
            "optimizer_betas": [0.9, 0.999],
            "optimizer_eps": 1e-8,
            "checkpoint_selection": "maximum dev Record F1; earliest epoch on ties",
        }

        with open(os.path.join(checkpoint_dir, "run_config.json"), "w", encoding = "utf-8") as f:
            json.dump(run_config, f, indent = 2)

        train_loader.dataset.tokenizer.save_pretrained(checkpoint_dir)

        optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

        # Train
        train(args, model, train_loader, dev_loader, optimizer, scheduler)

        del model, optimizer, scheduler

    # Evaluate the selected checkpoint
    model = load_model_from_checkpoint(args, best = True)
    model.eval()

    # Dev set
    dev_sql_path = os.path.join("results", f"t5_{model_type}_{args.experiment_name}_dev.sql")
    dev_record_path = os.path.join("records", f"t5_{model_type}_{args.experiment_name}_dev.pkl")

    dev_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
        args,
        model,
        dev_loader,
        "data/dev.sql",
        dev_sql_path,
        "records/ground_truth_dev.pkl",
        dev_record_path
    )

    print(
        f"Best checkpoint dev results: "
        f"Loss = {dev_loss:.4f}, "
        f"Record F1 = {record_f1:.4f}, "
        f"Record EM = {record_em:.4f}, "
        f"SQL EM = {sql_em:.4f}, "
        f"SQL execution error rate = {error_rate:.2%}"
    )

    # Save final dev metrics locally for experiment comparison
    dev_metrics = {
        "loss": float(dev_loss),
        "record_f1": float(record_f1),
        "record_em": float(record_em),
        "sql_em": float(sql_em),
        "error_rate": float(error_rate)
    }

    evaluation_result = {
        "checkpoint": os.path.join(checkpoint_dir, "best"),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "sql_preprocessing": args.sql_preprocessing,
        "metrics": dev_metrics
    }

    metrics_path = os.path.join(checkpoint_dir, "dev_metrics.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_result, f, indent=2)

    # Record the selected checkpoint's metrics in the W&B
    if wandb_run is not None:
        wandb_run.summary.update({f"best_dev/{name}": value for name, value in dev_metrics.items()})

    # Test set
    if args.run_test:
        # The assignment uses the ft prefix for both required submission names
        test_name = "t5_ft_experiment_test" if args.finetune else "t5_ft_experiment_ec_test"
        test_sql_path = os.path.join("results", f"{test_name}.sql")
        test_record_path = os.path.join("records", f"{test_name}.pkl")

        test_inference(
            args,
            model,
            test_loader,
            test_sql_path,
            test_record_path
        )

        print(
            f"Saved test predictions to {test_sql_path} "
            f"and {test_record_path}"
        )

    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    main()
