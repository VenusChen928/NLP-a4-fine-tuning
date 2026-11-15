"""
GPU interface check on a recorded train/dev subset, never the test set
"""

import argparse
import json
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace

# Support local and packaged imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Subset

from load_data import T5Dataset, normal_collate_fn
from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint
from train_t5 import train_epoch, eval_epoch
from utils import set_random_seeds
from sql_processing import SQL_PREPROCESSING_MODES, SQL_OPERATOR_TOKENS, encode_sql, resolve_sql_preprocessing
from experiment_report import sql_processing_checks


def select_smoke_indices(dataset, sql_path, count, sql_preprocessing, longest_first = False):
    indices = sorted(range(len(dataset)), key = lambda i: len(dataset.target_ids[i]), reverse = True) if longest_first else list(range(len(dataset)))

    if sql_preprocessing == 'none':
        return indices[:count]

    # Include examples of both SQL operators
    encoded = [encode_sql(sql, sql_preprocessing) for sql in Path(sql_path).read_text().splitlines()]
    required = []

    for marker in SQL_OPERATOR_TOKENS.values():
        index = next((i for i, sql in enumerate(encoded) if marker in sql), None)
        if index is None:
            raise ValueError(f'SQL processing smoke data has no example of {marker}')
        required.append(index)

    return list(dict.fromkeys(required + indices))[:count]


def main():
    # Read the preprocessing mode
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--sql_preprocessing', choices = SQL_PREPROCESSING_MODES, default = 'none')
    sql_preprocessing = parser.parse_args().sql_preprocessing

    assert torch.cuda.is_available(), 'Smoke test requires the allocated GPU'
    set_random_seeds(42)

    # Load small subsets with long SQL and both operators
    train_data = T5Dataset('data', 'train', sql_preprocessing = sql_preprocessing)
    dev_data = T5Dataset('data', 'dev', sql_preprocessing = sql_preprocessing)

    train_indices = select_smoke_indices(train_data, 'data/train.sql', 32, sql_preprocessing, longest_first = True)
    dev_indices = select_smoke_indices(dev_data, 'data/dev.sql', 8, sql_preprocessing)

    # SQL processing: check target recovery before training
    processing_checks = {}
    for split, dataset, indices in [('train', train_data, train_indices), ('dev', dev_data, dev_indices)]:
        queries = Path(f'data/{split}.sql').read_text().splitlines()
        processing_checks[split] = sql_processing_checks([queries[i].strip() for i in indices], dataset.tokenizer, sql_preprocessing)

    # Build the training and evaluation batches
    train_subset = Subset(train_data, train_indices)
    dev_subset = Subset(dev_data, dev_indices)
    dev_subset.tokenizer = dev_data.tokenizer
    train_loader = DataLoader(train_subset, batch_size = 16, shuffle = True, collate_fn = normal_collate_fn)
    dev_loader = DataLoader(dev_subset, batch_size = 8, shuffle = False, collate_fn = normal_collate_fn)

    # Run two optimizer steps
    args = SimpleNamespace(
        finetune = True, experiment_name = 'smoke', optimizer_type = 'AdamW',
        learning_rate = 3e-4, weight_decay = 0.0, scheduler_type = 'none', max_n_epochs = 1,
        num_warmup_epochs = 0, max_new_tokens = 512, num_beams = 1, sql_preprocessing = sql_preprocessing
    )
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
    loss = train_epoch(args, model, train_loader, optimizer, scheduler)
    assert torch.isfinite(torch.tensor(loss)), 'Non-finite training loss'

    # Save and reload the model and preprocessing mode
    checkpoint_dir = 'checkpoints/ft_experiments/smoke'
    save_model(checkpoint_dir, model, best = True)

    config_path = Path(checkpoint_dir) / 'run_config.json'
    config_path.write_text(json.dumps(vars(args), indent = 2) + '\n')
    args.sql_preprocessing = resolve_sql_preprocessing(json.loads(config_path.read_text()), requested = sql_preprocessing)

    reference = model.shared.weight.detach().cpu().clone()
    del model, optimizer, scheduler

    model = load_model_from_checkpoint(args, best = True)
    assert torch.equal(reference, model.shared.weight.detach().cpu()), 'Checkpoint round trip changed weights'

    # Prepare matching gold SQL and records for the subset
    sql_lines = Path('data/dev.sql').read_text().splitlines()
    Path('reports/smoke_gold.sql').write_text(''.join(sql_lines[i] + '\n' for i in dev_indices))

    with open('records/ground_truth_dev.pkl', 'rb') as f:
        gold_records, gold_errors = pickle.load(f)
    with open('reports/smoke_gold.pkl', 'wb') as f:
        pickle.dump(([gold_records[i] for i in dev_indices], [gold_errors[i] for i in dev_indices]), f)

    # Evaluate restored SQL and save the smoke report
    metrics = eval_epoch(args, model, dev_loader, 'reports/smoke_gold.sql', 'results/smoke_dev.sql', 'reports/smoke_gold.pkl', 'records/smoke_dev.pkl')

    report = {'purpose': 'interface validation only; not a baseline result',
        'train_indices': train_indices, 'dev_indices': dev_indices, 'optimizer_steps': 2,
        'train_loss': loss, 'dev_metrics': dict(zip(
            ['loss', 'record_f1', 'record_em', 'sql_em', 'error_rate'], map(float, metrics))),
        'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(), 'checkpoint_round_trip': True,
        'sql_preprocessing': sql_preprocessing, 'sql_processing_checks': processing_checks}

    Path('reports/smoke.json').write_text(json.dumps(report, indent = 2))
    print(json.dumps(report, indent = 2))


if __name__ == '__main__':
    main()
