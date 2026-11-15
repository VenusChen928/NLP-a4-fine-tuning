"""GPU interface check on a recorded train/dev subset, never the test set."""
import json
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace

# Allow execution from tools/ as well as the flattened frozen HPC source folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Subset

from load_data import T5Dataset, normal_collate_fn
from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint
from train_t5 import train_epoch, eval_epoch
from utils import set_random_seeds


def main():
    assert torch.cuda.is_available(), 'Smoke test requires the allocated GPU'
    set_random_seeds(42)
    train_data = T5Dataset('data', 'train')
    dev_data = T5Dataset('data', 'dev')
    # Include long SQL sequences to exercise the baseline batch-size memory use.
    train_indices = sorted(range(len(train_data)), key=lambda i: len(train_data.target_ids[i]), reverse=True)[:32]
    dev_indices = list(range(8))
    train_subset = Subset(train_data, train_indices)
    dev_subset = Subset(dev_data, dev_indices)
    dev_subset.tokenizer = dev_data.tokenizer
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, collate_fn=normal_collate_fn)
    dev_loader = DataLoader(dev_subset, batch_size=8, shuffle=False, collate_fn=normal_collate_fn)
    args = SimpleNamespace(finetune=True, experiment_name='smoke', optimizer_type='AdamW',
        learning_rate=3e-4, weight_decay=0.0, scheduler_type='none', max_n_epochs=1,
        num_warmup_epochs=0, max_new_tokens=512, num_beams=1)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
    loss = train_epoch(args, model, train_loader, optimizer, scheduler)
    assert torch.isfinite(torch.tensor(loss)), 'Non-finite training loss'
    checkpoint_dir = 'checkpoints/ft_experiments/smoke'
    save_model(checkpoint_dir, model, best=True)
    reference = model.shared.weight.detach().cpu().clone()
    del model, optimizer, scheduler
    model = load_model_from_checkpoint(args, best=True)
    assert torch.equal(reference, model.shared.weight.detach().cpu()), 'Checkpoint round trip changed weights'
    sql_lines = Path('data/dev.sql').read_text().splitlines()
    Path('reports/smoke_gold.sql').write_text(''.join(sql_lines[i] + '\n' for i in dev_indices))
    with open('records/ground_truth_dev.pkl', 'rb') as f:
        gold_records, gold_errors = pickle.load(f)
    with open('reports/smoke_gold.pkl', 'wb') as f:
        pickle.dump(([gold_records[i] for i in dev_indices], [gold_errors[i] for i in dev_indices]), f)
    metrics = eval_epoch(args, model, dev_loader, 'reports/smoke_gold.sql',
        'results/smoke_dev.sql', 'reports/smoke_gold.pkl', 'records/smoke_dev.pkl')
    report = {'purpose': 'interface validation only; not a baseline result',
        'train_indices': train_indices, 'dev_indices': dev_indices, 'optimizer_steps': 2,
        'train_loss': loss, 'dev_metrics': dict(zip(
            ['loss', 'record_f1', 'record_em', 'sql_em', 'error_rate'], map(float, metrics))),
        'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(), 'checkpoint_round_trip': True}
    Path('reports/smoke.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
