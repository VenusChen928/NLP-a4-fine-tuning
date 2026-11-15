"""Record provenance and report evidence without changing model or metric definitions."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import pickle
import platform
import sys

# Allow execution from tools/ as well as the flattened frozen HPC source folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def verify_release():
    release = json.loads(Path('release_manifest.json').read_text())
    for name, expected in release['data_sha256'].items():
        actual = sha256(name)
        if actual != expected:
            raise ValueError(f'Data checksum mismatch: {name}')
    for name, expected in release['source_sha256'].items():
        path = Path('source') / Path(name).name
        if sha256(path) != expected:
            raise ValueError(f'Source checksum mismatch: {name}')
    return release


def data_statistics(data_folder='data', tokenizer_path='google-t5/t5-small', output_path=None):
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)
    rows = []
    for stage in ['raw', 'processed']:
        for split in ['train', 'dev']:
            row = {'stage': stage, 'split': split}
            for extension, label in [('nl', 'natural_language'), ('sql', 'sql')]:
                texts = (Path(data_folder) / f'{split}.{extension}').read_text().splitlines()
                if stage == 'processed':
                    texts = [text.strip() for text in texts]
                ids = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)['input_ids']
                lengths = [len(sequence) for sequence in ids]
                row['number_of_examples'] = len(texts)
                row[f'{label}_mean_tokens'] = sum(lengths) / len(lengths)
                row[f'{label}_vocabulary_size'] = len({token for sequence in ids for token in sequence})
                row[f'{label}_max_tokens'] = max(lengths)
                row[f'{label}_examples_with_unk'] = sum(tokenizer.unk_token_id in sequence for sequence in ids)
            rows.append(row)
    report = {
        'tokenizer': str(tokenizer_path),
        'counting_convention': 'T5 tokens; exclude added EOS, batch padding, and decoder start token; include emitted UNK. Vocabulary is distinct observed token IDs per split and text field.',
        'processing': 'strip leading/trailing whitespace; no prefix, filtering, truncation, or vocabulary changes',
        'rows': rows,
    }
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        write_json(output_path, report)
    return report


def prepare():
    import torch
    release = verify_release()
    if not torch.cuda.is_available():
        raise RuntimeError('The allocated job cannot use CUDA')
    record = {
        'status': 'running', 'kind': os.environ['A4_RUN_KIND'],
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'job_id': os.environ.get('SLURM_JOB_ID'),
        'partition': os.environ.get('SLURM_JOB_PARTITION'),
        'account': os.environ.get('SLURM_JOB_ACCOUNT'),
        'cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK'),
        'hostname': platform.node(), 'python': sys.version,
        'gpu': torch.cuda.get_device_name(0),
        'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'torch_cuda_version': torch.version.cuda,
        'release': release,
        'test_generation_requested': False,
    }
    write_json('run_info.json', record)
    write_json('reports/packages.json', {d.metadata['Name']: d.version for d in metadata.distributions() if d.metadata['Name']})
    data_statistics(output_path='reports/q4_statistics.json')
    print(json.dumps(record, indent=2))


def dev_evidence():
    from utils import compute_record_F1
    predictions = Path('results/t5_ft_baseline_v1_dev.sql').read_text().splitlines()
    questions = Path('data/dev.nl').read_text().splitlines()
    gold_sql = Path('data/dev.sql').read_text().splitlines()
    with open('records/ground_truth_dev.pkl', 'rb') as f:
        gold_records, _ = pickle.load(f)
    with open('records/t5_ft_baseline_v1_dev.pkl', 'rb') as f:
        predicted_records, errors = pickle.load(f)
    if not len(predictions) == len(questions) == len(gold_sql) == len(gold_records) == len(predicted_records) == len(errors):
        raise ValueError('Dev output alignment failed')
    with open('reports/dev_evidence.jsonl', 'w') as f:
        for i, (question, gold, predicted, gold_rec, pred_rec, error) in enumerate(zip(
                questions, gold_sql, predictions, gold_records, predicted_records, errors)):
            row = {'index': i, 'question': question, 'gold_sql': gold, 'predicted_sql': predicted,
                'sql_exact_match': gold.strip() == predicted.strip(),
                'record_exact_match': set(gold_rec) == set(pred_rec),
                'record_f1': float(compute_record_F1([gold_rec], [pred_rec])),
                'gold_record_count': len(set(gold_rec)), 'predicted_record_count': len(set(pred_rec)),
                'execution_error': error}
            f.write(json.dumps(row) + '\n')
    return {
        'sql_timeouts': sum('timed out' in message.lower() for message in errors),
        'sql_execution_errors': sum(bool(message) for message in errors),
        'examples': len(predictions),
    }


def finalize(exit_code):
    path = Path('run_info.json')
    record = json.loads(path.read_text()) if path.exists() else {}
    record['process_exit_code'] = exit_code
    record['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    try:
        verify_release()
        if exit_code == 0:
            if os.environ['A4_RUN_KIND'] == 'baseline':
                record['dev_analysis'] = dev_evidence()
                record['final_dev'] = json.loads(Path('checkpoints/ft_experiments/baseline_v1/dev_metrics.json').read_text())
            elif not Path('reports/smoke.json').is_file():
                raise ValueError('Smoke output is missing')
        record['status'] = 'completed' if exit_code == 0 else 'failed'
    except Exception as error:
        record['status'] = 'failed'
        record['report_error'] = f'{type(error).__name__}: {error}'
        write_json(path, record)
        raise
    record['output_sha256'] = {str(p): sha256(p) for directory in ['checkpoints', 'results', 'records', 'reports']
        for p in sorted(Path(directory).rglob('*')) if p.is_file()}
    write_json(path, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'finalize', 'stats'])
    parser.add_argument('--exit-code', type=int, default=0)
    parser.add_argument('--data-folder', default='data')
    parser.add_argument('--tokenizer', default='google-t5/t5-small')
    parser.add_argument('--output', type=Path, help='Optional statistics JSON; otherwise print only')
    args = parser.parse_args()
    if args.action == 'stats':
        print(json.dumps(data_statistics(args.data_folder, args.tokenizer, args.output), indent=2))
    elif args.action == 'prepare':
        prepare()
    else:
        finalize(args.exit_code)
