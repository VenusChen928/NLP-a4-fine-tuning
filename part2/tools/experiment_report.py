"""
Record provenance and report evidence without changing model or metric definitions
"""

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

# Support local and packaged imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sql_processing import SQL_PREPROCESSING_MODES, encode_sql, decode_sql, resolve_sql_preprocessing, validate_sql_preprocessing


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent = 2) + '\n')
    temporary.replace(path)


def verify_release():
    release = json.loads(Path('release_manifest.json').read_text())

    # Verify data and source files separately
    for name, expected in release['data_sha256'].items():
        actual = sha256(name)
        if actual != expected:
            raise ValueError(f'Data checksum mismatch: {name}')

    for name, expected in release['source_sha256'].items():
        path = Path('source') / Path(name).name
        if sha256(path) != expected:
            raise ValueError(f'Source checksum mismatch: {name}')

    return release


def sql_processing_checks(queries, tokenizer, sql_preprocessing, max_new_tokens = 512):
    # SQL processing: check recovery through the tokenizer
    encoded = [encode_sql(query, sql_preprocessing) for query in queries]
    ids = tokenizer(encoded, add_special_tokens = True, padding = False, truncation = False)['input_ids']
    decoded = tokenizer.batch_decode(ids, skip_special_tokens = True, clean_up_tokenization_spaces = False)
    restored = [decode_sql(query, sql_preprocessing) for query in decoded]

    checks = {
        'examples': len(queries),
        'conversion_mismatches': sum(decode_sql(text, sql_preprocessing) != query for query, text in zip(queries, encoded)),
        'tokenizer_roundtrip_mismatches': sum(query != prediction for query, prediction in zip(queries, restored)),
        'examples_with_unk': sum(tokenizer.unk_token_id in sequence for sequence in ids),
        'max_tokens_with_eos': max(map(len, ids)),
        'examples_above_generation_limit': sum(len(sequence) > max_new_tokens for sequence in ids),
        'max_new_tokens': max_new_tokens
    }

    failures = ['conversion_mismatches', 'tokenizer_roundtrip_mismatches', 'examples_with_unk', 'examples_above_generation_limit']
    if sql_preprocessing != 'none' and any(checks[name] for name in failures):
        raise ValueError(f'SQL processing validation failed: {checks}')

    return checks


def data_statistics(data_folder = 'data', tokenizer_path = 'google-t5/t5-small', output_path = None, sql_preprocessing = 'none', max_new_tokens = 512):
    from transformers import T5TokenizerFast

    validate_sql_preprocessing(sql_preprocessing)
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, local_files_only = True)

    # Collect raw and processed statistics for both splits
    rows = []
    processing_checks = {}

    for stage in ['raw', 'processed']:
        for split in ['train', 'dev']:
            row = {'stage': stage, 'split': split}
            for extension, label in [('nl', 'natural_language'), ('sql', 'sql')]:
                texts = (Path(data_folder) / f'{split}.{extension}').read_text().splitlines()

                if stage == 'processed':
                    texts = [text.strip() for text in texts]

                    # SQL processing: count the actual decoder targets
                    if extension == 'sql':
                        processing_checks[split] = sql_processing_checks(texts, tokenizer, sql_preprocessing, max_new_tokens = max_new_tokens)
                        texts = [encode_sql(text, sql_preprocessing) for text in texts]

                ids = tokenizer(texts, add_special_tokens = False, padding = False, truncation = False)['input_ids']
                lengths = [len(sequence) for sequence in ids]

                row['number_of_examples'] = len(texts)
                row[f'{label}_mean_tokens'] = sum(lengths) / len(lengths)
                row[f'{label}_vocabulary_size'] = len({token for sequence in ids for token in sequence})
                row[f'{label}_max_tokens'] = max(lengths)
                row[f'{label}_examples_with_unk'] = sum(tokenizer.unk_token_id in sequence for sequence in ids)

            rows.append(row)

    # Keep counting conventions with the reported values
    report = {
        'tokenizer': str(tokenizer_path),
        'counting_convention': 'T5 tokens; exclude added EOS, batch padding, and decoder start token; include emitted UNK. Vocabulary is distinct observed token IDs per split and text field.',
        'processing': 'strip leading/trailing whitespace' + ('; encode < as SQL_LT and <= as SQL_LE outside quoted text and comments' if sql_preprocessing == 'operators_v1' else '; no SQL operator conversion'),
        'sql_preprocessing': sql_preprocessing,
        'sql_processing_checks': processing_checks,
        'rows': rows
    }

    if output_path is not None:
        Path(output_path).parent.mkdir(parents = True, exist_ok = True)
        write_json(output_path, report)

    return report


def prepare(experiment_name = 'baseline_v1', sql_preprocessing = 'none', model_mode = 'finetune'):
    import torch

    # Check the release and allocated device
    release = verify_release()
    if not torch.cuda.is_available():
        raise RuntimeError('The allocated job cannot use CUDA')

    # Record the environment and experiment configuration
    record = {
        'status': 'running', 'kind': os.environ['A4_RUN_KIND'],
        'experiment_name': experiment_name, 'sql_preprocessing': validate_sql_preprocessing(sql_preprocessing),
        'model_mode': model_mode, 'initialization': 'pretrained' if model_mode == 'finetune' else 'random',
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
        'test_generation_requested': False
    }

    # Save provenance and initial data statistics
    write_json('run_info.json', record)
    write_json('reports/packages.json', {d.metadata['Name']: d.version for d in metadata.distributions() if d.metadata['Name']})
    data_statistics(output_path = 'reports/q4_statistics.json', sql_preprocessing = sql_preprocessing)
    print(json.dumps(record, indent = 2))


def dev_evidence(experiment_name = 'baseline_v1', model_type = 'ft'):
    from utils import compute_record_F1

    # Load restored predictions and original gold answers
    predictions = Path(f'results/t5_{model_type}_{experiment_name}_dev.sql').read_text().splitlines()
    questions = Path('data/dev.nl').read_text().splitlines()
    gold_sql = Path('data/dev.sql').read_text().splitlines()

    with open('records/ground_truth_dev.pkl', 'rb') as f:
        gold_records, _ = pickle.load(f)
    with open(f'records/t5_{model_type}_{experiment_name}_dev.pkl', 'rb') as f:
        predicted_records, errors = pickle.load(f)

    if not len(predictions) == len(questions) == len(gold_sql) == len(gold_records) == len(predicted_records) == len(errors):
        raise ValueError('Dev output alignment failed')

    # Save per-example evidence and label operator subsets
    groups = {'all': [], 'target_operators': [], 'other': []}
    with open('reports/dev_evidence.jsonl', 'w') as f:
        for i, (question, gold, predicted, gold_rec, pred_rec, error) in enumerate(zip(questions, gold_sql, predictions, gold_records, predicted_records, errors)):
            row = {'index': i, 'question': question, 'gold_sql': gold, 'predicted_sql': predicted,
                'sql_exact_match': gold.strip() == predicted.strip(),
                'record_exact_match': set(gold_rec) == set(pred_rec),
                'record_f1': float(compute_record_F1([gold_rec], [pred_rec])),
                'gold_record_count': len(set(gold_rec)), 'predicted_record_count': len(set(pred_rec)),
                'execution_error': error,
                'has_target_operator': encode_sql(gold, 'operators_v1') != gold}

            groups['all'].append(row)
            groups['target_operators' if row['has_target_operator'] else 'other'].append(row)
            f.write(json.dumps(row) + '\n')

    # Aggregate metrics for each subset
    subsets = {}
    for name, group in groups.items():
        count = len(group)
        subsets[name] = {
            'examples': count,
            'record_f1': sum(row['record_f1'] for row in group) / count if count else None,
            'record_em': sum(row['record_exact_match'] for row in group) / count if count else None,
            'sql_em': sum(row['sql_exact_match'] for row in group) / count if count else None,
            'sql_execution_errors': sum(bool(row['execution_error']) for row in group),
            'error_rate': sum(bool(row['execution_error']) for row in group) / count if count else None
        }

    return {
        'sql_timeouts': sum('timed out' in message.lower() for message in errors),
        'sql_execution_errors': sum(bool(message) for message in errors),
        'examples': len(predictions),
        'subsets': subsets
    }


def finalize(exit_code, experiment_name = None):
    path = Path('run_info.json')
    record = json.loads(path.read_text()) if path.exists() else {}
    record['process_exit_code'] = exit_code
    record['finished_at_utc'] = datetime.now(timezone.utc).isoformat()

    # Build final reports from the saved training configuration
    try:
        verify_release()
        if exit_code == 0:
            if os.environ['A4_RUN_KIND'] != 'smoke':
                experiment_name = experiment_name or record.get('experiment_name', 'baseline_v1')

                # Read the selected mode, with the original default for older runs
                model_type = 'ft' if record.get('model_mode', 'finetune') == 'finetune' else 'scr'
                checkpoint = Path(f'checkpoints/{model_type}_experiments') / experiment_name
                config = json.loads((checkpoint / 'run_config.json').read_text())
                sql_preprocessing = resolve_sql_preprocessing(config)

                record['experiment_name'] = experiment_name
                record['sql_preprocessing'] = sql_preprocessing
                record['dev_analysis'] = dev_evidence(experiment_name = experiment_name, model_type = model_type)
                record['final_dev'] = json.loads((checkpoint / 'dev_metrics.json').read_text())
                beam_report = Path('dev/beam4_valid/run.json')
                if beam_report.is_file():
                    record['beam4_valid_evaluation'] = json.loads(beam_report.read_text())
                    record['beam4_valid_evaluation']['inference_log_sha256'] = sha256('dev_beam4_valid.log')
                data_statistics(output_path = 'reports/q4_statistics.json', sql_preprocessing = sql_preprocessing, max_new_tokens = config['max_new_tokens'])
            elif not Path('reports/smoke.json').is_file():
                raise ValueError('Smoke output is missing')

        record['status'] = 'completed' if exit_code == 0 else 'failed'
    except Exception as error:
        record['status'] = 'failed'
        record['report_error'] = f'{type(error).__name__}: {error}'
        write_json(path, record)
        raise

    # Save output hashes after all reports are complete
    record['output_sha256'] = {str(p): sha256(p) for directory in ['checkpoints', 'results', 'records', 'reports', 'dev']
        for p in sorted(Path(directory).rglob('*')) if p.is_file()}
    write_json(path, record)


if __name__ == '__main__':
    # Choose the report action and input files
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices = ['prepare', 'finalize', 'stats'])
    parser.add_argument('--exit-code', type = int, default = 0)
    parser.add_argument('--data-folder', default = 'data')
    parser.add_argument('--tokenizer', default = 'google-t5/t5-small')
    parser.add_argument('--output', type = Path, help = 'Optional statistics JSON; otherwise print only')

    # SQL processing: share the experiment settings
    parser.add_argument('--sql-preprocessing', choices = SQL_PREPROCESSING_MODES, default = 'none')
    parser.add_argument('--experiment-name', default = None)
    parser.add_argument('--max-new-tokens', type = int, default = 512)
    parser.add_argument('--model-mode', choices = ['finetune', 'scratch'], default = 'finetune')
    args = parser.parse_args()

    if args.action == 'stats':
        print(json.dumps(data_statistics(args.data_folder, args.tokenizer, args.output, sql_preprocessing = args.sql_preprocessing, max_new_tokens = args.max_new_tokens), indent = 2))
    elif args.action == 'prepare':
        prepare(experiment_name = args.experiment_name or 'baseline_v1', sql_preprocessing = args.sql_preprocessing, model_mode = args.model_mode)
    else:
        finalize(args.exit_code, experiment_name = args.experiment_name)
