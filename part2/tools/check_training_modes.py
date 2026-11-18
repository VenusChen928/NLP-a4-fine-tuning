"""
Check scratch initialization and keep the existing fine-tuning interfaces compatible
"""

import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

PART2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART2))
sys.path.insert(0, str(PART2 / 'tools'))

import train_t5
import t5_utils
import export_test
import experiment_report


def small_model():
    # Small tensors keep interface tests fast; the actual T5-small config is checked separately
    config = T5Config(vocab_size = 32, d_model = 16, d_ff = 32, d_kv = 8,
                      num_layers = 1, num_decoder_layers = 1, num_heads = 2,
                      decoder_start_token_id = 0, pad_token_id = 0, eos_token_id = 1)
    return T5ForConditionalGeneration(config)


@pytest.mark.parametrize('flags', [[], ['--finetune', '--scratch'],
                                  ['--scratch', '--min_n_epochs', '0'],
                                  ['--scratch', '--min_n_epochs', '21']])
def test_invalid_training_options(monkeypatch, flags):
    monkeypatch.setattr(sys, 'argv', ['train_t5.py'] + flags)
    with pytest.raises(SystemExit) as error:
        train_t5.get_args()
    assert error.value.code == 2


def test_finetune_defaults(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['train_t5.py', '--finetune'])
    args = train_t5.get_args()
    assert args.finetune and not args.scratch
    assert (args.max_n_epochs, args.min_n_epochs, args.patience_epochs) == (20, 1, 5)
    assert (args.learning_rate, args.batch_size, args.num_beams) == (3e-4, 16, 1)


def test_initialization_routes(monkeypatch):
    # Loading a config must never invoke the pretrained-weight loader in scratch mode
    config = small_model().config
    calls = []
    monkeypatch.setattr(t5_utils, 'DEVICE', torch.device('cpu'))
    monkeypatch.setattr(t5_utils.T5Config, 'from_pretrained', lambda name: calls.append(('config', name)) or config)
    monkeypatch.setattr(t5_utils.T5ForConditionalGeneration, 'from_pretrained',
                        lambda name: calls.append(('weights', name)) or small_model())

    torch.manual_seed(42)
    scratch = t5_utils.initialize_model(SimpleNamespace(finetune = False))
    assert calls == [('config', 'google-t5/t5-small')]
    assert scratch.config == config
    assert all(parameter.requires_grad for parameter in scratch.parameters())

    torch.manual_seed(43)
    other = t5_utils.initialize_model(SimpleNamespace(finetune = False))
    assert not torch.equal(scratch.shared.weight, other.shared.weight)
    calls.clear()
    t5_utils.initialize_model(SimpleNamespace(finetune = True))
    assert calls == [('weights', 'google-t5/t5-small')]


@pytest.mark.parametrize('minimum, expected_epochs', [(1, 6), (8, 8)])
def test_minimum_epochs_preserves_best_selection(monkeypatch, tmp_path, minimum, expected_epochs):
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / 'checkpoints/scr_experiments/check'
    checkpoint.mkdir(parents = True)
    args = SimpleNamespace(finetune = False, experiment_name = 'check', max_n_epochs = 10,
                           min_n_epochs = minimum, patience_epochs = 5, use_wandb = False)
    saved = []
    monkeypatch.setattr(train_t5, 'train_epoch', lambda *args: 1.0)
    monkeypatch.setattr(train_t5, 'eval_epoch', lambda *args: (1.0, 0.0, 0.0, 0.0, 1.0))
    monkeypatch.setattr(train_t5, 'save_model', lambda directory, model, best: saved.append(best))

    train_t5.train(args, None, None, None, SimpleNamespace(param_groups = [{'lr': 3e-4}]), None)
    history = (checkpoint / 'history.jsonl').read_text().splitlines()
    assert len(history) == expected_epochs
    assert saved.count(True) == 1
    assert json.loads((checkpoint / 'best_metrics.json').read_text())['epoch'] == 1


@pytest.mark.parametrize('finetune', [True, False])
def test_training_and_eval_paths(monkeypatch, tmp_path, finetune):
    # Exercise main(), save/reload and test naming without touching real data or results
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(t5_utils, 'DEVICE', torch.device('cpu'))
    mode = 'ft' if finetune else 'scr'
    other = 'scr' if finetune else 'ft'
    protected = tmp_path / f'checkpoints/{other}_experiments/check/keep.txt'
    protected.parent.mkdir(parents = True)
    protected.write_text('existing experiment')

    class SmallLoader:
        dataset = SimpleNamespace(tokenizer = SimpleNamespace(save_pretrained = lambda path: None))

        def __len__(self):
            return 1

    loader = SmallLoader()
    monkeypatch.setattr(train_t5, 'load_t5_data', lambda *args, **kwargs: (loader, None, None))
    monkeypatch.setattr(train_t5, 'initialize_model', lambda args: small_model())
    monkeypatch.setattr(train_t5, 'initialize_optimizer_and_scheduler',
                        lambda *args: (SimpleNamespace(param_groups = [{'lr': 3e-4}]), None))
    monkeypatch.setattr(train_t5, 'train_epoch', lambda *args: 1.0)
    outputs = []
    def evaluate(args, model, loader, gold_sql, sql, gold_records, records):
        outputs.append((sql, records))
        return 1.0, 0.5, 0.4, 0.3, 0.1

    monkeypatch.setattr(train_t5, 'eval_epoch', evaluate)
    monkeypatch.setattr(train_t5, 'test_inference', lambda args, model, loader, sql, records: outputs.append((sql, records)))
    flags = ['train_t5.py', '--finetune' if finetune else '--scratch',
             '--experiment_name', 'check', '--max_n_epochs', '1', '--run_test']
    monkeypatch.setattr(sys, 'argv', flags)
    train_t5.main()

    checkpoint = tmp_path / f'checkpoints/{mode}_experiments/check'
    config = json.loads((checkpoint / 'run_config.json').read_text())
    assert config['initialization'] == ('pretrained' if finetune else 'random')
    assert config['finetune'] == finetune
    assert (checkpoint / 'best/config.json').is_file()
    assert (checkpoint / 'dev_metrics.json').is_file()
    assert outputs[0][0] == f'results/t5_{mode}_check_dev.sql'
    assert outputs[1] == outputs[0]
    suffix = '' if finetune else '_ec'
    assert outputs[2] == (f'results/t5_ft_experiment{suffix}_test.sql', f'records/t5_ft_experiment{suffix}_test.pkl')

    # Evaluation reuses the selected checkpoint instead of initializing a new model
    monkeypatch.setattr(train_t5, 'initialize_model', lambda args: pytest.fail('Unexpected initialization'))
    monkeypatch.setattr(sys, 'argv', flags + ['--eval_only'])
    train_t5.main()
    assert outputs[-2:] == outputs[1:3]
    assert protected.read_text() == 'existing experiment'

    monkeypatch.setattr(sys, 'argv', flags)
    with pytest.raises(FileExistsError):
        train_t5.main()


@pytest.mark.parametrize('finetune, compact', [(True, False), (False, False), (True, True), (False, True), (None, True)])
def test_export_modes(monkeypatch, tmp_path, finetune, compact):
    run, output = tmp_path / 'run', tmp_path / 'export'
    run.mkdir()
    config = {'seed': 42, 'sql_preprocessing': 'operators_v1', 'test_batch_size': 1, 'num_beams': 1}
    if finetune is not None:
        config['finetune'] = finetune
    if compact:
        (run / 'checkpoint').mkdir()
        (run / 'run.json').write_text(json.dumps({'configuration': config}))
    else:
        # Local Burst archives have audit metadata as well as native checkpoints.
        (run / 'run.json').write_text(json.dumps({'job_id': '123', 'status': 'completed'}))
        path = run / f'checkpoints/{"ft" if finetune else "scr"}_experiments/check'
        path.mkdir(parents = True)
        (path / 'run_config.json').write_text(json.dumps(config))

    monkeypatch.setattr(export_test, 'PART2', tmp_path)
    monkeypatch.setattr(export_test.T5TokenizerFast, 'from_pretrained', lambda *args, **kwargs: None)
    monkeypatch.setattr(export_test.T5ForConditionalGeneration, 'from_pretrained', lambda *args, **kwargs: small_model())
    monkeypatch.setattr(export_test, 'SavedTokenizerDataset', lambda *args, **kwargs: [])
    outputs = []
    monkeypatch.setattr(export_test, 'test_inference', lambda args, model, loader, sql, records: outputs.extend([sql.name, records.name]))
    monkeypatch.setattr(sys, 'argv', ['export_test.py', '--run-dir', str(run), '--output-dir', str(output)])
    export_test.main()
    suffix = '_ec' if finetune is False else ''
    assert outputs == [f't5_ft_experiment{suffix}_test.sql', f't5_ft_experiment{suffix}_test.pkl']

    # An existing export directory must still be rejected before loading a model
    with pytest.raises(SystemExit) as error:
        export_test.main()
    assert error.value.code == 2


@pytest.mark.parametrize('model_mode', [None, 'finetune', 'scratch'])
def test_report_paths(monkeypatch, tmp_path, model_mode):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('A4_RUN_KIND', 'train')
    monkeypatch.setattr(experiment_report, 'verify_release', lambda: {})
    monkeypatch.setattr(experiment_report, 'data_statistics', lambda **kwargs: {})
    mode = 'scr' if model_mode == 'scratch' else 'ft'
    record = {'experiment_name': 'check'}
    if model_mode is not None:
        record['model_mode'] = model_mode
    Path('run_info.json').write_text(json.dumps(record))
    checkpoint = Path(f'checkpoints/{mode}_experiments/check')
    checkpoint.mkdir(parents = True)
    (checkpoint / 'run_config.json').write_text(json.dumps({'sql_preprocessing': 'none', 'max_new_tokens': 512}))
    (checkpoint / 'dev_metrics.json').write_text('{}')
    for directory in ['data', 'records', 'results', 'reports']:
        Path(directory).mkdir()
    Path('data/dev.nl').write_text('question\n')
    Path('data/dev.sql').write_text('SELECT 1\n')
    Path(f'results/t5_{mode}_check_dev.sql').write_text('SELECT 1\n')
    for name in ['ground_truth_dev', f't5_{mode}_check_dev']:
        Path(f'records/{name}.pkl').write_bytes(pickle.dumps(([[(1,)]], [''])))

    experiment_report.finalize(0)
    result = json.loads(Path('run_info.json').read_text())
    assert result['status'] == 'completed'
    assert result['dev_analysis']['subsets']['all']['record_f1'] == pytest.approx(1.0)
    assert f'results/t5_{mode}_check_dev.sql' in result['output_sha256']
