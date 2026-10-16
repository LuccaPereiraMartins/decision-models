"""Descriptive compact-encoder comparison on development cohorts only."""
from datetime import datetime
import hashlib
import json

import numpy as np

from protocol import metrics
from schemas import LABELS


def summarize(evidence):
    models = evidence['models']
    assert set(models) == {'bge_lora', 'finbert_lora', 'bge_frozen'}
    training = evidence['training']
    assert len(training) == 4000
    identities = [(r['body_sha256'], r['reference']) for r in training]
    train_sha = hashlib.sha256(json.dumps(identities).encode()).hexdigest()
    summary = {}
    for name, model in models.items():
        config, performance = model['config'], model['performance']
        assert config['samples'] == 4000 and config['train_subset_sha256'] == train_sha
        assert config['labels'] == list(LABELS)
        entry = {'training_documents': 4000, 'trainable_parameters': config.get('trainable_parameters'),
                 'training_seconds': model.get('history', {}).get('seconds'),
                 'selected_epoch': model.get('history', {}).get('best_epoch'),
                 'training_peak_gib': performance['training_peak_vram_bytes'] / 2**30 if 'training_peak_vram_bytes' in performance else None,
                 'inference_peak_gib': performance['peak_vram_bytes'] / 2**30,
                 'warm_mean_ms': performance['latency_seconds']['mean'] * 1000,
                 'warm_p95_ms': performance['latency_seconds']['p95'] * 1000,
                 'temperature': model['decision']['temperature']}
        for split in ('selection', 'calibration'):
            rows = model[split]['predictions']
            reference = models['bge_lora'][split]['predictions']
            assert [(r['body_sha256'], r['group_id'], r['reference']) for r in rows] == [
                (r['body_sha256'], r['group_id'], r['reference']) for r in reference]
            entry[split] = metrics(rows)
            raw = model[split]['raw_probabilities']
            entry[split + '_raw'] = metrics([{**r, 'probabilities': p, 'uncertainty': float(1 - max(p)), 'discard': {}}
                                            for r, p in zip(rows, raw, strict=True)])
            np.testing.assert_allclose(entry[split]['macro_f1'], entry[split + '_raw']['macro_f1'], atol=1e-12)
        summary[name] = entry
    for key in ('seed', 'epochs', 'max_len', 'lora_rank', 'loss_kind', 'class_weight_power',
                'learning_rate', 'encoder_learning_rate', 'pooling_method'):
        assert models['bge_lora']['config'][key] == models['finbert_lora']['config'][key], key
    zero = evidence['epoch_zero_selection']
    zero_rows = [{**r, 'probabilities': p, 'prediction': LABELS[np.argmax(p)], 'uncertainty': float(1 - max(p)), 'discard': {}}
                 for r, p in zip(zero['cohort'], zero['probabilities'], strict=True)]
    zero_metrics = metrics(zero_rows)
    np.testing.assert_allclose(zero_metrics['macro_f1'], models['bge_lora']['history']['history'][0]['validation']['macro_f1'])
    bge, finbert, frozen = [summary[n] for n in ('bge_lora', 'finbert_lora', 'bge_frozen')]
    execution = evidence['execution']
    return {'status': evidence['status'], 'freeze_sha256': evidence['freeze_sha256'], 'models': summary,
            'epoch_zero_selection': zero_metrics,
            'f1_delta_vs_finbert': bge['selection']['macro_f1'] - finbert['selection']['macro_f1'],
            'f1_delta_vs_frozen': bge['selection']['macro_f1'] - frozen['selection']['macro_f1'],
            'warm_mean_ratio_vs_finbert': bge['warm_mean_ms'] / finbert['warm_mean_ms'],
            'extension_wall_seconds': (datetime.fromisoformat(execution['finished_utc']) - datetime.fromisoformat(execution['started_utc'])).total_seconds(),
            'limits': 'Post-test development systems comparison; fitting cohorts, one seed and no significance, generalization or causal architecture/adaptation claim'}
