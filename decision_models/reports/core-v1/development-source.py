"""Descriptive development evidence; no model fitting or held-out evaluation."""
from collections import Counter
from datetime import datetime
import hashlib
import json

import numpy as np

from protocol import metrics


def summarize(evidence):
    models = evidence['models']
    assert set(models) == {'4000', '16000'}
    small, large = models['4000'], models['16000']
    assert large['training'][:4000] == small['training']
    for key in ('model', 'revision', 'source_sha256', 'versions', 'max_len', 'pooling_method', 'seed',
                'trainable_parameters', 'epochs', 'learning_rate', 'encoder_learning_rate', 'class_weight_power', 'lora_rank'):
        assert small['config'][key] == large['config'][key], key
    summary = {}
    for size, row in models.items():
        train, history, performance = row['training'], row['history'], row['performance']
        assert len(train) == int(size) == row['config']['samples']
        identities = [(r['body_sha256'], r['reference']) for r in train]
        assert hashlib.sha256(json.dumps(identities).encode()).hexdigest() == row['config']['train_subset_sha256']
        entry = {'training_documents': len(train), 'training_characters': sum(r['characters'] for r in train),
                 'training_classes': dict(Counter(r['reference'] for r in train)),
                 'training_families': dict(Counter(r['family'] for r in train)),
                 'selected_epoch': history['best_epoch'], 'epochs': history['epoch'], 'steps': history['steps'],
                 'epoch_selection_f1': [{'epoch': e['epoch'], 'macro_f1': e['validation']['macro_f1']} for e in history['history']],
                 'training_seconds': history['seconds'], 'training_peak_gib': performance['training_peak_vram_bytes'] / 2**30,
                 'inference_peak_gib': performance['peak_vram_bytes'] / 2**30,
                 'warm_mean_ms': performance['latency_seconds']['mean'] * 1000,
                 'warm_p50_ms': performance['latency_seconds']['p50'] * 1000,
                 'warm_p95_ms': performance['latency_seconds']['p95'] * 1000,
                 'temperature': row['decision']['temperature']}
        for split in ('selection', 'calibration'):
            predictions = row[split]['predictions']
            assert [(r['body_sha256'], r['group_id'], r['reference']) for r in predictions] == [
                (r['body_sha256'], r['group_id'], r['reference']) for r in small[split]['predictions']]
            entry[split] = metrics(predictions)
            raw = row[split]['raw_probabilities']
            entry[split + '_raw'] = metrics([{**r, 'probabilities': p, 'uncertainty': float(1 - max(p)), 'discard': {}}
                                            for r, p in zip(predictions, raw, strict=True)])
            np.testing.assert_allclose(entry[split]['macro_f1'], entry[split + '_raw']['macro_f1'], atol=1e-12)
        summary[size] = entry
    a, b = summary['4000'], summary['16000']
    execution = evidence['execution']
    return {'status': evidence['status'], 'freeze_sha256': evidence['freeze_sha256'], 'models': summary,
            'selection_f1_delta': b['selection']['macro_f1'] - a['selection']['macro_f1'],
            'selection_critical_miss_delta': b['selection']['default_discard']['dangerous_misses'] - a['selection']['default_discard']['dangerous_misses'],
            'training_time_ratio': b['training_seconds'] / a['training_seconds'],
            'warm_mean_ratio': b['warm_mean_ms'] / a['warm_mean_ms'],
            'extension_wall_seconds': (datetime.fromisoformat(execution['finished_utc']) - datetime.fromisoformat(execution['started_utc'])).total_seconds(),
            'limits': 'Post-test development choice; epoch-selected selection and calibration fitting evidence; no generalization or significance claim'}
