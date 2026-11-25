"""Validation-only HR protocol and source-training rate diagnostics.

Does not train, modify labels/checkpoints, or evaluate test waveforms. The test
CSV is read only to verify subject membership. The wider filter is a candidate
motivated by an already observed test failure, not a confirmed improvement.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import numpy as np
from scipy import signal
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import get_config
from neural_methods.trainer.RegionalDINOTrainer import build_model, model_arguments
from tools.audit_regional_dino import infer_segment
from tools.regional_audit_data import contiguous_segments, read_cache_records
from tools.regional_signal_diagnostics import classical_signals, detrend_signal


PROTOCOLS = ('current', 'detrend_only', 'wider_filter')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def processed(values, protocol, fs):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < max(30, int(fs)) or not np.isfinite(values).all():
        raise ValueError('Require a finite waveform of at least one second and 30 samples.')
    if protocol not in PROTOCOLS:
        raise ValueError(f'Unknown protocol: {protocol}')
    result = detrend_signal(values)
    if protocol != 'detrend_only':
        band = (.75, 2.5) if protocol == 'current' else (.6, 3.3)
        if band[1] >= fs / 2:
            raise ValueError('Sample rate does not support the candidate filter.')
        b, a = signal.butter(1, band, btype='bandpass', fs=fs)
        result = signal.filtfilt(b, a, result,
                                padlen=min(3 * max(len(a), len(b)), len(result) - 1))
    return result


def peak_hr(values, protocol, fs):
    """Identical 45..150 bpm search for every processing option; no GT input."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Require a finite 1D waveform.')
    if float(np.std(values)) <= 1e-12:
        return None
    result = processed(values, protocol, fs)
    if float(np.std(result)) <= 1e-12:
        return None
    nperseg = min(256, len(result))
    f, p = signal.welch(result, fs=fs, window='hann', nperseg=nperseg,
                        noverlap=nperseg // 2, nfft=max(3333, nperseg),
                        detrend=False, scaling='density')
    indices = np.flatnonzero((f >= .75) & (f <= 2.5))
    if not len(indices) or float(p[indices].sum()) <= 1e-12:
        return None
    return float(f[indices[np.argmax(p[indices])]] * 60)


def disjoint_subjects(groups):
    sets = {name: {record['subject'] for record in records}
            for name, records in groups.items()}
    for name, subjects in sets.items():
        if not subjects:
            raise ValueError(f'Empty subject set: {name}')
    names = list(sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            # Subject IDs belong to a dataset. Cross-dataset names need not
            # identify the same person, so callers group by dataset first.
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(f'{left}/{right} subject overlap: {sorted(overlap)}')
    return {name: sorted(subjects) for name, subjects in sets.items()}


def comparison_rows(prediction, reference, fs, metadata, method):
    """Hold reference HR fixed; alternative reference HR is diagnostic only."""
    fixed_reference = peak_hr(reference, 'current', fs)
    rows = []
    for protocol in PROTOCOLS:
        estimate = peak_hr(prediction, protocol, fs)
        rows.append({**metadata, 'method': method, 'protocol': protocol,
                     'fixed_reference_hr_bpm': fixed_reference,
                     'reference_hr_under_this_protocol_bpm': peak_hr(reference, protocol, fs),
                     'prediction_hr_bpm': estimate,
                     'absolute_error_bpm': (abs(estimate - fixed_reference)
                                            if estimate is not None and fixed_reference is not None
                                            else None)})
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)


def coverage(records, fs):
    """Load only source labels, keeping gaps separate and reporting clip rates."""
    clips, segments = [], []
    for record in tqdm(records, desc='Training labels', ncols=90):
        for segment_index, segment in enumerate(contiguous_segments(record)):
            references, lengths = [], set()
            for clip_id, _, label_path in segment:
                values = np.load(label_path, allow_pickle=False)
                if values.ndim != 1 or not np.isrealobj(values) or not np.isfinite(values).all():
                    raise ValueError(f'Invalid cached reference: {label_path}')
                references.append(values)
                lengths.add(len(values))
                clips.append({'recording': record['recording'], 'subject': record['subject'],
                              'segment': segment_index, 'clip': clip_id,
                              'reference_hr_bpm': peak_hr(values, 'current', fs),
                              'detrend_only_reference_hr_bpm': peak_hr(values, 'detrend_only', fs)})
            if len(lengths) != 1:
                raise ValueError('Clip lengths differ within a recording; verify the cache.')
            values = np.concatenate(references)
            clip_rates = [peak_hr(value, 'current', fs) for value in references]
            finite = [value for value in clip_rates if value is not None]
            segments.append({'recording': record['recording'], 'subject': record['subject'],
                             'segment': segment_index, 'first_clip': segment[0][0],
                             'last_clip': segment[-1][0], 'frames': len(values),
                             'duration_seconds': len(values) / fs,
                             'reference_hr_bpm': peak_hr(values, 'current', fs),
                             'detrend_only_reference_hr_bpm': peak_hr(values, 'detrend_only', fs),
                             'median_clip_hr_bpm': float(np.median(finite)) if finite else None})
    return clips, segments


def rate_bins(rows):
    rates = [row['reference_hr_bpm'] for row in rows if row['reference_hr_bpm'] is not None]
    return {'total': len(rows), 'undefined': len(rows) - len(rates),
            'below60': sum(v < 60 for v in rates),
            '60to80': sum(60 <= v < 80 for v in rates),
            '80to100': sum(80 <= v < 100 for v in rates),
            '100and_above': sum(v >= 100 for v in rates)}


def summarize(rows):
    result = []
    for method in sorted({r['method'] for r in rows}):
        for protocol in PROTOCOLS:
            selected = [r for r in rows if r['method'] == method and r['protocol'] == protocol]
            errors = [r['absolute_error_bpm'] for r in selected if r['absolute_error_bpm'] is not None]
            result.append({'method': method, 'protocol': protocol, 'windows': len(selected),
                           'usable': len(errors), 'unusable': len(selected) - len(errors),
                           'mae_bpm': float(np.mean(errors)) if errors else None,
                           'rmse_bpm': float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                           'within5': sum(v <= 5 for v in errors)})
    return result


def synthetic_cases(fs):
    rows = []
    # True input frequency is independent of the contact reference/legacy filter.
    for duration in (160 / fs, 64.0):
        time = np.arange(round(duration * fs)) / fs
        for bpm in (45.0, 45.36, 48.0, 55.0, 75.0, 100.0, 140.0, 150.0):
            for amplitude in (0.0, .25, .5, .65, 1.0):
                for phase in (0.0, math.pi / 2):
                    values = (np.sin(2 * np.pi * bpm / 60 * time + phase)
                              + amplitude * np.sin(4 * np.pi * bpm / 60 * time + .3))
                    for protocol in PROTOCOLS:
                        estimate = peak_hr(values, protocol, fs)
                        rows.append({'duration_seconds': duration, 'true_hr_bpm': bpm,
                                     'second_harmonic_amplitude': amplitude, 'phase': phase,
                                     'protocol': protocol, 'prediction_hr_bpm': estimate,
                                     'absolute_error_bpm': abs(estimate - bpm) if estimate is not None else None})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1:
        parser.error('Require positive batch-size and threads.')
    torch.set_num_threads(args.threads)
    config = get_config(argparse.Namespace(config_file=args.config))
    groups = {name: read_cache_records(config, name) for name in ('train', 'valid', 'test')}
    subjects = {}
    for dataset in {config[name.upper()].DATA.DATASET for name in groups}:
        subjects[dataset] = disjoint_subjects({name: records for name, records in groups.items()
                                              if config[name.upper()].DATA.DATASET == dataset})
    print('Verified subject membership:', json.dumps(subjects))
    fs = float(config.TRAIN.DATA.FS)
    for name in ('TRAIN', 'VALID'):
        d = config[name].DATA
        if (list(d.PREPROCESS.DATA_TYPE) != ['Raw'] or d.PREPROCESS.LABEL_TYPE != 'Standardized'
                or d.DATA_FORMAT != 'NDCHW' or float(d.FS) != fs):
            raise ValueError('Require matching Raw RGB / Standardized PPG / NDCHW source caches.')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if payload.get('model_arguments') != model_arguments(config):
        raise ValueError('Checkpoint architecture differs from config.')
    model = build_model(config)
    model.load_state_dict(payload['model_state'], strict=True)
    model.to(args.device).eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'command_arguments': vars(args), 'checkpoint_epoch': payload['epoch'],
                'checkpoint_sha256': file_hash(args.checkpoint), 'subjects': subjects,
                'file_list_sha256': {n: file_hash(config[n.upper()].DATA.FILE_LIST_PATH) for n in groups},
                'model_arguments': model_arguments(config), 'resolved_config': config.dump(),
                'reference_policy': 'Fixed current-protocol contact-PPG HR for all primary comparisons.',
                'protocols': {'current': 'Tarvainen100 + order1 zero-phase .75..2.5Hz filter',
                              'detrend_only': 'Tarvainen100 without bandpass',
                              'wider_filter': 'Tarvainen100 + order1 zero-phase .6..3.3Hz filter'},
                'hr_search_all_protocols': '45..150bpm; Welch256,50%overlap,nfft3333',
                'test_waveforms_loaded': False,
                'warning': 'Wider-filter candidate follows a test-case diagnosis; this is validation analysis, not new benchmark performance.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    train_clips, train_segments = coverage(groups['train'], fs)
    write_csv(output / 'training_clip_hr.csv', train_clips)
    write_csv(output / 'training_recording_hr.csv', train_segments)
    rows = []
    for record in tqdm(groups['valid'], desc='Validation only', ncols=90):
        for segment_index, clips in enumerate(contiguous_segments(record)):
            values = infer_segment(clips, model, config.MODEL.REGIONAL_DINO.GRID_SIZE,
                                   args.batch_size, args.device)
            metadata = {'recording': record['recording'], 'subject': record['subject'],
                        'segment': segment_index, 'frames': len(values['reference'])}
            methods = {'learned': values['learned'], 'uniform': values['uniform'],
                       **classical_signals(values['rgb_global'], fs)}
            for name, prediction in methods.items():
                rows.extend(comparison_rows(prediction, values['reference'], fs, metadata, name))
    write_csv(output / 'validation_protocols.csv', rows)
    synthetic = synthetic_cases(fs)
    write_csv(output / 'synthetic_frequency_response.csv', synthetic)
    summary = {'validation': summarize(rows), 'training_clip_hr_bins': rate_bins(train_clips),
               'training_recording_hr_bins': rate_bins(train_segments),
               'training_per_subject': {subject: rate_bins([r for r in train_segments if r['subject'] == subject])
                                        for subject in sorted({r['subject'] for r in train_segments})},
               'validation_reference_hr_changes': {
                   p: sum(r['fixed_reference_hr_bpm'] is not None
                          and r['reference_hr_under_this_protocol_bpm'] is not None
                          and abs(r['fixed_reference_hr_bpm'] - r['reference_hr_under_this_protocol_bpm']) > 5
                          for r in rows if r['method'] == 'learned' and r['protocol'] == p)
                   for p in PROTOCOLS},
               'synthetic_cases': len(synthetic), 'training_or_model_modified': False,
               'warning': 'Labels supply HR proxies, not certified physiology. Clip HR can be coarse; inspect recording/clip disagreement and per-subject coverage. No protocol is automatically selected.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, allow_nan=False))
    print('Saved:', output, '\nNo training, test inference, label changes or evaluator changes performed.')
    return summary


if __name__ == '__main__':
    main()
