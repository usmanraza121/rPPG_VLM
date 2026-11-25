"""Preflight paired slowing on source-training cache only; no model/training."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config
from neural_methods.augmentation.paired_rate import PairedRateAugment, paired_resample
from tools.regional_audit_data import load_cached_clip, read_cache_records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-clips', type=int, default=64)
    args = parser.parse_args(argv)
    if args.max_clips < 1:
        parser.error('--max-clips must be positive.')
    torch.set_num_threads(2)
    config = get_config(argparse.Namespace(config_file=args.config))
    d = config.TRAIN.DATA
    if (not config.TRAIN.REGIONAL_DINO.RATE_AUG.ENABLED
            or list(d.PREPROCESS.DATA_TYPE) != ['Raw']
            or d.PREPROCESS.LABEL_TYPE != 'Standardized' or d.DATA_FORMAT != 'NDCHW'):
        raise ValueError('Require enabled rate augmentation, Raw RGB, Standardized PPG and NDCHW.')
    settings = {k.lower(): v for k, v in config.TRAIN.REGIONAL_DINO.RATE_AUG.items() if k != 'ENABLED'}
    augmentation = PairedRateAugment(fs=float(d.FS), **settings)
    # Independent known-frequency mechanical control.
    t = torch.arange(160) / float(d.FS)
    label = torch.sin(2 * torch.pi * 1.5 * t)
    video = (120 + 3 * label[:, None, None, None]).expand(-1, 3, 3, 3).clone()
    x, y = paired_resample(video, label, .7, 10.)
    error = float(((x[:, 0, 0, 0] - 120) / 3 - y).abs().max())
    if error > 1e-5:
        raise AssertionError('Joint video/label interpolation control failed.')
    records = read_cache_records(config, 'train')
    # First distribute samples across recordings; do not read validation/test.
    candidates = [(record['recording'], record['subject'], *clip)
                  for record in records for clip in record['clips']]
    indices = np.unique(np.linspace(0, len(candidates) - 1,
                                   min(args.max_clips, len(candidates)), dtype=int))
    rows, counts = [], {}
    for index in indices:
        recording, subject, clip, input_path, label_path = candidates[index]
        video, target = load_cached_clip(input_path, label_path)
        altered, labels, plans = augmentation(video[None], target[None])
        if not torch.isfinite(altered).all() or not torch.isfinite(labels).all():
            raise AssertionError(f'Nonfinite augmented output: {recording}, clip{clip}')
        if altered.shape != video[None].shape or labels.shape != target[None].shape:
            raise AssertionError('Augmentation changed clip dimensions.')
        plan = plans[0]
        counts[plan['reason']] = counts.get(plan['reason'], 0) + 1
        rows.append({'recording': recording, 'subject': subject, 'clip': clip, **plan})
        print('Checked:', recording, clip, plan['reason'], flush=True)
    if counts.get('applied', 0) == 0:
        raise ValueError('No sampled clips were augmented. Inspect references/settings before training.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result = {'config': args.config, 'settings': settings,
              'train_file_list_sha256': hashlib.sha256(Path(d.FILE_LIST_PATH).read_bytes()).hexdigest(),
              'sampled_clips': len(rows), 'counts': counts,
              'applied_fraction': counts.get('applied', 0) / len(rows),
              'synthetic_shared_coordinate_max_error': error,
              'sampled_subjects': sorted({r['subject'] for r in rows}),
              'plans': rows, 'training_or_cache_modified': False,
              'warning': 'HR eligibility is a coarse short-clip proxy. This checks mechanics, not physiological accuracy or expected performance.'}
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'plans'}, indent=2))
    return result


if __name__ == '__main__':
    main()
