"""Audit cached clips and fit a fixed subset before committing to a full run.

Run from the repository root. No raw dataset preprocessing or training-run
directory is created, and no checkpoint is saved. Synthetic mode tests wiring,
not rPPG accuracy. --no-dino is an explicitly labelled offline pulse-path check.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    # Match main.py's existing physical GPU default while honoring the caller.
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config
from neural_methods.trainer.RegionalDINOTrainer import (
    audit_batch, build_loss, build_model,
)
from neural_methods.loss.RegionalPulseLoss import waveform_correlation


def load_cached_subset(config, clips):
    """Mirror BaseLoader's cached NDHWC-to-NDCHW transform without detector imports."""
    file_list = Path(config.TRAIN.DATA.FILE_LIST_PATH)
    if not file_list.is_file():
        raise FileNotFoundError(f'Training file list not available on this host: {file_list}')
    with file_list.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or 'input_files' not in rows[0]:
        raise ValueError('Expected a Toolbox CSV with input_files column.')
    # Evenly spaced entries avoid choosing only adjacent chunks of one recording.
    positions = np.linspace(0, len(rows) - 1, min(clips, len(rows)), dtype=int)
    videos, targets = [], []
    for position in positions:
        input_path = Path(rows[position]['input_files'])
        if '_input' not in input_path.name:
            raise ValueError(f'Expected *_input<clip>.npy, received {input_path}')
        label_path = input_path.with_name(input_path.name.replace('_input', '_label'))
        video = np.load(input_path, allow_pickle=False)
        label = np.load(label_path, allow_pickle=False)
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(f'Cached {input_path} must contain [T,H,W,3] raw RGB.')
        videos.append(torch.from_numpy(video.transpose(0, 3, 1, 2).copy()).float())
        targets.append(torch.from_numpy(label.copy()).float())
        print('Cached pair:', input_path, label_path)
    return torch.stack(videos), torch.stack(targets)


def synthetic_subset(config, clips):
    length = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
    height = config.TRAIN.DATA.PREPROCESS.RESIZE.H
    width = config.TRAIN.DATA.PREPROCESS.RESIZE.W
    fs = float(config.TRAIN.DATA.FS)
    time = torch.arange(length) / fs
    frequency = torch.linspace(1.0, 1.8, clips)
    target = torch.sin(2 * torch.pi * frequency[:, None] * time[None])
    target = (target - target.mean(-1, keepdim=True)) / target.std(-1, keepdim=True)
    texture = torch.rand(clips, 1, 3, height, width) * 35 + 100
    color = torch.tensor([0.35, 1.0, 0.15]).view(1, 1, 3, 1, 1)
    video = texture + 2 * target[:, :, None, None, None] * color
    return video, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/train_configs/cross/MMPD_PURE_RegionalDINO.yaml')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--clips', type=int, default=4)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--no-dino', action='store_true')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.clips < 1 or args.steps < 0 or args.threads < 1:
        parser.error('clips and threads must be positive; steps must be nonnegative.')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = get_config(argparse.Namespace(config_file=args.config))
    if args.no_dino:
        config.defrost()
        config.MODEL.REGIONAL_DINO.USE_DINO = False
        config.MODEL.REGIONAL_DINO.FINETUNE_BLOCKS = 0
        config.freeze()
    video, target = (synthetic_subset(config, args.clips) if args.synthetic
                     else load_cached_subset(config, args.clips))
    print('Scope:', 'SYNTHETIC wiring check' if args.synthetic else 'Fixed MMPD training subset')
    print('Use DINO:', config.MODEL.REGIONAL_DINO.USE_DINO)
    print('Audit:', json.dumps(audit_batch(video, target)))
    video, target = video.to(args.device), target.to(args.device)
    model = build_model(config).to(args.device)
    criterion = build_loss(config).to(args.device)
    groups = []
    for backbone_group, rate in ((False, config.TRAIN.LR),
                                 (True, config.TRAIN.REGIONAL_DINO.BACKBONE_LR)):
        params = [value for name, value in model.named_parameters()
                  if value.requires_grad and name.startswith('backbone.') == backbone_group]
        if params:
            groups.append({'params': params, 'lr': float(rate)})
    # Constant LR, no augmentation and eval-mode dropout for a fixed-batch test.
    # eval() does not disable gradients or freeze the selected DINO parameters.
    optimizer = torch.optim.AdamW(groups, weight_decay=0)
    model.eval()

    def report(step):
        with torch.no_grad():
            output = model(video, return_aux=True)
            pred = output['waveform']
            record = {key: float(value) for key, value in criterion(output, target).items()}
            record.update(step=step, raw_mse=float((pred - target).square().mean()),
                          zero_mse=float(target.square().mean()),
                          waveform_corr=float(waveform_correlation(pred, target).mean()),
                          prediction_std=float(pred.std(-1, unbiased=False).mean()),
                          target_std=float(target.std(-1, unbiased=False).mean()))
            print(json.dumps(record, allow_nan=False))

    report(0)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(video, return_aux=True)
        losses = criterion(output, target)
        if not torch.isfinite(losses['total']):
            raise FloatingPointError('Nonfinite loss in fixed-subset test.')
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
        if step in (0, args.steps - 1):
            for prefix in ('pulse_stem.', 'temporal_encoder.', 'quality_head.', 'backbone.'):
                norm = sum(float(p.grad.norm()) for name, p in model.named_parameters()
                           if name.startswith(prefix) and p.grad is not None)
                print('Gradient norm sum:', prefix, norm)
        optimizer.step()
        if (step + 1) % max(1, args.steps // 5) == 0 or step + 1 == args.steps:
            report(step + 1)

    # A deterministic static input should not produce a periodic pulse template.
    with torch.no_grad():
        static = video[:, :1].expand_as(video)
        static_pred = model(static)
        print('Static-input output max abs:', float(static_pred.abs().max()))
    if args.synthetic:
        print('No checkpoint saved. Synthetic improvement does not establish physiological accuracy.')
    else:
        print('No checkpoint saved. Fitting these training clips does not establish held-out accuracy.')


if __name__ == '__main__':
    main()
