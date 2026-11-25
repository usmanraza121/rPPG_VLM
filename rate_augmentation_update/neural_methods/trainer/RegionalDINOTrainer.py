"""Separate experimental trainer; the existing DINOv2-TCN baseline is unchanged."""

import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

from neural_methods.loss.RegionalPulseLoss import (
    RegionalPulseLoss, estimate_hr, waveform_correlation,
)
from neural_methods.model.regional_dino import RegionalDINO
from neural_methods.augmentation.paired_rate import PairedRateAugment


def model_arguments(config):
    """Keep architecture options identical for training, preflight and testing."""
    return {key.lower(): value for key, value in config.MODEL.REGIONAL_DINO.items()}


def build_model(config):
    return RegionalDINO(**model_arguments(config))


def build_loss(config):
    settings = config.TRAIN.REGIONAL_DINO
    keys = ('MIN_HZ', 'MAX_HZ', 'MSE_WEIGHT', 'CORRELATION_WEIGHT',
            'SPECTRAL_WEIGHT', 'REGIONAL_WEIGHT', 'QUALITY_WEIGHT',
            'QUALITY_TEMPERATURE')
    return RegionalPulseLoss(fs=float(config.TRAIN.DATA.FS), **{
        key.lower(): settings[key] for key in keys
    })


def audit_batch(video, labels):
    """Check an actual cached batch, rather than trusting its cache directory name."""
    if video.ndim != 5 or video.shape[2] != 3:
        raise ValueError(f'Expected [B,T,3,H,W] RGB, received {tuple(video.shape)}.')
    if labels.shape != video.shape[:2]:
        raise ValueError(f'Expected labels {tuple(video.shape[:2])}, received {tuple(labels.shape)}.')
    if not torch.isfinite(video).all() or not torch.isfinite(labels).all():
        raise ValueError('Cached input or label contains NaN/Inf.')
    low, high = float(video.min()), float(video.max())
    if low < 0 or high > 255 or high <= 1:
        raise ValueError(
            f'Raw RGB [0,255] required; observed [{low:.5g},{high:.5g}]. '
            'Verify/rebuild the cache; standardized or [0,1] video is incompatible.'
        )
    return {'shape': list(video.shape), 'rgb_min': low, 'rgb_max': high,
            'mean_rgb_temporal_std': float(video.std(dim=1, unbiased=False).mean()),
            'target_std': float(labels.std(dim=-1, unbiased=False).mean())}


class RegionalDINOTrainer:
    """Pulse-oriented objective, full-set diagnostics, and two rotating checkpoints.

    Selection uses source validation only. Diagnostic clip HR is distinct from
    the established full-recording evaluator used for the final test report.
    Checkpoints before CHECKPOINT_START_EPOCH are deliberately ineligible.
    """

    def __init__(self, config, data_loader, model=None):
        self.config = config
        self.settings = config.TRAIN.REGIONAL_DINO
        self.device = torch.device(config.DEVICE)
        self.model_dir = Path(config.MODEL.MODEL_DIR)
        self.run_dir = Path(config.LOG.PATH)
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.model_dir / f'{self.model_file_name}_best.pth'
        self.last_path = self.model_dir / f'{self.model_file_name}_last.pth'
        self.best_epoch, self.best_score = None, float('inf')
        self._audit_configuration()
        self.model = (model if model is not None else build_model(config)).to(self.device)
        if self.device.type == 'cuda' and config.NUM_OF_GPU_TRAIN > 1:
            self.model = torch.nn.DataParallel(self.model,
                                             device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        self.criterion = build_loss(config).to(self.device)
        self.fs = float(config.TRAIN.DATA.FS)
        self.min_hz, self.max_hz = self.settings.MIN_HZ, self.settings.MAX_HZ
        self.rate_augment = None
        if self.settings.RATE_AUG.ENABLED and config.TOOLBOX_MODE == 'train_and_test':
            settings = {k.lower(): v for k, v in self.settings.RATE_AUG.items() if k != 'ENABLED'}
            self.rate_augment = PairedRateAugment(fs=self.fs, **settings)
            print('Training-only paired slowing:', json.dumps(settings))
        if config.TOOLBOX_MODE == 'train_and_test':
            train_loader = data_loader.get('train')
            if train_loader is None or len(train_loader) == 0:
                raise ValueError('RegionalDINO requires a nonempty training loader.')
            if config.TRAIN.EPOCHS <= self.settings.CHECKPOINT_START_EPOCH:
                raise ValueError('EPOCHS must exceed CHECKPOINT_START_EPOCH (epochs are zero-indexed).')
            self._build_optimizer(len(train_loader))
        elif config.TOOLBOX_MODE != 'only_test':
            raise ValueError('RegionalDINO supports train_and_test or only_test.')
        print('RegionalDINO architecture:', json.dumps(model_arguments(config)))
        print('Trainable parameters:', sum(p.numel() for p in self.core.parameters() if p.requires_grad))
        print('Grid cells are image regions, not anatomical masks or perfusion measurements.')

    @property
    def core(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _audit_configuration(self):
        names = ('TEST',) if self.config.TOOLBOX_MODE == 'only_test' else ('TRAIN', 'VALID', 'TEST')
        for name in names:
            data = self.config[name].DATA
            if (list(data.PREPROCESS.DATA_TYPE) != ['Raw']
                    or data.PREPROCESS.LABEL_TYPE != 'Standardized'
                    or data.DATA_FORMAT != 'NDCHW'):
                raise ValueError(f'{name}: require Raw RGB, Standardized labels and NDCHW.')
            if float(data.FS) != float(self.config.TRAIN.DATA.FS):
                raise ValueError('All split sample rates must match TRAIN.DATA.FS.')
        if self.settings.SELECTION_METRIC not in ('loss', 'clip_hr_mae'):
            raise ValueError('SELECTION_METRIC must be loss or clip_hr_mae.')
        if self.config.TRAIN.AUG:
            raise ValueError('Set TRAIN.AUG: 0; legacy model-specific augmentation is not used here.')
        if self.settings.GRAD_CLIP <= 0 or self.settings.CHECKPOINT_START_EPOCH < 0:
            raise ValueError('GRAD_CLIP must be positive and CHECKPOINT_START_EPOCH nonnegative.')

    def _build_optimizer(self, steps_per_epoch):
        head, backbone = [], []
        for name, parameter in self.core.named_parameters():
            if parameter.requires_grad:
                (backbone if name.startswith('backbone.') else head).append(parameter)
        head_lr, backbone_lr = float(self.config.TRAIN.LR), float(self.settings.BACKBONE_LR)
        if not head or head_lr <= 0 or backbone_lr <= 0:
            raise ValueError('Trainable pulse parameters and positive learning rates are required.')
        groups = [{'params': head, 'lr': head_lr}]
        max_rates = [head_lr]
        if backbone:
            groups.append({'params': backbone, 'lr': backbone_lr})
            max_rates.append(backbone_lr)
        self.optimizer = torch.optim.AdamW(groups, weight_decay=float(self.settings.WEIGHT_DECAY))
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=max_rates, epochs=self.config.TRAIN.EPOCHS,
            steps_per_epoch=steps_per_epoch,
        )
        print('OneCycle peak learning rates (pulse/context, optional DINO):', max_rates)

    def _run_epoch(self, loader, training=False):
        if loader is None:
            raise ValueError('A validation loader is required for source-domain model selection.')
        self.model.train(training)
        sums, count, elements = {}, 0, 0
        squared_error, zero_error = 0.0, 0.0
        augmentation_counts, speeds, augmentation_records = {}, [], []
        bar = tqdm(loader, desc='Train' if training else 'Source validation', ncols=95)
        with torch.set_grad_enabled(training):
            for step, batch in enumerate(bar):
                video, target = batch[0].float(), batch[1].float()
                if step == 0:
                    print('Input audit:', json.dumps(audit_batch(video, target)))
                video, target = video.to(self.device), target.to(self.device)
                if training and self.rate_augment is not None:
                    video, target, plans = self.rate_augment(video, target)
                    for item, plan in enumerate(plans):
                        reason = plan['reason']
                        augmentation_counts[reason] = augmentation_counts.get(reason, 0) + 1
                        if plan['applied']:
                            speeds.append(plan['speed'])
                        augmentation_records.append({
                            'recording': str(batch[2][item]), 'clip': int(batch[3][item]), **plan})
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                output = self.model(video, return_aux=True)
                losses = self.criterion(output, target)
                if not all(bool(torch.isfinite(value)) for value in losses.values()):
                    raise FloatingPointError('Nonfinite loss; inspect cache alignment and waveform outputs.')
                if training:
                    losses['total'].backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.core.parameters() if p.requires_grad],
                        float(self.settings.GRAD_CLIP), error_if_nonfinite=True,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                batch_size = target.shape[0]
                count += batch_size
                elements += target.numel()
                with torch.no_grad():
                    pred = output['waveform'].detach()
                    squared_error += float((pred - target).square().sum())
                    zero_error += float(target.square().sum())
                    values = {key: float(value.detach()) for key, value in losses.items()}
                    values['waveform_corr'] = float(waveform_correlation(pred, target).mean())
                    values['prediction_std'] = float(pred.std(dim=-1, unbiased=False).mean())
                    values['target_std'] = float(target.std(dim=-1, unbiased=False).mean())
                    target_hr = estimate_hr(target, self.fs, self.min_hz, self.max_hz)
                    pred_hr = estimate_hr(pred, self.fs, self.min_hz, self.max_hz)
                    usable = torch.isfinite(target_hr)
                    valid_prediction = torch.isfinite(pred_hr)
                    # Undefined predictions receive the full band-width penalty,
                    # rather than disappearing and making an empty model look good.
                    penalty = 60.0 * (self.max_hz - self.min_hz)
                    errors = torch.where(valid_prediction, (pred_hr - target_hr).abs(), penalty)
                    sums['hr_error_sum'] = sums.get('hr_error_sum', 0.0) + float(errors[usable].sum())
                    sums['hr_count'] = sums.get('hr_count', 0) + int(usable.sum())
                    values['undefined_hr_fraction'] = float((~valid_prediction).float().mean())
                    weights = output['region_weights'].detach().clamp_min(1e-8)
                    values['weight_entropy'] = float(-(weights * weights.log()).sum(dim=1).mean())
                    for key, value in values.items():
                        sums[key] = sums.get(key, 0.0) + value * batch_size
                bar.set_postfix(loss=f"{float(losses['total'].detach()):.3f}")
        if count == 0:
            raise ValueError('Loader produced no samples.')
        result = {key: value / count for key, value in sums.items()
                  if key not in ('hr_count', 'hr_error_sum')}
        result['clip_hr_mae'] = sums['hr_error_sum'] / sums['hr_count'] if sums['hr_count'] else None
        result['raw_mse'] = squared_error / elements
        result['zero_baseline_mse'] = zero_error / elements
        result['samples'] = count
        if training and self.rate_augment is not None:
            result['rate_augmentation'] = {
                'counts': augmentation_counts, 'applied_fraction': len(speeds) / count,
                'mean_speed': sum(speeds) / len(speeds) if speeds else None,
                'min_speed': min(speeds) if speeds else None,
                'max_speed': max(speeds) if speeds else None,
            }
            self._rate_augmentation_records = augmentation_records
        return result

    def valid(self, data_loader):
        return self._run_epoch(data_loader['valid'])

    def train(self, data_loader):
        history_path = self.run_dir / 'regional_history.jsonl'
        for epoch in range(self.config.TRAIN.EPOCHS):
            print(f'\n====RegionalDINO epoch {epoch}====')
            train_stats = self._run_epoch(data_loader['train'], training=True)
            if self.rate_augment is not None:
                with (self.run_dir / 'rate_augmentation.jsonl').open('a', encoding='utf-8') as stream:
                    for row in self._rate_augmentation_records:
                        stream.write(json.dumps({'epoch': epoch, **row}, allow_nan=False) + '\n')
            valid_stats = self.valid(data_loader)
            key = 'total' if self.settings.SELECTION_METRIC == 'loss' else 'clip_hr_mae'
            score = valid_stats[key]
            if score is None or not math.isfinite(score):
                raise ValueError('No finite validation selection score. Inspect target signal quality.')
            eligible = epoch >= self.settings.CHECKPOINT_START_EPOCH
            improved = eligible and score < self.best_score
            if improved:
                self.best_score, self.best_epoch = score, epoch
            record = {'epoch': epoch, 'train': train_stats, 'valid': valid_stats,
                      'learning_rates': [group['lr'] for group in self.optimizer.param_groups],
                      'selection_metric': self.settings.SELECTION_METRIC, 'selection_score': score,
                      'checkpoint_eligible': eligible, 'best_epoch': self.best_epoch}
            print(json.dumps(record, allow_nan=False))
            with history_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            if eligible:
                self._save_checkpoint(self.last_path, epoch, record)
                if improved:
                    self._save_checkpoint(self.best_path, epoch, record)
                    print('Updated best source-validation checkpoint:', self.best_path)
        print('Best eligible epoch:', self.best_epoch, 'score:', self.best_score)

    def _save_checkpoint(self, path, epoch, metrics):
        # State dictionaries only; no pickled model objects. Atomic replacement
        # avoids leaving a partially written checkpoint after interruption.
        payload = {'format_version': 1, 'epoch': epoch,
                   'model_arguments': model_arguments(self.config),
                   'model_state': self.core.state_dict(),
                   'optimizer_state': self.optimizer.state_dict(),
                   'scheduler_state': self.scheduler.state_dict(),
                   'metrics': metrics}
        if self.rate_augment is not None:
            payload['rate_augmentation_settings'] = dict(self.settings.RATE_AUG)
            payload['rate_augmentation_state'] = self.rate_augment.state_dict()
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    def _load_checkpoint(self, path):
        if not path.is_file():
            raise FileNotFoundError(f'Checkpoint does not exist: {path}')
        # Explicit weights_only is also available on the baseline torch 2.0.
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if 'model_state' not in payload:
            raise ValueError('Expected a RegionalDINO checkpoint, not a DINOv2-TCN baseline checkpoint.')
        if payload['model_arguments'] != model_arguments(self.config):
            raise ValueError('Model settings differ from checkpoint; use the training architecture settings.')
        self.core.load_state_dict(payload['model_state'], strict=True)
        print('Loaded RegionalDINO epoch:', payload['epoch'], 'from', path)

    def test(self, data_loader):
        from evaluation.metrics import calculate_metrics

        loader = data_loader.get('test')
        if loader is None or len(loader) == 0:
            raise ValueError('No test data.')
        path = (Path(self.config.INFERENCE.MODEL_PATH) if self.config.TOOLBOX_MODE == 'only_test'
                else self.last_path if self.config.TEST.USE_LAST_EPOCH else self.best_path)
        self._load_checkpoint(path)
        self.model.eval()
        predictions, references, regional = {}, {}, {}
        with torch.no_grad():
            for step, batch in enumerate(tqdm(loader, desc='Test', ncols=95)):
                video, target = batch[0].float(), batch[1].float()
                if step == 0:
                    print('Test input audit:', json.dumps(audit_batch(video, target)))
                output = self.model(video.to(self.device), return_aux=True)
                for item in range(video.shape[0]):
                    recording, clip = str(batch[2][item]), int(batch[3][item])
                    predictions.setdefault(recording, {})[clip] = output['waveform'][item].cpu().reshape(-1, 1)
                    references.setdefault(recording, {})[clip] = target[item].cpu().reshape(-1, 1)
                    regional.setdefault(recording, {})[clip] = {
                        key: value[item].detach().cpu() for key, value in output.items()
                        if key != 'waveform'
                    }
        torch.save({'pred_raw': predictions, 'gt': references, 'regional': regional,
                    'fs': float(self.config.TEST.DATA.FS),
                    'grid_size': self.core.grid_size, 'normalization': 'raw',
                    'label_type': self.config.TEST.DATA.PREPROCESS.LABEL_TYPE},
                   self.run_dir / 'regional_waveforms.pt')
        print('Final metrics use raw clips joined by recording; no per-clip z-score amplification.')
        print('The legacy evaluator retains its historical RMSE-uncertainty/SNR definitions.')
        calculate_metrics(predictions, references, self.config)
