"""Frozen attention candidates with matched moment/optional pulse-summary gates.

Pulse summaries use predictions alone. They are relative descriptors, not
physiological confidence. No peak is selected with reference PPG at inference.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from neural_methods.model.regional_st_attention import RegionalSTAttention


def candidate_summaries(candidates, valid, fs=30., min_hz=.75, max_hz=2.5):
    """Six detached per-region, per-clip descriptors [B,R,6].

    Relative log-RMS, band power fraction, normalized spectral entropy, peak
    concentration, leave-one-out spectral cosine agreement, signed waveform
    correlation with the mean of other valid/nonflat regions. Agreement can
    reflect nuisance, so no descriptor is called a pulse probability.
    """
    q = candidates.detach().float()
    q = q - q.mean(-1, keepdim=True)
    eps = 1e-8
    rms = q.square().mean(-1).sqrt()
    active = valid & (rms > 1e-6)
    count = active.sum(1, keepdim=True)
    mean_rms = (rms * active).sum(1, keepdim=True) / count.clamp_min(1)
    relative_rms = ((rms + eps)/(mean_rms + eps)).log().clamp(-5, 5)/5
    length = q.shape[-1]
    nfft = max(256, 1 << (length - 1).bit_length())
    window = torch.hann_window(length, periodic=False, device=q.device, dtype=q.dtype)
    power = torch.fft.rfft(q * window, n=nfft, dim=-1).abs().square()
    frequency = torch.fft.rfftfreq(nfft, d=1/fs, device=q.device)
    band = (frequency >= min_hz) & (frequency <= max_hz)
    if int(band.sum()) < 2:
        raise ValueError('Pulse summaries require at least two in-band FFT bins.')
    band_power = power[..., band]
    band_total = band_power.sum(-1)
    fraction = band_total / power[..., 1:].sum(-1).clamp_min(eps)
    distribution = band_power / band_total.unsqueeze(-1).clamp_min(eps)
    entropy = -(distribution * distribution.clamp_min(eps).log()).sum(-1) / math.log(int(band.sum()))
    concentration = distribution.max(-1).values
    masked_distribution = distribution * active.unsqueeze(-1)
    other_distribution = masked_distribution.sum(1, keepdim=True) - masked_distribution
    spectral_agreement = F.cosine_similarity(masked_distribution, other_distribution, dim=-1, eps=eps)
    masked_q = q * active.unsqueeze(-1)
    other_q = (masked_q.sum(1, keepdim=True) - masked_q) / (count-active.long()).clamp_min(1).unsqueeze(-1)
    covariance = (q * other_q).mean(-1)
    waveform_agreement = covariance / (rms * other_q.square().mean(-1).sqrt()).clamp_min(eps)
    result = torch.stack((relative_rms, fraction.clamp(0, 1), entropy.clamp(0, 1),
                          concentration.clamp(0, 1), spectral_agreement.clamp(0, 1),
                          waveform_agreement.clamp(-1, 1)), dim=-1)
    return result.masked_fill(~active.unsqueeze(-1), 0)


class RegionalFrozenFusion(nn.Module):
    """Train only a new gate on fixed, eval-mode RegionalSTAttention outputs.

    Both modes use the same 18->context_dim->1 gate and uniform initialization.
    'moments' supplies six zero channels; 'waveform' supplies six clip summaries.
    Thus the trainable tensor shapes match; the null-input control has fewer
    effective input features, not an independently matched information capacity.
    """
    def __init__(self, gate_mode='waveform', temporal_backend='attention', spatial_size=12,
                 hidden_dim=48, grid_size=3, context_dim=32, dropout=.1, fusion='learned',
                 fs=30., min_hz=.75, max_hz=2.5):
        super().__init__()
        if gate_mode not in ('moments', 'waveform'):
            raise ValueError('gate_mode must be moments or waveform.')
        if temporal_backend != 'attention' or fusion != 'learned':
            raise ValueError('Frozen fusion experiment requires the learned attention extractor.')
        if not (math.isfinite(fs) and 0 < min_hz < max_hz < fs/2):
            raise ValueError('Require 0 < min_hz < max_hz < fs/2.')
        self.extractor_arguments = dict(temporal_backend=temporal_backend, spatial_size=spatial_size,
                                        hidden_dim=hidden_dim, grid_size=grid_size,
                                        context_dim=context_dim, dropout=dropout, fusion=fusion)
        self.extractor = RegionalSTAttention(**self.extractor_arguments)
        self.extractor.requires_grad_(False)
        self.extractor.eval()
        self.grid_size, self.num_regions, self.use_dino = grid_size, grid_size**2, False
        self.gate_mode = gate_mode
        self.fs, self.min_hz, self.max_hz = fs, min_hz, max_hz
        self.quality_smoothing = self.extractor.quality_smoothing
        self.quality_head = nn.Sequential(nn.Conv1d(18, context_dim, 1), nn.GELU(), nn.Conv1d(context_dim, 1, 1))
        nn.init.zeros_(self.quality_head[-1].weight)
        nn.init.zeros_(self.quality_head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.extractor.eval()  # Frozen dropout must not change the candidates.
        return self

    def load_extractor_checkpoint(self, payload):
        if (not isinstance(payload, dict) or payload.get('model_name') != 'RegionalSTAttention'
                or payload.get('model_arguments') != self.extractor_arguments):
            raise ValueError('Source checkpoint must match the original RegionalSTAttention architecture.')
        self.extractor.load_state_dict(payload['model_state'], strict=True)
        self.extractor.requires_grad_(False)
        self.extractor.eval()

    def _moments(self, video):
        rgb = video.float()/255.
        previous = torch.cat((rgb[:, :1], rgb[:, :-1]), dim=1)
        delta = (rgb-previous)/(rgb+previous+1e-3)
        height, width = rgb.shape[-2:]
        features = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                y, z = row*height//self.grid_size, (row+1)*height//self.grid_size
                x, end = col*width//self.grid_size, (col+1)*width//self.grid_size
                color, change = rgb[..., y:z, x:end], delta[..., y:z, x:end]
                features.append(torch.cat((color.mean((-1, -2)), change.mean((-1, -2)),
                                           color.square().mean((-1, -2)), change.square().mean((-1, -2))), dim=-1))
        return torch.stack(features, dim=1)  # B,R,T,12

    def forward(self, video, return_aux=False, region_mask=None):
        with torch.no_grad():
            output = self.extractor(video, return_aux=True, region_mask=region_mask)
            valid = output['valid_regions']
            regional = output['regional_waveforms']
            # The old gate only contributes one shared DC shift to these outputs.
            # Remove it regionwise to recover the gate-independent candidates.
            regional = regional-regional.mean(-1, keepdim=True)
            moments = self._moments(video).masked_fill(~valid[:, :, None, None], 0)
            summaries = (candidate_summaries(regional, valid, self.fs, self.min_hz, self.max_hz)
                         if self.gate_mode == 'waveform' else regional.new_zeros((*regional.shape[:2], 6)))
        batch, regions, time = regional.shape
        extra = summaries.unsqueeze(2).expand(-1, -1, time, -1)
        features = torch.cat((moments, extra), dim=-1).permute(0, 1, 3, 2).reshape(batch*regions, 18, time)
        logits = self.quality_head(features)
        pad = self.quality_smoothing//2
        logits = F.avg_pool1d(F.pad(logits, (pad, pad), mode='replicate'), self.quality_smoothing, stride=1)
        logits = logits.reshape(batch, regions, time)
        weights = logits.masked_fill(~valid[:, :, None], -torch.inf).softmax(1)
        logits = logits.masked_fill(~valid[:, :, None], 0)
        fused_dc = (weights*regional).sum(1, keepdim=True).mean(-1, keepdim=True)
        regional = (regional-fused_dc).masked_fill(~valid[:, :, None], 0)
        waveform = (weights*regional).sum(1)
        if not return_aux:
            return waveform
        return dict(waveform=waveform, regional_waveforms=regional, region_weights=weights,
                    quality_logits=logits, valid_regions=valid)
