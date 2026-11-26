"""Waveform, regional pulse, and reliability losses using only PyTorch.

The reference PPG is used here for training supervision. The model's regional
weights and waveform must be computable without a reference at inference.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _floating(signal):
    """Use an FFT-safe dtype while preserving float64 diagnostic inputs."""
    if not torch.is_tensor(signal):
        raise TypeError("Signals must be torch tensors.")
    if signal.ndim < 1 or signal.shape[-1] < 1:
        raise ValueError("Signals need a nonempty final time dimension.")
    if not signal.is_floating_point() or signal.dtype in (torch.float16, torch.bfloat16):
        signal = signal.float()
    return signal


def _centered_variance(signal):
    centered = signal - signal.mean(dim=-1, keepdim=True)
    return centered, centered.square().mean(dim=-1)


def waveform_correlation(pred, target, eps=1e-6):
    """Return a safe Pearson correlation over time, without batch reduction.

    Leading dimensions may broadcast. Additive variance regularization keeps
    gradients finite, including for a constant prediction and nonconstant target.
    A constant waveform has correlation zero rather than an undefined value.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")
    pred = _floating(pred)
    target = _floating(target).to(device=pred.device, dtype=pred.dtype)
    if pred.shape[-1] != target.shape[-1]:
        raise ValueError("Prediction and target time dimensions must match.")
    pred_centered, pred_var = _centered_variance(pred)
    target_centered, target_var = _centered_variance(target)
    covariance = (pred_centered * target_centered).mean(dim=-1)
    denominator = torch.sqrt((pred_var + eps) * (target_var + eps))
    return (covariance / denominator).clamp(-1.0, 1.0)


def _spectral_features(signal, fs, min_hz, max_hz, eps):
    signal = _floating(signal)
    if not (math.isfinite(fs) and fs > 0):
        raise ValueError("fs must be finite and positive.")
    if not (math.isfinite(min_hz) and math.isfinite(max_hz) and 0 < min_hz < max_hz <= fs / 2):
        raise ValueError("Require 0 < min_hz < max_hz <= fs / 2.")
    if eps <= 0:
        raise ValueError("eps must be positive.")
    length = signal.shape[-1]
    nfft = max(256, 1 << (length - 1).bit_length())
    centered, variance = _centered_variance(signal)
    window = torch.hann_window(length, periodic=False, dtype=signal.dtype, device=signal.device)
    spectrum = torch.fft.rfft(centered * window, n=nfft, dim=-1)
    frequencies = torch.fft.rfftfreq(nfft, d=1.0 / fs, device=signal.device, dtype=signal.dtype)
    in_band = (frequencies >= min_hz) & (frequencies <= max_hz)
    if not bool(in_band.any()):
        raise ValueError("No FFT bins lie in the requested physiological band.")
    power = spectrum.real.square() + spectrum.imag.square()
    # Normalize the window energy so validity does not depend on clip length.
    power = power[..., in_band] / window.square().sum().clamp_min(eps)
    total_power = power.sum(dim=-1)
    # A uniform fallback is a proper distribution and gives finite JS/KL terms.
    distribution = (power + eps) / (total_power.unsqueeze(-1) + eps * power.shape[-1])
    usable = (variance > eps) & (total_power > eps)
    return distribution, frequencies[in_band], usable


def spectral_distribution(signal, fs=30.0, min_hz=0.75, max_hz=2.5, eps=1e-6):
    """Return ``(in_band_power_distribution, frequencies_hz)``.

    The final dimension contains the physiological frequency bins. Signals are
    centered, Hann-windowed, and zero-padded to a power of two of at least 256.
    Flat signals return a uniform distribution; this is a numerical fallback,
    not evidence that they contain a physiological signal.
    """
    distribution, frequencies, _ = _spectral_features(signal, fs, min_hz, max_hz, eps)
    return distribution, frequencies


def estimate_hr(signal, fs=30.0, min_hz=0.75, max_hz=2.5, eps=1e-6):
    """Return FFT-peak BPM per waveform; flat/unusable signals return NaN.

    This helper is for diagnostics. Use the established repository evaluator
    for protocol-comparable reported HR results.
    """
    distribution, frequencies, usable = _spectral_features(signal, fs, min_hz, max_hz, eps)
    bpm = frequencies[distribution.argmax(dim=-1)] * 60.0
    return torch.where(usable, bpm, torch.full_like(bpm, float("nan")))


def _masked_mean(values, mask):
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _js_divergence(first, second, eps):
    midpoint = 0.5 * (first + second)
    log_midpoint = midpoint.clamp_min(eps).log()
    first_kl = (first * (first.clamp_min(eps).log() - log_midpoint)).sum(dim=-1)
    second_kl = (second * (second.clamp_min(eps).log() - log_midpoint)).sum(dim=-1)
    return (0.5 * (first_kl + second_kl)).clamp_min(0.0)


class RegionalPulseLoss(nn.Module):
    """Supervise a fused waveform, regional candidates, and quality logits.

    ``outputs`` contains waveform [B,T], regional_waveforms [B,R,T],
    region_weights [B,R,T], quality_logits [B,R,T], and valid_regions [B,R].
    Candidate shape/periodicity is supervised without matching its raw amplitude.
    Quality targets are detached reference-derived regional errors, so gradients
    cannot improve the quality target by changing a candidate waveform.
    """

    def __init__(
        self,
        fs=30,
        min_hz=0.75,
        max_hz=2.5,
        mse_weight=0.1,
        correlation_weight=1.0,
        spectral_weight=1.0,
        regional_weight=0.25,
        quality_weight=0.1,
        quality_temperature=0.25,
        eps=1e-6,
    ):
        super().__init__()
        weights = (mse_weight, correlation_weight, spectral_weight, regional_weight, quality_weight)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("Loss weights must be finite and nonnegative.")
        if not math.isfinite(quality_temperature) or quality_temperature <= 0:
            raise ValueError("quality_temperature must be finite and positive.")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive.")
        if not (math.isfinite(fs) and fs > 0 and 0 < min_hz < max_hz <= fs / 2):
            raise ValueError("Require positive fs and 0 < min_hz < max_hz <= fs / 2.")
        self.fs = float(fs)
        self.min_hz = float(min_hz)
        self.max_hz = float(max_hz)
        self.mse_weight = float(mse_weight)
        self.correlation_weight = float(correlation_weight)
        self.spectral_weight = float(spectral_weight)
        self.regional_weight = float(regional_weight)
        self.quality_weight = float(quality_weight)
        self.quality_temperature = float(quality_temperature)
        self.eps = float(eps)

    def forward(self, outputs, target):
        waveform = _floating(outputs["waveform"])
        regional = _floating(outputs["regional_waveforms"])
        quality_logits = _floating(outputs["quality_logits"])
        region_weights = outputs["region_weights"]
        valid_regions = outputs["valid_regions"]
        target = _floating(target).to(device=waveform.device, dtype=waveform.dtype)
        if waveform.ndim != 2 or target.shape != waveform.shape:
            raise ValueError("waveform and target must have matching [B,T] shapes.")
        if regional.ndim != 3 or regional.shape[0] != waveform.shape[0] or regional.shape[-1] != waveform.shape[-1]:
            raise ValueError("regional_waveforms must have shape [B,R,T].")
        if regional.shape[1] < 1:
            raise ValueError("At least one regional candidate is required.")
        if quality_logits.shape != regional.shape or region_weights.shape != regional.shape:
            raise ValueError("quality_logits and region_weights must match regional_waveforms.")
        if valid_regions.shape != regional.shape[:2] or valid_regions.dtype != torch.bool:
            raise ValueError("valid_regions must be a bool tensor of shape [B,R].")
        if regional.device != waveform.device or quality_logits.device != waveform.device:
            raise ValueError("All model outputs must be on the same device.")
        valid_regions = valid_regions.to(device=waveform.device)

        _, target_variance = _centered_variance(target)
        target_has_variation = target_variance > self.eps
        target_distribution, _, target_has_power = _spectral_features(
            target, self.fs, self.min_hz, self.max_hz, self.eps
        )
        main_distribution, _, _ = _spectral_features(
            waveform, self.fs, self.min_hz, self.max_hz, self.eps
        )
        mse = F.mse_loss(waveform, target)
        main_correlation_error = 1.0 - waveform_correlation(waveform, target, self.eps)
        correlation = _masked_mean(main_correlation_error, target_has_variation)
        main_spectral_error = _js_divergence(main_distribution, target_distribution, self.eps)
        spectral = _masked_mean(main_spectral_error, target_has_power)

        regional_distribution, _, _ = _spectral_features(
            regional, self.fs, self.min_hz, self.max_hz, self.eps
        )
        regional_correlation_error = 1.0 - waveform_correlation(regional, target.unsqueeze(1), self.eps)
        regional_spectral_error = _js_divergence(
            regional_distribution, target_distribution.unsqueeze(1), self.eps
        )
        regional_errors = (
            regional_correlation_error * target_has_variation.unsqueeze(1)
            + regional_spectral_error * target_has_power.unsqueeze(1)
        )
        regional_usable = valid_regions & (target_has_variation | target_has_power).unsqueeze(1)
        regional_loss = _masked_mean(regional_errors, regional_usable)

        # Supervision ranks candidate regions, not time instants. Invalid regions
        # have exactly zero target probability and no quality-logit gradient.
        quality_scores = -regional_errors.detach() / self.quality_temperature
        finite_floor = -torch.finfo(quality_logits.dtype).max
        quality_target = F.softmax(quality_scores.masked_fill(~valid_regions, finite_floor), dim=-1)
        quality_target = quality_target * valid_regions.to(dtype=quality_target.dtype)
        quality_target = quality_target / quality_target.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        quality_log_probability = F.log_softmax(
            quality_logits.mean(dim=-1).masked_fill(~valid_regions, finite_floor), dim=-1
        )
        quality_per_sample = (
            quality_target * (quality_target.clamp_min(self.eps).log() - quality_log_probability)
        ).sum(dim=-1).clamp_min(0.0)
        quality_usable = valid_regions.any(dim=-1) & (target_has_variation | target_has_power)
        quality = _masked_mean(quality_per_sample, quality_usable)

        total = (
            self.mse_weight * mse
            + self.correlation_weight * correlation
            + self.spectral_weight * spectral
            + self.regional_weight * regional_loss
            + self.quality_weight * quality
        )
        return {
            "total": total,
            "mse": mse,
            "correlation": correlation,
            "spectral": spectral,
            "regional": regional_loss,
            "quality": quality,
        }
