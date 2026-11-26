"""Joint video/PPG slowing without padding, wrapping, or changing FPS metadata.

For output sample i, read both inputs at start + speed*i. Speeds <=1 use a
shorter interval of the same cached clip, so the output rhythm is approximately
speed*source_frequency. This also slows motion and smooths via interpolation;
it is not a pure physiological intervention or bidirectional rate invariance.
"""

import math

import torch


def paired_resample(video, target, speed, start=0.0):
    """Resample [T,3,H,W] raw RGB and [T] PPG at identical coordinates."""
    if video.ndim != 4 or video.shape[1] != 3 or target.shape != video.shape[:1]:
        raise ValueError('Require video [T,3,H,W] and target [T].')
    if not video.is_floating_point() or not target.is_floating_point():
        raise ValueError('Require floating inputs; RGB must retain raw [0,255] units.')
    if video.device != target.device or len(target) < 3:
        raise ValueError('Require at least three samples on the same device.')
    length = len(target)
    if not (math.isfinite(speed) and 0 < speed <= 1 and math.isfinite(start)
            and start >= 0 and start + speed * (length - 1) <= length - 1 + 1e-7):
        raise ValueError('Coordinates must stay inside the original clip; no extrapolation.')
    if speed == 1 and start == 0:
        return video, target
    position = (start + speed * torch.arange(length, device=video.device,
                                             dtype=torch.float64)).clamp_max(length - 1)
    left = position.floor().long()
    right = (left + 1).clamp_max(length - 1)
    fraction = (position - left).to(video.dtype)
    result = torch.empty_like(video)
    # Bound temporary image memory instead of gathering a whole batch twice.
    for offset in range(0, length, 32):
        sl = slice(offset, offset + 32)
        result[sl] = torch.lerp(video[left[sl]], video[right[sl]],
                                fraction[sl, None, None, None])
    label = torch.lerp(target[left], target[right], fraction.to(target.dtype))
    return result, label


def reference_rate(target, fs, min_hz=.75, max_hz=2.5):
    """Conservative short-clip eligibility proxy, not certified physiological HR.

    Require an in-band spectral peak and an autocorrelation local maximum to
    agree within max(10 bpm, 12%). Zero-padding does not remove the limited
    information in a 160-frame clip. Flat/ambiguous examples remain unaugmented.
    """
    values = target.detach().to(device='cpu', dtype=torch.float64)
    if values.ndim != 1 or len(values) < 30 or not bool(torch.isfinite(values).all()):
        return None, 'invalid_reference'
    values = values - values.mean()
    energy = float(values.square().mean())
    if energy <= 1e-6:
        return None, 'flat_reference'
    taper = torch.hann_window(len(values), periodic=False, dtype=values.dtype)
    nfft = max(2048, 1 << (len(values) - 1).bit_length())
    power = torch.fft.rfft(values * taper, n=nfft).abs().square()
    frequency = torch.fft.rfftfreq(nfft, d=1 / fs, dtype=values.dtype)
    indices = torch.where((frequency >= min_hz) & (frequency <= max_hz))[0]
    index = indices[power[indices].argmax()]
    # A maximum at a search boundary has uncertain support outside that band.
    if int(index) in (int(indices[0]), int(indices[-1])):
        return None, 'boundary_peak'
    spectral_hr = float(frequency[index] * 60)
    n = len(values)
    correlations = torch.fft.irfft(torch.fft.rfft(values, n=2 * n).abs().square(), n=2 * n)[:n]
    lower = max(1, int(math.ceil(fs / max_hz)))
    upper = min(n - 2, int(math.floor(fs / min_hz)))
    candidates = [i for i in range(lower, upper + 1)
                  if correlations[i] > correlations[i - 1] and correlations[i] >= correlations[i + 1]]
    if not candidates:
        return None, 'no_autocorrelation_peak'
    best = max(candidates, key=lambda i: float(correlations[i]))
    autocorr_hr = 60 * fs / best
    if abs(spectral_hr - autocorr_hr) > max(10., .12 * spectral_hr):
        return None, 'reference_disagreement'
    return spectral_hr, None


class PairedRateAugment:
    """Independent RNG; same batch size/update count; validation bypasses this."""

    def __init__(self, fs, probability=.5, min_speed=.6, max_speed=1.,
                 min_output_bpm=50., max_output_bpm=140., seed=1701):
        if not (0 <= probability <= 1 and 0 < min_speed < max_speed <= 1
                and 0 < min_output_bpm < max_output_bpm < fs * 30):
            raise ValueError('Invalid paired slowing settings.')
        self.fs = float(fs)
        self.probability = float(probability)
        self.min_speed, self.max_speed = float(min_speed), float(max_speed)
        self.min_output_bpm, self.max_output_bpm = float(min_output_bpm), float(max_output_bpm)
        self.generator = torch.Generator(device='cpu').manual_seed(seed)

    def state_dict(self):
        return {'generator_state': self.generator.get_state()}

    def load_state_dict(self, state):
        self.generator.set_state(state['generator_state'].cpu())

    def __call__(self, video, target):
        if video.ndim != 5 or video.shape[2] != 3 or target.shape != video.shape[:2]:
            raise ValueError('Require video [B,T,3,H,W] and target [B,T].')
        # No random draw or tensor operation when disabled.
        if self.probability == 0:
            return video, target, [{'applied': False, 'reason': 'probability'} for _ in target]
        plans = []
        for label in target:
            if float(torch.rand((), generator=self.generator)) >= self.probability:
                plans.append({'applied': False, 'reason': 'probability'})
                continue
            hr, reason = reference_rate(label, self.fs)
            if hr is None:
                plans.append({'applied': False, 'reason': reason})
                continue
            low = max(self.min_speed, self.min_output_bpm / hr)
            high = min(self.max_speed, self.max_output_bpm / hr)
            if high - low < 1e-4:
                plans.append({'applied': False, 'reason': 'no_admissible_speed', 'source_hr_bpm': hr})
                continue
            speed = low + float(torch.rand((), generator=self.generator)) * (high - low)
            start = float(torch.rand((), generator=self.generator)) * (len(label) - 1) * (1 - speed)
            plans.append({'applied': True, 'reason': 'applied', 'source_hr_bpm': hr,
                          'expected_output_hr_bpm': hr * speed, 'speed': speed, 'start': start})
        if not any(p['applied'] for p in plans):
            return video, target, plans
        result, labels = video.clone(), target.clone()
        for i, plan in enumerate(plans):
            if plan['applied']:
                result[i], labels[i] = paired_resample(video[i], target[i], plan['speed'], plan['start'])
        return result, labels, plans
