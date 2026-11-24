"""Reference-aware pulse diagnostics, separate from the published evaluator.

GREEN/POS/CHROM extraction uses RGB alone. Reference PPG enters only the
diagnostic comparisons; harmonic bands and lag searches are not inference rules.
The common diagnostic pipeline uses the legacy smoothness-prior detrend
(lambda=100), a first-order zero-phase Butterworth bandpass and Welch PSD with
nfft=3333 (at 30 Hz, 0.540 bpm bins). The detrend uses a sparse solve rather
than explicitly inverting a dense matrix.
"""

import math

import numpy as np
from scipy import signal as scipy_signal
from scipy import sparse
from scipy.sparse.linalg import spsolve


_EPS = 1e-12


def _validate_band(fs, min_hz, max_hz):
    if not (math.isfinite(fs) and fs > 0 and 0 < min_hz < max_hz < fs / 2):
        raise ValueError("Require finite fs and 0 < min_hz < max_hz < fs/2.")


def _vector(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1:
        raise ValueError("Expected a one-dimensional waveform.")
    if not np.isfinite(value).all():
        raise ValueError("Waveform contains nonfinite values.")
    return value


def _safe_filter(values, fs, min_hz, max_hz, order=1):
    b, a = scipy_signal.butter(order, [min_hz, max_hz], btype="bandpass", fs=fs)
    if len(values) < 2:
        return np.zeros_like(values)
    return scipy_signal.filtfilt(b, a, values, padlen=min(3 * max(len(a), len(b)), len(values) - 1))


def detrend_signal(signal, smoothness=100.0):
    """Legacy Tarvainen smoothness-prior detrend using an efficient sparse solve.

    Numerically implements ``x - solve(I + smoothness**2 * D.T @ D, x)``
    with D the second-difference matrix. No dense N-by-N inverse is built.
    """
    values = _vector(signal)
    if len(values) < 3:
        raise ValueError("Detrending requires at least three samples.")
    if not math.isfinite(smoothness) or smoothness <= 0:
        raise ValueError("smoothness must be finite and positive.")
    differences = sparse.diags(
        (np.ones(len(values) - 2), -2 * np.ones(len(values) - 2), np.ones(len(values) - 2)),
        (0, 1, 2), shape=(len(values) - 2, len(values)), format="csc",
    )
    system = sparse.eye(len(values), format="csc") + smoothness ** 2 * (differences.T @ differences)
    return values - spsolve(system.tocsc(), values)


def preprocess_signal(signal, fs=30, min_hz=0.75, max_hz=2.5):
    """Return a same-length diagnostic waveform, without amplitude z-scoring.

    Reject malformed/nonfinite/short input with ValueError. analyze_signal
    converts such failures into explicit JSON-safe unusable records.
    """
    _validate_band(fs, min_hz, max_hz)
    values = _vector(signal)
    if len(values) < max(12, int(math.ceil(fs))):
        raise ValueError("Waveform is too short: diagnostics require at least one second and 12 samples.")
    if float(np.std(values)) <= _EPS:
        return np.zeros_like(values)
    centered = detrend_signal(values)
    return _safe_filter(centered, fs, min_hz, max_hz)


def _welch(values, fs):
    nperseg = min(256, len(values))
    return scipy_signal.welch(
        values, fs=fs, window="hann", nperseg=nperseg,
        noverlap=nperseg // 2, nfft=max(3333, nperseg), detrend=False,
        scaling="density",
    )


def spectrum(signal, fs=30, min_hz=0.75, max_hz=2.5, bandpass=True):
    """Return all Welch ``(frequency_hz, power_density)`` bins.

    The returned spectrum covers zero through Nyquist. The peak HR search uses
    only min_hz..max_hz. Set bandpass=False for the un-bandpassed, detrended
    spectrum used to inspect second harmonics above the HR-search band.
    """
    filtered = preprocess_signal(signal, fs, min_hz, max_hz)
    return _welch(filtered if bandpass else detrend_signal(signal), fs)


def classical_signals(rgb_trace, fs=30):
    """Extract GREEN, POS, and CHROM from an RGB trace [T,3], without GT.

    Inputs are spatial RGB means. POS uses G-B and G+B-2R after windowwise
    RGB-mean normalization. CHROM uses 3R-2G and 1.5R+G-1.5B, third-order
    zero-phase filtering in 0.7..2.5 Hz before alpha/std combination. Both use
    1.6-second windows, Hann 50%-overlap-add and edge padding. Outputs preserve
    the input length, including short clips; subsequent analyze_signal can mark
    short clips unusable. This implementation uses no deprecated np.mat/hanning.
    """
    _validate_band(fs, 0.7, 2.5)
    rgb = np.asarray(rgb_trace, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("rgb_trace must have shape [T,3].")
    if not np.isfinite(rgb).all():
        raise ValueError("RGB trace contains nonfinite values.")
    length = len(rgb)
    result = {"GREEN": rgb[:, 1].copy(), "POS": np.zeros(length), "CHROM": np.zeros(length)}
    if length == 0:
        return result
    window_length = max(4, int(math.ceil(1.6 * fs)))
    window_length += window_length % 2
    hop = window_length // 2
    padded = np.pad(rgb, ((hop, hop), (0, 0)), mode="edge")
    # Complete the final half-window instead of truncating the last samples.
    remainder = (len(padded) - window_length) % hop
    if remainder:
        padded = np.pad(padded, ((0, hop - remainder), (0, 0)), mode="edge")
    taper = scipy_signal.windows.hann(window_length, sym=False)
    accumulator = {"POS": np.zeros(len(padded)), "CHROM": np.zeros(len(padded))}
    weights = np.zeros(len(padded))
    for start in range(0, len(padded) - window_length + 1, hop):
        block = padded[start:start + window_length]
        means = block.mean(axis=0)
        denominators = np.where(np.abs(means) > _EPS, means, np.where(means < 0, -_EPS, _EPS))
        normalized = block / denominators
        red, green, blue = normalized.T
        first, second = green - blue, green + blue - 2.0 * red
        alpha = np.std(first) / max(float(np.std(second)), _EPS)
        pos = first + alpha * second
        pos -= pos.mean()
        x = _safe_filter(3.0 * red - 2.0 * green, fs, 0.7, 2.5, order=3)
        y = _safe_filter(1.5 * red + green - 1.5 * blue, fs, 0.7, 2.5, order=3)
        chrom = x - np.std(x) / max(float(np.std(y)), _EPS) * y
        chrom -= chrom.mean()
        for name, value in (("POS", pos), ("CHROM", chrom)):
            accumulator[name][start:start + window_length] += taper * value
        weights[start:start + window_length] += taper
    for name in accumulator:
        result[name] = (accumulator[name] / np.maximum(weights, _EPS))[hop:hop + length]
    return result


def _correlation(first, second):
    first = first - first.mean()
    second = second - second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= _EPS:
        return None
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def _dominant_peak(frequencies, power, min_hz, max_hz):
    band = (frequencies >= min_hz) & (frequencies <= max_hz)
    if not band.any() or float(power[band].sum()) <= _EPS:
        return None
    indices = np.flatnonzero(band)
    return float(frequencies[indices[np.argmax(power[band])]])


def _peaks(frequencies, power, min_hz, max_hz):
    band = (frequencies >= min_hz) & (frequencies <= max_hz)
    indices = np.flatnonzero(band)
    if not len(indices):
        return []
    local, _ = scipy_signal.find_peaks(power[indices])
    # A maximum at the search boundary is still worth showing explicitly.
    maximum = int(np.argmax(power[indices]))
    candidates = np.unique(np.append(local, maximum))
    candidates = sorted(candidates, key=lambda index: power[indices[index]], reverse=True)[:3]
    total = max(float(power[indices].sum()), _EPS)
    return [{"frequency_hz": float(frequencies[indices[index]]),
             "hr_bpm": float(frequencies[indices[index]] * 60),
             "relative_bin_power": float(power[indices[index]] / total)}
            for index in candidates]


def _autocorrelation_hr(values, fs, min_hz, max_hz):
    values = values - values.mean()
    if float(np.dot(values, values)) <= _EPS:
        return None
    autocorrelation = scipy_signal.correlate(values, values, mode="full", method="fft")[len(values) - 1:]
    lower, upper = max(1, int(math.ceil(fs / max_hz))), min(len(values) - 2, int(math.floor(fs / min_hz)))
    if upper < lower:
        return None
    candidates, _ = scipy_signal.find_peaks(autocorrelation)
    candidates = candidates[(candidates >= lower) & (candidates <= upper)]
    if not len(candidates):
        return None
    best = int(candidates[np.argmax(autocorrelation[candidates])])
    return float(60.0 * fs / best)


def _power_sum(frequencies, power, mask):
    df = frequencies[1] - frequencies[0]
    return float(power[mask].sum() * df)


def analyze_signal(signal, reference=None, fs=30, min_hz=0.75, max_hz=2.5,
                   max_lag_seconds=1.0):
    """Return JSON-safe pulse diagnostics, including explicit unusable reasons.

    Positive best_lag_seconds means prediction is delayed: prediction[lag:]
    aligns with reference[:-lag]. Lagged correlation maximizes absolute
    correlation and therefore tolerates sign inversion; its signed value is
    also returned. Reference spectral bands use +/-0.1 Hz around its dominant
    peak and twice that peak. Harmonic bands intentionally extend beyond the
    fundamental HR-search band and are clipped at Nyquist. Their powers are
    measured on smoothness-prior detrended, un-bandpassed prediction, to avoid erasing
    a second harmonic above max_hz. SNR uses 10*log10(power ratio).

    Spectral agreement, harmonic indicators and best lag are diagnostic only.
    A standardized reference does not establish sensor clipping or true HR;
    reference_flat flags only observed variance, not physiological validity.
    """
    result = {
        "status": "unusable", "reason": None, "sample_count": None,
        "hr_bpm": None, "prediction_std": None, "waveform_corr": None,
        "lagged_abs_corr": None, "lagged_signed_corr": None,
        "best_lag_seconds": None, "best_lag_samples": None,
        "reference_status": "not_provided" if reference is None else "unusable",
        "reference_reason": None, "reference_std": None, "reference_flat": None,
        "reference_peak_hz": None, "reference_hr_bpm": None,
        "reference_autocorr_hr_bpm": None, "hr_abs_error_bpm": None,
        "near_double_reference": None, "near_half_reference": None,
        "fundamental_power": None, "second_harmonic_power": None,
        "second_to_fundamental_ratio": None, "fundamental_fraction": None,
        "harmonics_fraction": None, "snr_fundamental_db": None,
        "snr_harmonics_db": None, "dominant_peaks": [],
        "fundamental_band_hz": None, "second_harmonic_band_hz": None,
        "harmonic_spectrum": "tarvainen_lambda100_without_bandpass",
        "lag_sign": "positive_prediction_delayed",
    }
    try:
        if not math.isfinite(max_lag_seconds) or max_lag_seconds < 0:
            raise ValueError("max_lag_seconds must be finite and nonnegative.")
        values = _vector(signal)
        result["sample_count"] = len(values)
        result["prediction_std"] = float(np.std(values)) if len(values) else None
        filtered = preprocess_signal(values, fs, min_hz, max_hz)
        if result["prediction_std"] <= _EPS or float(np.std(filtered)) <= _EPS:
            result["reason"] = "flat_or_no_bandpassed_variation"
            return result
        frequencies, power = _welch(filtered, fs)
        peak = _dominant_peak(frequencies, power, min_hz, max_hz)
        if peak is None:
            result["reason"] = "no_in_band_power"
            return result
        result.update(status="ok", hr_bpm=float(peak * 60.0),
                      dominant_peaks=_peaks(frequencies, power, min_hz, max_hz))
    except (TypeError, ValueError, OverflowError) as error:
        result["reason"] = str(error)
        return result
    if reference is None:
        return result
    try:
        reference = _vector(reference)
        if len(reference) != len(values):
            raise ValueError("Prediction and reference lengths differ; no implicit trimming.")
        result["reference_std"] = float(np.std(reference))
        result["reference_flat"] = result["reference_std"] <= _EPS
        reference_filtered = preprocess_signal(reference, fs, min_hz, max_hz)
        if result["reference_flat"] or float(np.std(reference_filtered)) <= _EPS:
            raise ValueError("Reference is flat or has no bandpassed variation.")
        ref_frequencies, ref_power = _welch(reference_filtered, fs)
        reference_peak = _dominant_peak(ref_frequencies, ref_power, min_hz, max_hz)
        if reference_peak is None:
            raise ValueError("Reference has no in-band power.")
        result.update(reference_status="ok", reference_peak_hz=reference_peak,
                      reference_hr_bpm=float(reference_peak * 60),
                      reference_autocorr_hr_bpm=_autocorrelation_hr(reference_filtered, fs, min_hz, max_hz),
                      waveform_corr=_correlation(filtered, reference_filtered))
        result["hr_abs_error_bpm"] = abs(result["hr_bpm"] - result["reference_hr_bpm"])
        result["near_double_reference"] = abs(result["hr_bpm"] - 2 * result["reference_hr_bpm"]) <= 5.0
        result["near_half_reference"] = abs(result["hr_bpm"] - 0.5 * result["reference_hr_bpm"]) <= 5.0
        max_lag = min(int(round(max_lag_seconds * fs)), len(values) - max(12, int(math.ceil(fs))))
        comparisons = []
        for lag in range(-max_lag, max_lag + 1):
            if lag > 0:
                corr = _correlation(filtered[lag:], reference_filtered[:-lag])
            elif lag < 0:
                corr = _correlation(filtered[:lag], reference_filtered[-lag:])
            else:
                corr = result["waveform_corr"]
            if corr is not None:
                comparisons.append((abs(corr), -abs(lag), lag, corr))
        if comparisons:
            _, _, best_lag, signed = max(comparisons)
            result.update(lagged_abs_corr=abs(signed), lagged_signed_corr=signed,
                          best_lag_samples=int(best_lag), best_lag_seconds=float(best_lag / fs))

        # Use the broader un-bandpassed spectrum for harmonic accounting. The
        # fundamental HR search remains on the common bandpassed Welch spectrum.
        harmonic_frequencies, harmonic_power = _welch(detrend_signal(values), fs)
        width = 0.1
        first_band = (max(0.0, reference_peak - width), min(fs / 2, reference_peak + width))
        second_band = (max(0.0, 2 * reference_peak - width), min(fs / 2, 2 * reference_peak + width))
        if second_band[0] > second_band[1]:
            second_band = None  # This harmonic is unobservable above Nyquist.
        first_mask = (harmonic_frequencies >= first_band[0]) & (harmonic_frequencies <= first_band[1])
        second_mask = (np.zeros_like(harmonic_frequencies, dtype=bool) if second_band is None else
                       (harmonic_frequencies >= second_band[0]) & (harmonic_frequencies <= second_band[1]))
        evaluation_mask = ((harmonic_frequencies >= min_hz) & (harmonic_frequencies <= max_hz)) | first_mask | second_mask
        first_power = _power_sum(harmonic_frequencies, harmonic_power, first_mask)
        second_power = _power_sum(harmonic_frequencies, harmonic_power, second_mask)
        total_power = _power_sum(harmonic_frequencies, harmonic_power, evaluation_mask)
        harmonic_total = _power_sum(harmonic_frequencies, harmonic_power, first_mask | second_mask)
        first_noise = max(total_power - first_power, 0.0)
        harmonic_noise = max(total_power - harmonic_total, 0.0)
        result.update(
            fundamental_power=first_power, second_harmonic_power=second_power,
            second_to_fundamental_ratio=second_power / max(first_power, _EPS),
            fundamental_fraction=first_power / max(total_power, _EPS),
            harmonics_fraction=harmonic_total / max(total_power, _EPS),
            snr_fundamental_db=float(10 * np.log10(max(first_power, _EPS) / max(first_noise, _EPS))),
            snr_harmonics_db=float(10 * np.log10(max(harmonic_total, _EPS) / max(harmonic_noise, _EPS))),
            fundamental_band_hz=list(first_band),
            second_harmonic_band_hz=list(second_band) if second_band is not None else None,
        )
    except (TypeError, ValueError, OverflowError) as error:
        result["reference_reason"] = str(error)
    return result
