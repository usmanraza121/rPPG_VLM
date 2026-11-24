"""Synthetic numerical checks; these do not establish real-data accuracy."""

import json
import unittest

import numpy as np

from tools.regional_signal_diagnostics import (
    analyze_signal, classical_signals, detrend_signal, preprocess_signal, spectrum,
)


class RegionalSignalDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.fs = 30
        self.time = np.arange(900) / self.fs
        self.reference = np.sin(2 * np.pi * 1.2 * self.time)

    def test_known_frequency_and_signed_correlation(self):
        result = analyze_signal(-self.reference, self.reference)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["reference_status"], "ok")
        self.assertLess(abs(result["hr_bpm"] - 72), 0.7)
        self.assertLess(result["waveform_corr"], -0.99)
        self.assertGreater(result["lagged_abs_corr"], 0.99)
        self.assertEqual(result["best_lag_samples"], 0)
        self.assertFalse(result["near_double_reference"])
        self.assertLess(abs(result["reference_autocorr_hr_bpm"] - 72), 3)
        json.dumps(result, allow_nan=False)

    def test_positive_lag_means_delayed_prediction(self):
        # Two frequencies break the half-cycle ambiguity of a pure sinusoid.
        reference = self.reference + 0.6 * np.sin(2 * np.pi * 1.83 * self.time + 0.3)
        delayed = np.concatenate((np.repeat(reference[0], 6), reference[:-6]))
        result = analyze_signal(delayed, reference, max_lag_seconds=0.3)
        self.assertEqual(result["best_lag_samples"], 6)
        self.assertAlmostEqual(result["best_lag_seconds"], 0.2)
        self.assertGreater(result["lagged_signed_corr"], 0.97)

    def test_second_harmonic_does_not_establish_correct_hr(self):
        reference = np.sin(2 * np.pi * 0.8 * self.time)
        prediction = 0.03 * reference + np.sin(2 * np.pi * 1.6 * self.time)
        result = analyze_signal(prediction, reference)
        self.assertTrue(result["near_double_reference"])
        self.assertGreater(result["hr_abs_error_bpm"], 40)
        self.assertGreater(result["second_to_fundamental_ratio"], 100)
        self.assertGreater(result["snr_harmonics_db"], result["snr_fundamental_db"] + 10)

    def test_second_harmonic_above_hr_band_is_reported(self):
        prediction = self.reference + 2 * np.sin(2 * np.pi * 2.4 * self.time)
        reference = np.sin(2 * np.pi * 1.2 * self.time)
        result = analyze_signal(prediction, reference, max_hz=2.0)
        self.assertGreater(result["second_harmonic_band_hz"][0], 2.0)
        self.assertGreater(result["second_harmonic_power"], result["fundamental_power"])

    def test_harmonic_above_nyquist_is_explicitly_unobservable(self):
        time = np.arange(180) / 6.0
        values = np.sin(2 * np.pi * 1.8 * time)
        result = analyze_signal(values, values, fs=6)
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["second_harmonic_band_hz"])
        self.assertEqual(result["second_harmonic_power"], 0.0)
        json.dumps(result, allow_nan=False)

    def test_invalid_flat_short_inputs_are_json_safe(self):
        for values in (np.ones(160), np.arange(8), np.array([1., np.nan] * 80), np.zeros(0)):
            result = analyze_signal(values, self.reference)
            self.assertEqual(result["status"], "unusable")
            self.assertIsNone(result["hr_bpm"])
            self.assertIsNotNone(result["reason"])
            json.dumps(result, allow_nan=False)
        result = analyze_signal(self.reference, np.ones_like(self.reference))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["reference_status"], "unusable")
        self.assertTrue(result["reference_flat"])
        self.assertIsNone(result["waveform_corr"])
        json.dumps(result, allow_nan=False)

    def test_no_implicit_length_trimming(self):
        result = analyze_signal(self.reference, self.reference[:-1])
        self.assertEqual(result["reference_status"], "unusable")
        self.assertIn("lengths differ", result["reference_reason"])

    def test_classical_output_lengths_and_green_identity(self):
        for length in (0, 1, 13, 47, 48, 49, 160, 901):
            t = np.arange(length) / self.fs
            pulse = np.sin(2 * np.pi * 1.2 * t)
            rgb = np.array([100., 130., 90.])[None] + pulse[:, None] * np.array([0.3, 1.0, 0.1])[None]
            result = classical_signals(rgb)
            self.assertEqual(set(result), {"GREEN", "POS", "CHROM"})
            for value in result.values():
                self.assertEqual(value.shape, (length,))
                self.assertTrue(np.isfinite(value).all())
            np.testing.assert_array_equal(result["GREEN"], rgb[:, 1])

    def test_classical_known_pulse_and_constant_trace(self):
        rgb = np.array([100., 130., 90.])[None] + self.reference[:, None] * np.array([0.3, 1.0, 0.1])[None]
        for values in classical_signals(rgb).values():
            result = analyze_signal(values, self.reference)
            self.assertLess(abs(result["hr_bpm"] - 72), 1.0)
        for values in classical_signals(np.ones((160, 3)) * 100).values():
            self.assertLess(float(np.std(values)), 1e-8)

    def test_preprocess_and_spectrum_preserve_protocol(self):
        processed = preprocess_signal(self.reference)
        frequencies, power = spectrum(self.reference)
        self.assertEqual(len(processed), len(self.reference))
        self.assertEqual(len(frequencies), len(power))
        self.assertAlmostEqual(frequencies[1] - frequencies[0], 30 / 3333)
        self.assertTrue(np.isfinite(power).all())
        self.assertTrue((power >= 0).all())

    def test_sparse_detrend_matches_legacy_dense_inverse(self):
        rng = np.random.default_rng(4)
        values = rng.normal(size=80) + np.linspace(-3, 5, 80)
        differences = np.diff(np.eye(len(values)), n=2, axis=0)
        identity = np.eye(len(values))
        legacy = (identity - np.linalg.inv(identity + 100 ** 2 * differences.T @ differences)) @ values
        np.testing.assert_allclose(detrend_signal(values), legacy, rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
