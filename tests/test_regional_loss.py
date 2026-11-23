"""CPU checks for the pulse loss's masking and low-variance behavior."""

import unittest

import torch

from neural_methods.loss.RegionalPulseLoss import (
    RegionalPulseLoss,
    estimate_hr,
    spectral_distribution,
    waveform_correlation,
)


def make_outputs(waveform, regional=None, valid=None, quality_logits=None):
    if regional is None:
        regional = waveform.detach().unsqueeze(1).repeat(1, 3, 1).clone().requires_grad_(True)
    batch, regions, _ = regional.shape
    if valid is None:
        valid = torch.ones(batch, regions, dtype=torch.bool)
    if quality_logits is None:
        quality_logits = torch.zeros_like(regional, requires_grad=True)
    return {
        "waveform": waveform,
        "regional_waveforms": regional,
        "region_weights": torch.full_like(regional, 1.0 / regions),
        "quality_logits": quality_logits,
        "valid_regions": valid,
    }


class RegionalPulseLossTests(unittest.TestCase):
    def setUp(self):
        self.time = torch.arange(160, dtype=torch.float32) / 30.0
        self.target = torch.sin(2.0 * torch.pi * 1.2 * self.time).unsqueeze(0)

    def test_matching_signal_beats_wrong_frequency(self):
        criterion = RegionalPulseLoss()
        matched = make_outputs(self.target.clone().requires_grad_(True))
        wrong_signal = torch.sin(2.0 * torch.pi * 2.1 * self.time).unsqueeze(0)
        wrong = make_outputs(wrong_signal.clone().requires_grad_(True))
        matched_loss = criterion(matched, self.target)
        wrong_loss = criterion(wrong, self.target)
        self.assertLess(matched_loss["total"].item(), 1e-4)
        self.assertGreater(wrong_loss["correlation"].item(), 0.5)
        self.assertGreater(wrong_loss["spectral"].item(), 0.1)

    def test_constant_prediction_has_finite_nonzero_learning_gradient(self):
        outputs = make_outputs(torch.zeros_like(self.target, requires_grad=True))
        losses = RegionalPulseLoss()(outputs, self.target)
        self.assertEqual(set(losses), {"total", "mse", "correlation", "spectral", "regional", "quality"})
        for value in losses.values():
            self.assertEqual(value.ndim, 0)
            self.assertTrue(torch.isfinite(value))
        losses["total"].backward()
        for name in ("waveform", "regional_waveforms", "quality_logits"):
            self.assertTrue(torch.isfinite(outputs[name].grad).all(), name)
        self.assertGreater(outputs["waveform"].grad.abs().sum().item(), 0)
        self.assertGreater(outputs["regional_waveforms"].grad.abs().sum().item(), 0)
        self.assertLess(outputs["waveform"].grad.abs().max().item(), 100)

    def test_flat_target_retains_only_raw_mse(self):
        outputs = make_outputs(torch.ones_like(self.target, requires_grad=True))
        losses = RegionalPulseLoss()(outputs, torch.zeros_like(self.target))
        self.assertAlmostEqual(losses["mse"].item(), 1.0)
        self.assertAlmostEqual(losses["total"].item(), 0.1, places=6)
        for name in ("correlation", "spectral", "regional", "quality"):
            self.assertEqual(losses[name].item(), 0.0)
        losses["total"].backward()
        for name in ("waveform", "regional_waveforms", "quality_logits"):
            self.assertTrue(torch.isfinite(outputs[name].grad).all(), name)

    def test_quality_targets_are_detached_and_invalid_regions_are_excluded(self):
        wrong_signal = torch.sin(2.0 * torch.pi * 2.1 * self.time)
        candidates = torch.stack((self.target[0], wrong_signal, -self.target[0])).unsqueeze(0).requires_grad_(True)
        logits = torch.zeros_like(candidates, requires_grad=True)
        outputs = make_outputs(
            self.target.clone().requires_grad_(True),
            regional=candidates,
            valid=torch.tensor([[True, True, False]]),
            quality_logits=logits,
        )
        criterion = RegionalPulseLoss(
            mse_weight=0, correlation_weight=0, spectral_weight=0, regional_weight=0, quality_weight=1
        )
        losses = criterion(outputs, self.target)
        losses["total"].backward()
        self.assertEqual(candidates.grad.abs().sum().item(), 0.0)
        self.assertLess(logits.grad[0, 0].sum().item(), 0.0)
        self.assertGreater(logits.grad[0, 1].sum().item(), 0.0)
        self.assertEqual(logits.grad[0, 2].abs().sum().item(), 0.0)

    def test_all_invalid_regions_are_finite_and_receive_no_auxiliary_gradient(self):
        outputs = make_outputs(
            torch.zeros_like(self.target, requires_grad=True),
            valid=torch.zeros(1, 3, dtype=torch.bool),
        )
        losses = RegionalPulseLoss()(outputs, self.target)
        self.assertEqual(losses["regional"].item(), 0.0)
        self.assertEqual(losses["quality"].item(), 0.0)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["total"].backward()
        self.assertEqual(outputs["regional_waveforms"].grad.abs().sum().item(), 0.0)
        self.assertEqual(outputs["quality_logits"].grad.abs().sum().item(), 0.0)

    def test_spectral_helpers_and_flat_hr(self):
        distribution, frequencies = spectral_distribution(self.target)
        self.assertEqual(distribution.shape[-1], len(frequencies))
        self.assertTrue(torch.allclose(distribution.sum(dim=-1), torch.ones(1)))
        self.assertLess(abs(estimate_hr(self.target).item() - 72.0), 4.0)
        self.assertTrue(torch.isnan(estimate_hr(torch.zeros_like(self.target))).all())
        self.assertEqual(waveform_correlation(torch.zeros_like(self.target), self.target).item(), 0.0)

    def test_near_constant_inputs_are_finite(self):
        nearly_flat = (torch.ones_like(self.target) + 1e-7 * self.target).requires_grad_(True)
        outputs = make_outputs(nearly_flat)
        losses = RegionalPulseLoss()(outputs, self.target)
        losses["total"].backward()
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        self.assertTrue(torch.isfinite(nearly_flat.grad).all())


if __name__ == "__main__":
    unittest.main()
