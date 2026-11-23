"""Behavioral CPU checks without downloading a pretrained backbone."""

import copy
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from neural_methods.model.regional_dino import RegionalDINO


class _StubBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.linear = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, tokens):
        return tokens + self.dropout(F.gelu(self.linear(tokens)))


class _StubBackbone(nn.Module):
    """Small patch encoder with global mixing, for exclusion and gradient tests."""

    def __init__(self):
        super().__init__()
        self.embed_dim = 12
        self.patch_embed = nn.Conv2d(3, self.embed_dim, 1)
        self.blocks = nn.ModuleList([_StubBlock(self.embed_dim) for _ in range(3)])
        self.norm = nn.LayerNorm(self.embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.chunk_sizes = []

    def forward_features(self, frames):
        self.chunk_sizes.append(frames.shape[0])
        tokens = F.adaptive_avg_pool2d(self.patch_embed(frames), 3).flatten(2).transpose(1, 2)
        # Global mixing makes post-hoc token masking alone insufficient.
        tokens = tokens + tokens.mean(dim=1, keepdim=True)
        for block in self.blocks:
            tokens = block(tokens)
        return {"x_norm_patchtokens": self.norm(tokens)}


def _small_model(**overrides):
    settings = dict(use_dino=False, hidden_dim=8, context_dim=6,
                    num_tcn_blocks=2, dropout=0.1, encoder_size=28,
                    context_stride=4, frame_chunk_size=3)
    settings.update(overrides)
    return RegionalDINO(**settings)


class RegionalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(123)
        self.video = torch.rand(2, 13, 3, 17, 19) * 255.0

    def _assert_contract(self, outputs, batch, time):
        self.assertEqual(set(outputs), {"waveform", "regional_waveforms", "region_weights",
                                        "quality_logits", "valid_regions"})
        self.assertEqual(outputs["waveform"].shape, (batch, time))
        for key in ("regional_waveforms", "region_weights", "quality_logits"):
            self.assertEqual(outputs[key].shape, (batch, 9, time))
            self.assertTrue(torch.isfinite(outputs[key]).all(), key)
        self.assertEqual(outputs["valid_regions"].shape, (batch, 9))
        self.assertEqual(outputs["valid_regions"].dtype, torch.bool)
        torch.testing.assert_close(outputs["region_weights"].sum(dim=1),
                                   torch.ones(batch, time), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(outputs["waveform"],
                                   (outputs["region_weights"] * outputs["regional_waveforms"]).sum(dim=1),
                                   rtol=0, atol=0)
        torch.testing.assert_close(outputs["waveform"].mean(dim=1),
                                   torch.zeros(batch), rtol=0, atol=1e-6)

    def test_output_contract_and_exact_centered_fusion(self):
        model = _small_model().eval()
        nn.init.normal_(model.quality_head[-1].weight, std=0.3)
        with torch.no_grad():
            outputs = model(self.video, return_aux=True)
            plain = model(self.video)
        self._assert_contract(outputs, 2, 13)
        torch.testing.assert_close(plain, outputs["waveform"], rtol=0, atol=0)
        self.assertGreater(outputs["waveform"].std().item(), 0)

    def test_static_frames_have_no_temporal_template_in_eval_or_train(self):
        static = self.video[:, :1].expand(-1, 31, -1, -1, -1).clone()
        for use_dino in (False, True):
            model = _small_model(use_dino=use_dino,
                                 backbone=_StubBackbone() if use_dino else None)
            nn.init.normal_(model.quality_head[-1].weight, std=0.3)
            for mode in (False, True):
                with self.subTest(use_dino=use_dino, train=mode):
                    model.train(mode)
                    outputs = model(static, return_aux=True)
                    self.assertLess(outputs["waveform"].abs().max().item(), 2e-6)
                    self.assertLess(outputs["regional_waveforms"].abs().max().item(), 2e-6)

    def test_rejects_malformed_or_nonfinite_input(self):
        model = _small_model()
        for bad in (torch.zeros(2, 3, 16, 16), torch.zeros(2, 13, 6, 16, 16),
                    torch.zeros(0, 13, 3, 16, 16), torch.zeros(2, 0, 3, 16, 16),
                    torch.zeros(2, 13, 3, 2, 16), "video"):
            with self.subTest(shape=getattr(bad, "shape", None)), self.assertRaises(ValueError):
                model(bad)
        for nonfinite in (float("nan"), float("inf")):
            bad = self.video.clone()
            bad[0, 0, 0, 0, 0] = nonfinite
            with self.assertRaisesRegex(ValueError, "finite"):
                model(bad)

    def test_rejects_invalid_region_masks(self):
        model = _small_model()
        for mask in (torch.zeros(2, 9), torch.ones(2, 8, dtype=torch.bool),
                     torch.zeros(2, 9, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                model(self.video, region_mask=mask)

    def test_masked_pixels_cannot_change_fused_output_or_valid_gates(self):
        valid = torch.ones(2, 9, dtype=torch.bool)
        valid[:, 0] = False
        perturbed = self.video.clone()
        perturbed[:, :, :, :17 // 3, :19 // 3] = torch.rand(2, 13, 3, 17 // 3, 19 // 3) * 255.0
        for use_dino in (False, True):
            for fusion in ("learned", "uniform"):
                with self.subTest(use_dino=use_dino, fusion=fusion):
                    model = _small_model(use_dino=use_dino, fusion=fusion,
                                         backbone=_StubBackbone() if use_dino else None).eval()
                    # A live gate makes DINO-context leakage detectable.
                    nn.init.normal_(model.quality_head[-1].weight, std=0.4)
                    with torch.no_grad():
                        original = model(self.video, return_aux=True, region_mask=valid)
                        changed = model(perturbed, return_aux=True, region_mask=valid)
                    self._assert_contract(original, 2, 13)
                    for key in ("waveform", "regional_waveforms", "region_weights", "quality_logits"):
                        torch.testing.assert_close(original[key], changed[key], rtol=0, atol=0)
                    self.assertEqual(original["region_weights"][:, 0].abs().sum().item(), 0)
                    self.assertEqual(original["regional_waveforms"][:, 0].abs().sum().item(), 0)

    def test_frozen_backbone_stays_in_eval_mode(self):
        model = _small_model(use_dino=True, backbone=_StubBackbone())
        model.train()
        self.assertTrue(model.pulse_stem.training)
        self.assertFalse(model.backbone.training)
        self.assertTrue(all(not module.training for module in model.backbone.modules()))
        self.assertTrue(all(not value.requires_grad for value in model.backbone.parameters()))
        loss = model(self.video).square().mean()
        loss.backward()
        self.assertTrue(all(value.grad is None for value in model.backbone.parameters()))
        self.assertGreater(model.waveform_head.weight.grad.abs().sum().item(), 0)

    def test_only_selected_backbone_suffix_and_norm_receive_gradients(self):
        model = _small_model(use_dino=True, backbone=_StubBackbone(), finetune_blocks=2)
        nn.init.normal_(model.quality_head[-1].weight, std=0.4)
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.backbone.patch_embed.training)
        self.assertFalse(model.backbone.blocks[0].training)
        self.assertTrue(all(block.training for block in model.backbone.blocks[1:]))
        self.assertTrue(model.backbone.norm.training)
        loss = (model(self.video) - torch.randn(2, 13)).square().mean()
        loss.backward()
        for name, parameter in model.backbone.named_parameters():
            selected = name.startswith(("blocks.1.", "blocks.2.", "norm."))
            self.assertEqual(parameter.requires_grad, selected, name)
            if selected:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
            else:
                self.assertIsNone(parameter.grad, name)
        model.eval()
        self.assertTrue(all(not module.training for module in model.backbone.modules()))

    def test_sparse_context_chunks_and_variable_time_lengths(self):
        for time in (1, 2, 8, 13, 17):
            with self.subTest(time=time):
                model = _small_model(use_dino=True, backbone=_StubBackbone(), frame_chunk_size=3).eval()
                video = torch.rand(2, time, 3, 17, 19) * 255.0
                with torch.no_grad():
                    outputs = model(video, return_aux=True)
                self._assert_contract(outputs, 2, time)
                expected_samples = len(set(range(0, time, model.context_stride)) | {time - 1})
                self.assertEqual(sum(model.backbone.chunk_sizes), 2 * expected_samples)
                self.assertLessEqual(max(model.backbone.chunk_sizes), 3)

    def test_context_interpolation_uses_true_sample_timestamps(self):
        model = _small_model(use_dino=True, backbone=_StubBackbone()).eval()
        model.context_projection = nn.Identity()
        time = 11
        ramp = torch.arange(time).float().view(1, time, 1, 1, 1).expand(1, time, 3, 18, 18)

        def encode(chunk):
            return chunk.mean(dim=(1, 2, 3))[:, None, None].expand(-1, 9, 1)

        model._encode_context_chunk = encode
        with torch.no_grad():
            context = model._context_features(ramp, torch.ones(1, 9, dtype=torch.bool))
        torch.testing.assert_close(context[0, :, 0], torch.arange(time).float().expand(9, -1),
                                   rtol=1e-6, atol=1e-6)

    def test_chunking_does_not_change_frozen_context_results(self):
        model = _small_model(use_dino=True, backbone=_StubBackbone(), frame_chunk_size=1).eval()
        other = copy.deepcopy(model)
        other.frame_chunk_size = 64
        with torch.no_grad():
            left = model(self.video, return_aux=True)
            right = other(self.video, return_aux=True)
        for key in ("waveform", "region_weights", "quality_logits"):
            torch.testing.assert_close(left[key], right[key], rtol=1e-5, atol=1e-6)

    def test_context_cannot_change_regional_waveforms(self):
        model = _small_model(use_dino=True, backbone=_StubBackbone()).eval()
        nn.init.normal_(model.quality_head[-1].weight, std=0.4)
        with torch.no_grad():
            original = model(self.video, return_aux=True)
            for parameter in model.backbone.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.1)
            changed = model(self.video, return_aux=True)
        # The common DC fusion correction is the only gate-dependent shift.
        before = original["regional_waveforms"] - original["regional_waveforms"].mean(dim=-1, keepdim=True)
        after = changed["regional_waveforms"] - changed["regional_waveforms"].mean(dim=-1, keepdim=True)
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
        self.assertGreater((original["region_weights"] - changed["region_weights"]).abs().sum().item(), 0)

    def test_no_dino_and_uniform_ablation_never_download(self):
        with mock.patch("torch.hub.load", side_effect=AssertionError("unexpected download")) as load:
            for kwargs in (dict(use_dino=False), dict(use_dino=True, fusion="uniform"),
                           dict(use_dino=True, backbone=_StubBackbone())):
                model = _small_model(**kwargs).eval()
                with torch.no_grad():
                    self.assertEqual(model(self.video).shape, (2, 13))
            load.assert_not_called()

    def test_actual_clip_dimensions_smoke(self):
        model = RegionalDINO(use_dino=False).eval()
        with torch.no_grad():
            outputs = model(torch.rand(2, 160, 3, 128, 128) * 255.0, return_aux=True)
        self._assert_contract(outputs, 2, 160)


if __name__ == "__main__":
    unittest.main()
