"""Alignment, frequency, RNG and trainer controls for paired slowing."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import get_config
from neural_methods.augmentation.paired_rate import (
    PairedRateAugment, paired_resample, reference_rate,
)
from neural_methods.trainer.RegionalDINOTrainer import RegionalDINOTrainer


ROOT = Path(__file__).resolve().parents[1]


class PairedRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def signals(self, length=160, hz=1.5):
        t = torch.arange(length) / 30
        target = torch.sin(2 * torch.pi * hz * t)
        video = (120 + 3 * target[:, None, None, None]).expand(-1, 3, 9, 9).clone()
        return video, target

    def test_joint_coordinates_preserve_alignment_and_slow_frequency(self):
        video, target = self.signals()
        altered, label = paired_resample(video, target, .7, 10)
        self.assertEqual(altered.shape, video.shape)
        torch.testing.assert_close((altered[:, 0, 0, 0] - 120) / 3, label,
                                   atol=3e-6, rtol=1e-5)
        positions = 10 + .7 * torch.arange(160)
        analytic = torch.sin(2 * torch.pi * 1.5 * positions / 30)
        self.assertLess(float((label - analytic).abs().max()), .013)
        hr, reason = reference_rate(label, 30)
        self.assertIsNone(reason)
        self.assertAlmostEqual(hr, 63, delta=2)
        self.assertGreaterEqual(float(altered.min()), float(video.min()) - 1e-6)
        self.assertLessEqual(float(altered.max()), float(video.max()) + 1e-6)

    def test_identity_disabled_and_random_streams_are_unchanged(self):
        video, target = self.signals()
        a, b = paired_resample(video, target, 1)
        self.assertIs(a, video)
        self.assertIs(b, target)
        aug = PairedRateAugment(30, probability=0)
        state = aug.generator.get_state().clone()
        batch_video, batch_target = video[None], target[None]
        x, y, _ = aug(batch_video, batch_target)
        self.assertIs(x, batch_video)
        self.assertIs(y, batch_target)
        torch.testing.assert_close(x, video[None], rtol=0, atol=0)
        torch.testing.assert_close(y, target[None], rtol=0, atol=0)
        self.assertTrue(torch.equal(state, aug.generator.get_state()))
        aug = PairedRateAugment(30, probability=1)
        state = torch.get_rng_state().clone()
        aug(video[None], target[None])
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_reproducible_plans_state_restore_and_no_input_mutation(self):
        video, target = self.signals()
        original = video.clone()
        a = PairedRateAugment(30, probability=1, seed=1701)
        b = PairedRateAugment(30, probability=1, seed=1701)
        av, at, ap = a(video[None], target[None])
        bv, bt, bp = b(video[None], target[None])
        self.assertEqual(ap, bp)
        self.assertTrue(ap[0]['applied'])
        self.assertGreaterEqual(ap[0]['expected_output_hr_bpm'], 50)
        self.assertLessEqual(ap[0]['expected_output_hr_bpm'], 140)
        torch.testing.assert_close(av, bv)
        torch.testing.assert_close(at, bt)
        torch.testing.assert_close(video, original)
        checkpoint = a.state_dict()
        expected = a(video[None], target[None])[2]
        b.load_state_dict(checkpoint)
        self.assertEqual(expected, b(video[None], target[None])[2])

    def test_flat_and_invalid_clips_are_not_fabricated(self):
        video, target = self.signals()
        aug = PairedRateAugment(30, probability=1)
        result, labels, plans = aug(video[None], torch.zeros_like(target)[None])
        self.assertEqual(plans[0]['reason'], 'flat_reference')
        self.assertFalse(plans[0]['applied'])
        self.assertTrue(torch.isfinite(labels).all())
        with self.assertRaises(ValueError):
            paired_resample(video, target, 1.1)
        with self.assertRaises(ValueError):
            paired_resample(video, target, .8, 100)

    def test_training_integration_validation_bypass_and_checkpoint_provenance(self):
        outer = self

        class Tiny(Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, i):
                video, target = outer.signals(hz=1.5 + .1 * i)
                return video, target, f'subject{i + 1}_recording', 0

        loader = DataLoader(Tiny(), batch_size=2, shuffle=False)
        with tempfile.TemporaryDirectory() as temporary:
            with contextlib.redirect_stdout(io.StringIO()):
                cfg = get_config(argparse.Namespace(config_file=str(
                    ROOT / 'configs/train_configs/ablations/regional_shallow_rate_aug.yaml')))
            cfg.defrost()
            cfg.DEVICE = 'cpu'
            cfg.TRAIN.EPOCHS = 2
            cfg.TRAIN.REGIONAL_DINO.CHECKPOINT_START_EPOCH = 0
            cfg.TRAIN.REGIONAL_DINO.RATE_AUG.PROBABILITY = 1.
            cfg.MODEL.REGIONAL_DINO.HIDDEN_DIM = 8
            cfg.MODEL.REGIONAL_DINO.CONTEXT_DIM = 4
            cfg.MODEL.REGIONAL_DINO.NUM_TCN_BLOCKS = 1
            cfg.MODEL.MODEL_DIR = temporary + '/checkpoints'
            cfg.LOG.PATH = temporary + '/run'
            cfg.freeze()
            with contextlib.redirect_stdout(io.StringIO()):
                trainer = RegionalDINOTrainer(cfg, {'train': loader})
                state = trainer.rate_augment.generator.get_state().clone()
                valid = trainer.valid({'valid': loader})
                self.assertNotIn('rate_augmentation', valid)
                self.assertTrue(torch.equal(state, trainer.rate_augment.generator.get_state()))
                trainer.train({'train': loader, 'valid': loader})
            checkpoint = torch.load(trainer.best_path, weights_only=True)
            self.assertTrue(checkpoint['rate_augmentation_settings']['ENABLED'])
            self.assertIn('generator_state', checkpoint['rate_augmentation_state'])
            self.assertEqual(trainer.scheduler.last_epoch, 4)
            history = [json.loads(x) for x in Path(temporary, 'run/regional_history.jsonl').read_text().splitlines()]
            self.assertEqual(history[0]['train']['rate_augmentation']['applied_fraction'], 1)
            self.assertEqual(history[0]['valid']['samples'], 4)
            self.assertEqual(len(Path(temporary, 'run/rate_augmentation.jsonl').read_text().splitlines()), 8)


if __name__ == '__main__':
    unittest.main()
