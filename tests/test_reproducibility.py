"""Seed controls exercised on actual regional models and shuffled training."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_config
from reproducibility import initialize_random_state, validate_seed
from neural_methods.trainer.RegionalDINOTrainer import RegionalDINOTrainer, build_model


ROOT = Path(__file__).resolve().parents[1]
ATTENTION = ROOT / 'configs/train_configs/ablations/regional_st_attention.yaml'


def read_config(path=ATTENTION, **args):
    with contextlib.redirect_stdout(io.StringIO()):
        return get_config(argparse.Namespace(config_file=str(path), **args))


class ReproducibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default_yaml_and_cli_precedence_and_invalid_values(self):
        self.assertEqual(read_config().RANDOM_SEED, 100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'seed.yaml'
            path.write_text('BASE: [' + json.dumps(str(ATTENTION)) + ']\nRANDOM_SEED: 102\n')
            self.assertEqual(read_config(path).RANDOM_SEED, 102)
            self.assertEqual(read_config(path, seed=None).RANDOM_SEED, 102)
            self.assertEqual(read_config(path, seed=101).RANDOM_SEED, 101)
            self.assertEqual(read_config(path, seed=0).RANDOM_SEED, 0)
        for seed in [-1, 2 ** 32, True, 1.5, '101']:
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                validate_seed(seed)
        with self.assertRaises(ValueError):
            read_config(seed=-1)

    def test_seed100_preserves_legacy_initialization_and_generator_setup(self):
        # Original entry-point sequence before the refactor.
        torch.manual_seed(100)
        torch.cuda.manual_seed(100)
        torch.cuda.manual_seed_all(100)
        np.random.seed(100)
        random.seed(100)
        general = torch.Generator().manual_seed(100)
        train = torch.Generator().manual_seed(100)
        config = read_config()
        expected = build_model(config).state_dict()
        expected_order = torch.randperm(20, generator=train)
        expected_random = (random.random(), np.random.rand(), torch.rand(3))
        expected_general = general.get_state()
        new_general, new_train = initialize_random_state(config.RANDOM_SEED)
        actual = build_model(config).state_dict()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        torch.testing.assert_close(torch.randperm(20, generator=new_train), expected_order)
        self.assertEqual(random.random(), expected_random[0])
        self.assertEqual(np.random.rand(), expected_random[1])
        torch.testing.assert_close(torch.rand(3), expected_random[2], rtol=0, atol=0)
        self.assertTrue(torch.equal(new_general.get_state(), expected_general))

    def test_repeated_seed_reproduces_initialization_shuffle_and_training(self):
        # Fixed local data; do not regenerate the data for each training seed.
        time = torch.arange(32) / 30
        labels = torch.stack([torch.sin(2 * torch.pi * f * time) for f in [1., 1.2, 1.4, 1.6]])
        texture = torch.linspace(100, 140, 4 * 3 * 8 * 8).reshape(4, 1, 3, 8, 8)
        videos = texture + 2 * labels[:, :, None, None, None]
        samples = [(videos[i], labels[i], str(i), 0) for i in range(4)]

        def run(seed, backend, root):
            config = read_config(seed=seed)
            config.defrost()
            config.DEVICE = 'cpu'
            config.TRAIN.EPOCHS = 2
            config.TRAIN.REGIONAL_DINO.CHECKPOINT_START_EPOCH = 0
            config.MODEL.REGIONAL_ST.TEMPORAL_BACKEND = backend
            config.MODEL.REGIONAL_ST.GRID_SIZE = 1
            config.MODEL.REGIONAL_ST.HIDDEN_DIM = 8
            config.MODEL.REGIONAL_ST.CONTEXT_DIM = 4
            config.MODEL.MODEL_DIR = str(root / 'checkpoints')
            config.LOG.PATH = str(root / 'run')
            config.freeze()
            general, train = initialize_random_state(seed)
            loader = DataLoader(samples, batch_size=2, shuffle=True, generator=train)
            valid = DataLoader(samples, batch_size=2, generator=general)
            # The separately seeded sampler probe does not advance the real loader.
            order = list(iter(DataLoader(list(range(20)), batch_size=20, shuffle=True,
                                        generator=torch.Generator().manual_seed(seed))))[0]
            with contextlib.redirect_stdout(io.StringIO()):
                trainer = RegionalDINOTrainer(config, {'train': loader, 'valid': valid})
                initial = {k: v.clone() for k, v in trainer.core.state_dict().items()}
                trainer.train({'train': loader, 'valid': valid})
            state = {k: v.clone() for k, v in trainer.core.state_dict().items()}
            checkpoint = torch.load(trainer.best_path, weights_only=True)
            self.assertEqual(checkpoint['random_seed'], seed)
            rows = [json.loads(line) for line in (root / 'run/regional_history.jsonl').read_text().splitlines()]
            self.assertTrue(all(row['random_seed'] == seed for row in rows))
            # Iterating validation cannot advance the independent training RNG.
            before = train.get_state().clone()
            list(valid)
            self.assertTrue(torch.equal(before, train.get_state()))
            return initial, state, order, [row['valid']['total'] for row in rows]

        with tempfile.TemporaryDirectory() as directory:
            for backend in ['attention', 'tcn']:
                with self.subTest(backend=backend):
                    root = Path(directory) / backend
                    first = run(101, backend, root / 'first')
                    repeat = run(101, backend, root / 'repeat')
                    other = run(102, backend, root / 'other')
                    for index in [0, 1]:
                        for key in first[index]:
                            torch.testing.assert_close(first[index][key], repeat[index][key], rtol=0, atol=0)
                    self.assertTrue(torch.equal(first[2], repeat[2]))
                    self.assertEqual(first[3], repeat[3])
                    self.assertFalse(torch.equal(first[2], other[2]))
                    self.assertTrue(any(not torch.equal(first[0][k], other[0][k]) for k in first[0]))


if __name__ == '__main__':
    unittest.main()
