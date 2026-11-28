"""Integration checks with small local tensors and no backbone downloads."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import get_config
from neural_methods.model.regional_dino import RegionalDINO
from neural_methods.trainer.RegionalDINOTrainer import (
    RegionalDINOTrainer, audit_batch, build_model, model_arguments,
)
from tools.check_regional_dino import load_cached_subset


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/train_configs/cross/MMPD_PURE_RegionalDINO.yaml'


class TinyDataset(Dataset):
    def __init__(self):
        time = torch.arange(64) / 30
        self.target = torch.stack([torch.sin(2 * torch.pi * f * time) for f in (1.0, 1.0, 1.6, 1.6)])
        self.target -= self.target.mean(-1, keepdim=True)
        texture = 90 + torch.rand(4, 1, 3, 16, 16) * 60
        self.video = texture + 3 * self.target[:, :, None, None, None]

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return self.video[index], self.target[index], f'recording{index // 2}', index % 2


class RegionalTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(21)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        with contextlib.redirect_stdout(io.StringIO()):
            self.config = get_config(argparse.Namespace(config_file=str(CONFIG)))
        self.config.defrost()
        self.config.DEVICE = 'cpu'
        self.config.MODEL.MODEL_DIR = str(self.root / 'checkpoints')
        self.config.LOG.PATH = str(self.root / 'run')
        self.config.TRAIN.EPOCHS = 3
        self.config.TRAIN.REGIONAL_DINO.CHECKPOINT_START_EPOCH = 1
        self.config.MODEL.REGIONAL_DINO.USE_DINO = False
        self.config.MODEL.REGIONAL_DINO.GRID_SIZE = 2
        self.config.MODEL.REGIONAL_DINO.HIDDEN_DIM = 8
        self.config.MODEL.REGIONAL_DINO.CONTEXT_DIM = 4
        self.config.MODEL.REGIONAL_DINO.NUM_TCN_BLOCKS = 1
        self.config.MODEL.REGIONAL_DINO.DROPOUT = 0.0
        self.config.TEST.METRICS = ['MAE']
        for name in ('TRAIN', 'VALID', 'TEST'):
            self.config[name].DATA.PREPROCESS.CHUNK_LENGTH = 64
        self.config.freeze()
        loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False)
        self.loaders = {'train': loader, 'valid': loader, 'test': loader}

    def tearDown(self):
        self.temporary.cleanup()

    def make_trainer(self, config=None, model=None):
        config = self.config if config is None else config
        with contextlib.redirect_stdout(io.StringIO()):
            return RegionalDINOTrainer(config, self.loaders, model=model)

    def test_train_validate_checkpoint_reload_and_raw_test_export(self):
        trainer = self.make_trainer()
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.train(self.loaders)
        checkpoints = list((self.root / 'checkpoints').glob('*.pth'))
        self.assertEqual(len(checkpoints), 2)
        history = [json.loads(line) for line in (self.root / 'run/regional_history.jsonl').read_text().splitlines()]
        self.assertFalse(history[0]['checkpoint_eligible'])
        self.assertTrue(history[1]['checkpoint_eligible'])
        self.assertIn(trainer.best_epoch, (1, 2))
        for row in history:
            self.assertGreater(row['valid']['zero_baseline_mse'], 0)
            self.assertEqual(row['valid']['samples'], 4)
            self.assertLess(row['valid']['undefined_hr_fraction'], 1)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer._load_checkpoint(trainer.best_path)
        trainer.model.eval()
        video = next(iter(self.loaders['test']))[0]
        with torch.no_grad():
            expected = trainer.model(video)

        # Reload via only_test into a newly constructed model, then exercise
        # the real repository evaluator and saved regional tensor structure.
        config = self.config.clone()
        config.defrost()
        config.TOOLBOX_MODE = 'only_test'
        config.INFERENCE.MODEL_PATH = str(trainer.best_path)
        config.LOG.PATH = str(self.root / 'test_run')
        config.freeze()
        reloaded = self.make_trainer(config)
        with contextlib.redirect_stdout(io.StringIO()):
            reloaded.test(self.loaders)
        with torch.no_grad():
            torch.testing.assert_close(reloaded.model(video), expected)
        saved = torch.load(self.root / 'test_run/regional_waveforms.pt', weights_only=True)
        torch.testing.assert_close(saved['pred_raw']['recording0'][0].squeeze(1), expected[0])
        self.assertEqual(saved['normalization'], 'raw')
        self.assertEqual(saved['regional']['recording0'][0]['region_weights'].shape, (4, 64))
        self.assertTrue((self.root / 'test_run/outputs/rPPGandLabel/prediction_recording0.npy').is_file())

    def test_validation_baseline_is_aggregated_over_all_examples(self):
        trainer = self.make_trainer()
        with contextlib.redirect_stdout(io.StringIO()):
            stats = trainer.valid(self.loaders)
        target = self.loaders['valid'].dataset.target
        self.assertAlmostEqual(stats['zero_baseline_mse'], float(target.square().mean()), places=6)

    def test_empty_test_loader_is_rejected_before_loading_checkpoint(self):
        trainer = self.make_trainer()
        with self.assertRaisesRegex(ValueError, 'No test data'):
            trainer.test({'test': DataLoader([])})

    def test_onecycle_preserves_backbone_rate_ratio(self):
        class ParameterGroups(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(3, 3)
                self.pulse = torch.nn.Linear(3, 1)
        trainer = self.make_trainer(model=ParameterGroups())
        steps = self.config.TRAIN.EPOCHS * len(self.loaders['train'])
        for _ in range(steps):
            head, backbone = trainer.optimizer.param_groups
            self.assertAlmostEqual(head['lr'] / backbone['lr'], 10.0, places=6)
            trainer.optimizer.step()
            trainer.scheduler.step()

    def test_input_audit_rejects_normalized_cache_and_wrong_channels(self):
        video, target = next(iter(self.loaders['train']))[:2]
        with self.assertRaisesRegex(ValueError, 'Raw RGB'):
            audit_batch(video / 255, target)
        with self.assertRaises(ValueError):
            audit_batch(torch.cat([video, video], dim=2), target)
        self.assertEqual(audit_batch(video, target)['shape'], list(video.shape))

    def test_cached_subset_matches_toolbox_layout(self):
        dataset = self.loaders['train'].dataset
        paths = []
        for index in range(2):
            path = self.root / f'subject1_input{index}.npy'
            np.save(path, dataset.video[index].permute(0, 2, 3, 1).numpy())
            np.save(self.root / f'subject1_label{index}.npy', dataset.target[index].numpy())
            paths.append(str(path))
        file_list = self.root / 'cache.csv'
        file_list.write_text('input_files\n' + '\n'.join(paths), encoding='utf-8')
        config = self.config.clone()
        config.defrost()
        config.TRAIN.DATA.FILE_LIST_PATH = str(file_list)
        config.freeze()
        with contextlib.redirect_stdout(io.StringIO()):
            video, target = load_cached_subset(config, 2)
        torch.testing.assert_close(video, dataset.video[:2])
        torch.testing.assert_close(target, dataset.target[:2])

    def test_all_experiment_configs_merge_and_build_offline(self):
        paths = [CONFIG] + list((ROOT / 'configs/train_configs/ablations').glob('regional_*.yaml'))
        for path in paths:
            with self.subTest(path=path.name), contextlib.redirect_stdout(io.StringIO()):
                config = get_config(argparse.Namespace(config_file=str(path)))
                config.defrost()
                config.MODEL.REGIONAL_DINO.USE_DINO = False
                config.MODEL.REGIONAL_DINO.FINETUNE_BLOCKS = 0
                config.freeze()
                model = build_model(config)
                self.assertEqual(model.num_regions, 9)
                self.assertIn(config.MODEL.NAME, ('RegionalDINO', 'RegionalSTAttention', 'RegionalFrozenFusion'))


if __name__ == '__main__':
    unittest.main()
