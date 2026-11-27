"""Exercise the new model through the real regional trainer and audit paths."""

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_config
from reproducibility import initialize_random_state
from neural_methods.model.regional_st_attention import RegionalSTAttention
from neural_methods.trainer.RegionalDINOTrainer import (
    RegionalDINOTrainer, build_model, model_arguments, validate_model_checkpoint,
)
from tools.audit_regional_dino import main as audit_main


ROOT = Path(__file__).resolve().parents[1]
ATTENTION = ROOT / 'configs/train_configs/ablations/regional_st_attention.yaml'


def read_config(path=ATTENTION):
    with contextlib.redirect_stdout(io.StringIO()):
        return get_config(argparse.Namespace(config_file=str(path)))


class RegionalSTIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_factory_and_matched_training_settings(self):
        original = read_config(ROOT / 'configs/train_configs/ablations/regional_shallow_learned.yaml')
        variants = {
            'regional_st_attention.yaml': (RegionalSTAttention, 101368),
            'regional_st_tcn.yaml': (RegionalSTAttention, 100936),
            'regional_shallow_long_tcn.yaml': (None, 91024),
        }
        for name, (kind, count) in variants.items():
            config = read_config(ATTENTION.with_name(name))
            model = build_model(config)
            if kind is not None:
                self.assertIsInstance(model, kind)
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            self.assertFalse(model.use_dino)
            self.assertFalse(config.TRAIN.REGIONAL_DINO.RATE_AUG.ENABLED)
            self.assertEqual(config.TRAIN.EPOCHS, original.TRAIN.EPOCHS)
            self.assertEqual(config.TRAIN.BATCH_SIZE, original.TRAIN.BATCH_SIZE)
            self.assertEqual(config.TRAIN.LR, original.TRAIN.LR)
            self.assertEqual(config.TRAIN.REGIONAL_DINO, original.TRAIN.REGIONAL_DINO)
            for split in ('TRAIN', 'VALID', 'TEST'):
                self.assertEqual(config[split].DATA, original[split].DATA)
        # Preserve the old constructor dictionary and old unnamed checkpoints.
        args = model_arguments(original)
        self.assertEqual(args, {k.lower(): v for k, v in original.MODEL.REGIONAL_DINO.items()})
        validate_model_checkpoint({'model_state': {}, 'model_arguments': args}, original)

    def test_trainer_roundtrip_exports_and_strict_checkpoint_identity(self):
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = read_config()
            config.defrost()
            config.DEVICE = 'cpu'
            config.RANDOM_SEED = 101
            config.TRAIN.EPOCHS = 3
            config.TRAIN.REGIONAL_DINO.CHECKPOINT_START_EPOCH = 1
            config.MODEL.REGIONAL_ST.GRID_SIZE = 2
            config.MODEL.REGIONAL_ST.HIDDEN_DIM = 8
            config.MODEL.REGIONAL_ST.CONTEXT_DIM = 4
            config.MODEL.REGIONAL_ST.DROPOUT = 0.
            config.MODEL.MODEL_DIR = str(root / 'checkpoints')
            config.LOG.PATH = str(root / 'run')
            config.TEST.METRICS = ['MAE']
            config.freeze()
            initialize_random_state(config.RANDOM_SEED)
            t = torch.arange(64) / 30
            targets = torch.stack([torch.sin(2 * torch.pi * f * t) for f in (1., 1., 1.5, 1.5)])
            texture = 110 + torch.rand(4, 1, 3, 16, 16) * 10
            videos = texture + 2 * targets[:, :, None, None, None]
            samples = [(videos[i], targets[i], f'recording{i // 2}', i % 2) for i in range(4)]
            loader = DataLoader(samples, batch_size=2, shuffle=False)
            loaders = {'train': loader, 'valid': loader, 'test': loader}
            with contextlib.redirect_stdout(io.StringIO()):
                trainer = RegionalDINOTrainer(config, loaders)
                trainer.train(loaders)
            self.assertEqual(len(list((root / 'checkpoints').glob('*.pth'))), 2)
            payload = torch.load(trainer.best_path, weights_only=True)
            self.assertEqual(payload['model_name'], 'RegionalSTAttention')
            self.assertEqual(payload['random_seed'], 101)
            validate_model_checkpoint(payload, config)
            backend = config.clone()
            backend.defrost()
            backend.MODEL.REGIONAL_ST.TEMPORAL_BACKEND = 'tcn'
            backend.freeze()
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                validate_model_checkpoint(payload, backend)
            unnamed = dict(payload)
            unnamed.pop('model_name')
            with self.assertRaisesRegex(ValueError, 'model name'):
                validate_model_checkpoint(unnamed, config)
            original = read_config(ROOT / 'configs/train_configs/ablations/regional_shallow_learned.yaml')
            with self.assertRaisesRegex(ValueError, 'model name'):
                validate_model_checkpoint(payload, original)
            with contextlib.redirect_stdout(io.StringIO()):
                trainer._load_checkpoint(trainer.best_path)
            trainer.model.eval()
            with torch.no_grad():
                expected = trainer.model(videos[:2])
            test_config = config.clone()
            test_config.defrost()
            test_config.TOOLBOX_MODE = 'only_test'
            test_config.RANDOM_SEED = 100  # Inference seed need not match training.
            test_config.INFERENCE.MODEL_PATH = str(trainer.best_path)
            test_config.LOG.PATH = str(root / 'test_run')
            test_config.freeze()
            with contextlib.redirect_stdout(io.StringIO()):
                reloaded = RegionalDINOTrainer(test_config, {'test': loader})
                reloaded.test({'test': loader})
            with torch.no_grad():
                torch.testing.assert_close(reloaded.model(videos[:2]), expected)
            saved = torch.load(root / 'test_run/regional_waveforms.pt', weights_only=True)
            torch.testing.assert_close(saved['pred_raw']['recording0'][0].squeeze(1), expected[0])
            self.assertEqual(saved['normalization'], 'raw')
            self.assertEqual(saved['training_seed'], 101)
            self.assertEqual(saved['inference_seed'], 100)
            self.assertEqual(saved['regional']['recording0'][0]['regional_waveforms'].shape, (4, 64))
            self.assertFalse((root / 'run/rate_augmentation.jsonl').exists())

    def test_audit_cli_loads_new_model_and_saves_recording_waveforms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = 'subject2_L1_MO1_E2_S4_GE1_GL1_H2_MA2'
            paths = []
            for clip in range(2):
                t = (np.arange(160) + clip * 160) / 30
                pulse = np.sin(2 * np.pi * 1.2 * t)
                video = 120 + pulse[:, None, None, None] * np.array([.3, 1., .1])
                video = np.broadcast_to(video, (160, 9, 9, 3)).astype('float32').copy()
                path = root / f'{prefix}_input{clip}.npy'
                np.save(path, video)
                np.save(root / f'{prefix}_label{clip}.npy', pulse.astype('float32'))
                paths.append(str(path))
            train_csv, valid_csv = root / 'train.csv', root / 'valid.csv'
            for file, inputs in ((valid_csv, paths), (train_csv, [paths[0].replace('subject2_', 'subject1_')])):
                with file.open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['input_files'])
                    writer.writerows([[p] for p in inputs])
            cfg_path = root / 'audit.yaml'
            cfg_path.write_text(
                'BASE: [' + json.dumps(str(ATTENTION)) + ']\n'
                'TRAIN:\n  DATA:\n    FILE_LIST_PATH: ' + json.dumps(str(train_csv)) + '\n'
                'VALID:\n  DATA:\n    FILE_LIST_PATH: ' + json.dumps(str(valid_csv)) + '\n'
                'MODEL:\n  REGIONAL_ST:\n    HIDDEN_DIM: 8\n    CONTEXT_DIM: 4\n', encoding='utf-8')
            cfg = read_config(cfg_path)
            initialize_random_state(101)
            model = build_model(cfg)
            checkpoint = root / 'best.pth'
            torch.save({'model_name': cfg.MODEL.NAME, 'model_arguments': model_arguments(cfg),
                        'model_state': model.state_dict(), 'epoch': 5, 'random_seed': 101}, checkpoint)
            output = root / 'audit'
            with contextlib.redirect_stdout(io.StringIO()):
                result = audit_main(['--config', str(cfg_path), '--checkpoint', str(checkpoint),
                                     '--device', 'cpu', '--batch-size', '2', '--output', str(output),
                                     '--plot-recordings', '0', '--save-waveforms'])
            self.assertEqual(result['evaluated_recordings'], 1)
            self.assertEqual(result['evaluated_frames'], 320)
            self.assertEqual(result['omitted_tail_frames'], 0)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['model_name'], 'RegionalSTAttention')
            self.assertEqual(manifest['checkpoint_training_seed'], 101)
            saved = next((output / 'waveforms').glob('*.npz'))
            with np.load(saved) as arrays:
                self.assertEqual(arrays['candidates'].shape, (9, 320))
                self.assertEqual(arrays['learned'].shape, (320,))


if __name__ == '__main__':
    unittest.main()
