"""Checkpoint audit wiring on synthetic cached clips; no DINO downloads."""

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

from config import get_config
from neural_methods.model.regional_dino import RegionalDINO
from neural_methods.trainer.RegionalDINOTrainer import model_arguments
from tools.audit_regional_dino import (
    analyze_segment, fusion_controls, main, plot_panel, summarize, window_slices,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'configs/train_configs/cross/MMPD_PURE_RegionalDINO.yaml'


class RegionalAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_uniform_control_uses_the_same_candidates_without_reference(self):
        waves = torch.tensor([[[1., 2., 3.], [3., 1., 4.], [100., 100., 100.]]])
        valid = torch.tensor([[True, True, False]])
        output = {'waveform': torch.tensor([[8., 9., 10.]]),
                  'regional_waveforms': waves, 'valid_regions': valid}
        fused, uniform, regions = fusion_controls(output)
        torch.testing.assert_close(fused, output['waveform'])
        expected = waves[:, :2].mean(1)
        expected -= expected.mean(-1, keepdim=True)
        torch.testing.assert_close(uniform, expected)
        self.assertEqual(float(regions[:, 2].abs().sum()), 0)

    def test_oracle_is_explicit_and_harmonic_failure_is_detected(self):
        time = np.arange(600) / 30
        target = np.sin(2 * np.pi * 0.9 * time)
        wrong = np.sin(2 * np.pi * 1.8 * time)
        rgb = 120 + target[:, None] * np.array([.3, 1., .1])[None]
        data = {'reference': target, 'rgb_global': rgb,
                'rgb_regions': np.stack([rgb] * 9), 'learned': wrong,
                'uniform': (target * 8 + wrong) / 9,
                'candidates': np.stack([target] * 8 + [wrong]),
                'weights': np.ones((9, 600)) / 9}
        methods, regions, refs, decisions, panels = analyze_segment(
            data, {'recording': 'example', 'subject': 'subject2', 'segment': 0},
            30, .75, 2.5, 0)
        indexed = {row['method']: row for row in methods}
        self.assertGreater(indexed['learned']['hr_abs_error_bpm'], 40)
        self.assertTrue(indexed['learned']['near_double_reference'])
        self.assertLess(indexed['uniform']['hr_abs_error_bpm'], 1)
        self.assertLess(indexed['oracle_region_hr']['hr_abs_error_bpm'], 1)
        self.assertTrue(indexed['oracle_region_hr']['reference_dependent_oracle'])
        self.assertTrue(decisions[0]['candidate_correct_but_fusion_wrong'])
        self.assertFalse(decisions[0]['no_candidate_within_5_bpm'])
        self.assertEqual(len(regions), 36)

    def test_failed_predictions_are_counted_and_penalized(self):
        time = np.arange(300) / 30
        target = np.sin(2 * np.pi * 1.2 * time)
        rgb = np.ones((300, 3)) * 120
        data = {'reference': target, 'rgb_global': rgb,
                'rgb_regions': np.stack([rgb] * 9), 'learned': np.zeros(300),
                'uniform': np.zeros(300), 'candidates': np.zeros((9, 300)),
                'weights': np.ones((9, 300)) / 9}
        methods, _, _, decisions, _ = analyze_segment(data, {}, 30, .75, 2.5, 0)
        result = summarize(methods)
        self.assertEqual(result['learned']['windows_usable'], 0)
        self.assertEqual(result['learned']['reference_usable_windows'], 1)
        self.assertEqual(result['learned']['mae_with_failure_penalty_bpm'], 105)
        self.assertEqual(result['oracle_region_hr']['windows_total'], 1)
        self.assertTrue(decisions[0]['no_candidate_within_5_bpm'])

    def test_windows_preserve_tail_without_inventing_samples(self):
        self.assertEqual([(s.start, s.stop) for s in window_slices(650, 30, 10)],
                         [(0, 300), (300, 600), (600, 650)])
        self.assertEqual([(s.start, s.stop) for s in window_slices(610, 30, 10)],
                         [(0, 300), (300, 600)])
        self.assertEqual(window_slices(20, 30, 0), [])

    def test_flat_reference_is_reported_without_blame_on_experts(self):
        rgb = np.ones((300, 3)) * 120
        data = {'reference': np.zeros(300), 'rgb_global': rgb,
                'rgb_regions': np.stack([rgb] * 9), 'learned': np.zeros(300),
                'uniform': np.zeros(300), 'candidates': np.zeros((9, 300)),
                'weights': np.ones((9, 300)) / 9}
        _, _, refs, decisions, _ = analyze_segment(data, {}, 30, .75, 2.5, 0)
        self.assertEqual(refs[0]['reference_std'], 0)
        self.assertTrue(refs[0]['reference_flat'])
        self.assertIsNone(decisions[0]['no_candidate_within_5_bpm'])

    def test_invalid_plot_does_not_abort_remaining_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.png'
            plot_panel(path, {'recording': 'invalid', 'segment': 0, 'window': 0},
                       {'reference': np.full(300, np.nan), 'GREEN': np.zeros(300)},
                       [], 30, .75, 2.5)
            self.assertTrue(path.is_file())

    def test_cli_checkpoint_cache_reports_and_plots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = 'subject2_L1_MO1_E2_S4_GE1_GL1_H2_MA2'
            paths = []
            for clip in range(2):
                time = (np.arange(160) + clip * 160) / 30
                pulse = np.sin(2 * np.pi * 1.2 * time)
                video = (120 + pulse[:, None, None, None]
                         * np.array([.3, 1, .1])[None, None, None])
                video = np.broadcast_to(video, (160, 9, 9, 3)).astype('float32').copy()
                input_path = root / f'{prefix}_input{clip}.npy'
                np.save(input_path, video)
                np.save(root / f'{prefix}_label{clip}.npy', pulse.astype('float32'))
                paths.append(str(input_path))
            valid_csv, train_csv = root / 'valid.csv', root / 'train.csv'
            for path, inputs in ((valid_csv, paths), (train_csv, [paths[0].replace('subject2_', 'subject1_')])):
                with path.open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['input_files'])
                    writer.writerows([[p] for p in inputs])
            config_path = root / 'audit.yaml'
            config_path.write_text(
                'BASE: [' + json.dumps(str(BASE)) + ']\n'
                'TRAIN:\n  DATA:\n    FILE_LIST_PATH: ' + json.dumps(str(train_csv)) + '\n'
                'VALID:\n  DATA:\n    FILE_LIST_PATH: ' + json.dumps(str(valid_csv)) + '\n'
                'MODEL:\n  REGIONAL_DINO:\n    USE_DINO: False\n'
                '    HIDDEN_DIM: 8\n    CONTEXT_DIM: 4\n    NUM_TCN_BLOCKS: 1\n',
                encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                config = get_config(argparse.Namespace(config_file=str(config_path)))
            arguments = model_arguments(config)
            torch.manual_seed(11)
            model = RegionalDINO(**arguments)
            checkpoint = root / 'best.pth'
            torch.save({'model_arguments': arguments, 'model_state': model.state_dict(), 'epoch': 37}, checkpoint)
            output = root / 'audit_output'
            with contextlib.redirect_stdout(io.StringIO()):
                result = main(['--config', str(config_path), '--checkpoint', str(checkpoint),
                               '--device', 'cpu', '--batch-size', '2', '--output', str(output),
                               '--plot-recordings', '1', '--save-waveforms'])
            self.assertEqual(result['evaluated_recordings'], 1)
            self.assertEqual(result['paired_fusion']['reference_usable_windows'], 1)
            for name in ('methods.csv', 'records.csv', 'regional.csv', 'reference_checks.csv',
                         'fusion_decisions.csv', 'coverage.csv', 'manifest.json', 'summary.json',
                         'plots/01_audit.png', 'plots/01_regional_spectra.png'):
                self.assertTrue((output / name).is_file(), name)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertFalse(manifest['ground_truth_certified'])
            self.assertEqual(manifest['checkpoint_epoch'], 37)
            self.assertEqual(manifest['train_subjects'], ['subject1'])
            self.assertEqual(manifest['audited_subjects'], ['subject2'])
            self.assertEqual(result['omitted_tail_frames'], 0)
            self.assertTrue(any((output / 'waveforms').glob('*.npz')))


if __name__ == '__main__':
    unittest.main()
