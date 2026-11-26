"""Scientific controls for the validation-only decoder diagnostic."""

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from config import get_config
from neural_methods.model.regional_dino import RegionalDINO
from neural_methods.trainer.RegionalDINOTrainer import model_arguments
from tools.check_regional_protocol import (
    comparison_rows, disjoint_subjects, main, peak_hr, summarize,
)


BASE = Path(__file__).resolve().parents[1] / 'configs/train_configs/intra/PURE_PURE_RegionalShallowLearned.yaml'


class RegionalProtocolTests(unittest.TestCase):
    def test_known_fundamental_harmonic_mixture_exposes_filter_edge(self):
        time = np.arange(1920) / 30
        values = (np.sin(2 * np.pi * .756 * time)
                  + .65 * np.sin(4 * np.pi * .756 * time + .3))
        self.assertAlmostEqual(peak_hr(values, 'current', 30), 90.72, delta=1)
        self.assertAlmostEqual(peak_hr(values, 'detrend_only', 30), 45.36, delta=1)
        # A wider passband is a candidate, not a guarantee of fundamental
        # recovery when the waveform contains a sufficiently strong harmonic.
        for protocol in ('current', 'detrend_only', 'wider_filter'):
            pure = np.sin(2 * np.pi * 1.2 * time)
            self.assertAlmostEqual(peak_hr(pure, protocol, 30), 72, delta=1)

    def test_fixed_reference_does_not_move_with_protocol(self):
        time = np.arange(1920) / 30
        target = (np.sin(2 * np.pi * .756 * time)
                  + .65 * np.sin(4 * np.pi * .756 * time + .3))
        prediction = np.sin(2 * np.pi * .756 * time)
        rows = comparison_rows(prediction, target, 30, {}, 'learned')
        self.assertEqual(len({r['fixed_reference_hr_bpm'] for r in rows}), 1)
        self.assertGreater(abs(rows[0]['fixed_reference_hr_bpm']
                               - rows[1]['reference_hr_under_this_protocol_bpm']), 30)
        self.assertGreater(rows[1]['absolute_error_bpm'], 30)

    def test_undefined_output_is_counted_and_subject_overlap_rejected(self):
        time = np.arange(160) / 30
        rows = comparison_rows(np.zeros(160), np.sin(2 * np.pi * time),
                               30, {}, 'learned')
        self.assertTrue(all(r['unusable'] == 1 and r['mae_bpm'] is None
                            for r in summarize(rows)))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            disjoint_subjects({'train': [{'subject': 'subject01'}],
                               'valid': [{'subject': 'subject01'}]})

    def test_cli_uses_training_labels_validation_rgb_and_never_test_arrays(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lists = {}
            for split, recording in [('TRAIN', '101'), ('VALID', '701'), ('TEST', '901')]:
                input_path = root / f'{recording}_input0.npy'
                time = np.arange(160) / 30
                pulse = np.sin(2 * np.pi * 1.2 * time).astype('float32')
                if split != 'TEST':
                    np.save(root / f'{recording}_label0.npy', pulse)
                if split == 'VALID':
                    video = 120 + pulse[:, None, None, None] * np.array([.3, 1., .1])
                    np.save(input_path, np.broadcast_to(video, (160, 9, 9, 3)).copy())
                # Training RGB and both test arrays deliberately do not exist.
                path = root / f'{split.lower()}.csv'
                with path.open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['input_files'])
                    writer.writerow([str(input_path)])
                lists[split] = path
            cfg = root / 'protocol.yaml'
            cfg.write_text('BASE: [' + json.dumps(str(BASE)) + ']\n'
                           + ''.join(f'{s}:\n  DATA:\n    FILE_LIST_PATH: {json.dumps(str(p))}\n'
                                     for s, p in lists.items())
                           + 'MODEL:\n  REGIONAL_DINO:\n    HIDDEN_DIM: 8\n'
                             '    CONTEXT_DIM: 4\n    NUM_TCN_BLOCKS: 1\n', encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                config = get_config(argparse.Namespace(config_file=str(cfg)))
            arguments = model_arguments(config)
            model = RegionalDINO(**arguments)
            checkpoint = root / 'best.pth'
            torch.save({'model_arguments': arguments, 'model_state': model.state_dict(),
                        'epoch': 25}, checkpoint)
            output = root / 'results'
            with contextlib.redirect_stdout(io.StringIO()), patch(
                    'tools.check_regional_protocol.synthetic_cases', return_value=[]):
                result = main(['--config', str(cfg), '--checkpoint', str(checkpoint),
                               '--device', 'cpu', '--output', str(output)])
            self.assertEqual(result['training_clip_hr_bins']['total'], 1)
            self.assertEqual(len(result['validation']), 15)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertFalse(manifest['test_waveforms_loaded'])
            self.assertEqual(manifest['subjects']['PURE']['test'], ['subject09'])
            self.assertTrue((output / 'validation_protocols.csv').is_file())


if __name__ == '__main__':
    unittest.main()
