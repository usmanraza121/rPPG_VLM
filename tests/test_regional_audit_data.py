"""Read-only audit cache parsing and gap/subject safety checks."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from tools.regional_audit_data import (
    assert_subject_disjoint, contiguous_segments, extract_rgb_traces,
    load_cached_clip, read_cache_records,
)


def _recording(subject=1, **conditions):
    codes = {"L": 1, "MO": 1, "E": 1, "S": 3, "GE": 1, "GL": 1, "H": 1, "MA": 1}
    codes.update(conditions)
    return "subject" + str(subject) + "".join(f"_{name}{value}" for name, value in codes.items())


class RegionalAuditDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / "input_cache"
        self.folder.mkdir()
        self.csv_path = self.folder / "files.csv"

    def _config(self, paths, dataset="MMPD", info=None):
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["", "input_files"])
            writer.writeheader()
            for i, path in enumerate(paths):
                writer.writerow({"": i, "input_files": str(path)})
        settings = info or dict(LIGHT=[1, 2, 3, 4], MOTION=[1, 2, 3, 4], EXERCISE=[1, 2],
                                SKIN_COLOR=[3, 4, 5, 6], GENDER=[1, 2], GLASSER=[1, 2],
                                HAIR_COVER=[1, 2], MAKEUP=[1, 2])
        data = SimpleNamespace(DATASET=dataset, DATA_FORMAT="NDCHW", FILE_LIST_PATH=str(self.csv_path),
                               INFO=SimpleNamespace(**settings))
        split = SimpleNamespace(DATA=data)
        return SimpleNamespace(TRAIN=split, VALID=split, TEST=split)

    def _clip_paths(self, video=None, target=None):
        path = self.folder / f"{_recording()}_input0.npy"
        label = path.with_name(path.name.replace("_input0", "_label0"))
        np.save(path, np.full((8, 5, 7, 3), 100, dtype=np.uint8) if video is None else video)
        np.save(label, np.arange(8, dtype=np.float32) if target is None else target)
        return path, label

    def test_groups_by_recording_and_sorts_numeric_clips_and_subjects(self):
        names = [(10, 2), (2, 10), (2, 1), (2, 0)]
        paths = [self.folder / f"{_recording(subject)}_input{clip}.npy" for subject, clip in names]
        records = read_cache_records(self._config(paths))
        self.assertEqual([r["subject"] for r in records], ["subject2", "subject10"])
        self.assertEqual([c[0] for c in records[0]["clips"]], [0, 1, 10])
        self.assertEqual(records[0]["clips"][0][2].parent, self.folder)
        self.assertTrue(records[0]["clips"][0][2].name.endswith("_label0.npy"))

    def test_each_mmpd_info_filter_matches_loader_membership(self):
        fields = dict(L="LIGHT", MO="MOTION", E="EXERCISE", S="SKIN_COLOR",
                      GE="GENDER", GL="GLASSER", H="HAIR_COVER", MA="MAKEUP")
        for token, field in fields.items():
            with self.subTest(field=field):
                changed = 4 if token == "S" else 2
                matching = self.folder / f"{_recording()}_input0.npy"
                excluded = self.folder / f"{_recording(**{token: changed})}_input0.npy"
                config = self._config([matching, excluded])
                setattr(config.VALID.DATA.INFO, field, [3 if token == "S" else 1])
                records = read_cache_records(config)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["recording"], _recording())

    def test_pure_integer_cache_ids_preserve_subject_trial_meaning(self):
        paths = [self.folder / f"{name}_input0.npy" for name in ("1006", "101", "0102")]
        records = read_cache_records(self._config(paths, dataset="PURE"))
        self.assertEqual({r["recording"]: r["subject"] for r in records},
                         {"101": "subject01", "0102": "subject01", "1006": "subject10"})

    def test_duplicate_logical_clip_fails_even_with_distinct_paths_or_zeros(self):
        first = self.folder / f"{_recording()}_input1.npy"
        second = self.folder / "other" / f"{_recording()}_input01.npy"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_cache_records(self._config([first, second]))

    def test_malformed_unknown_or_filtered_empty_lists_fail(self):
        for dataset, name in (("MMPD", "mystery_input0.npy"),
                              ("MMPD", f"{_recording()}_inputbad.npy"),
                              ("PURE", "unknown_input0.npy"), ("OTHER", "101_input0.npy")):
            with self.subTest(dataset=dataset, name=name), self.assertRaises(ValueError):
                read_cache_records(self._config([self.folder / name], dataset=dataset))
        config = self._config([self.folder / f"{_recording()}_input0.npy"])
        config.VALID.DATA.INFO.LIGHT = [4]
        with self.assertRaisesRegex(ValueError, "No MMPD"):
            read_cache_records(config)

    def test_missing_input_column_and_empty_rows_fail(self):
        config = self._config([])
        self.csv_path.write_text("wrong_column\nfile.npy\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "input_files column"):
            read_cache_records(config)
        config = self._config([""])
        with self.assertRaisesRegex(ValueError, "Empty input_files"):
            read_cache_records(config)

    def test_subject_overlap_audit_checks_names_and_metadata(self):
        make = lambda n: {"recording": _recording(n), "subject": f"subject{n}", "clips": []}
        assert_subject_disjoint([make(1), make(2)], [make(3)])
        with self.assertRaisesRegex(ValueError, "subject2"):
            assert_subject_disjoint([make(1), make(2)], [make(2)])
        bad = make(1)
        bad["subject"] = "subject99"
        with self.assertRaisesRegex(ValueError, "disagrees"):
            assert_subject_disjoint([bad], [make(2)])
        with self.assertRaisesRegex(ValueError, "Cannot identify"):
            assert_subject_disjoint([{"recording": "unknown", "subject": "subject1"}], [make(2)])
        with self.assertRaisesRegex(ValueError, "empty"):
            assert_subject_disjoint([], [make(2)])

    def test_gaps_create_separate_numeric_sorted_segments(self):
        make = lambda n: (n, Path(f"input{n}.npy"), Path(f"label{n}.npy"))
        record = {"recording": "r", "clips": [make(n) for n in (8, 3, 0, 4, 1, 10)]}
        self.assertEqual([[clip[0] for clip in group] for group in contiguous_segments(record)],
                         [[0, 1], [3, 4], [8], [10]])
        self.assertEqual(contiguous_segments({"clips": []}), [])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            contiguous_segments({"clips": [make(0), make(0)]})

    def test_load_converts_hwc_raw_cache_to_float_rgb_tensor(self):
        video = np.arange(8 * 5 * 7 * 3, dtype=np.uint16).reshape(8, 5, 7, 3) % 255
        x, y = load_cached_clip(*self._clip_paths(video=video))
        self.assertEqual(x.shape, (8, 3, 5, 7))
        self.assertEqual(y.shape, (8,))
        self.assertEqual(x.dtype, torch.float32)
        self.assertTrue(x.is_contiguous())
        np.testing.assert_array_equal(x.numpy(), video.transpose(0, 3, 1, 2))

    def test_load_rejects_wrong_shapes_nonfinite_and_normalized_input(self):
        variants = [np.zeros((8, 3, 5, 7), dtype=np.uint8),
                    np.zeros((8, 5, 7, 6), dtype=np.uint8),
                    np.zeros((8, 2, 7, 3), dtype=np.uint8),
                    np.full((8, 5, 7, 3), 0.5, dtype=np.float32),
                    np.full((8, 5, 7, 3), -1.0, dtype=np.float32),
                    np.full((8, 5, 7, 3), 256.0, dtype=np.float32),
                    np.full((8, 5, 7, 3), np.nan, dtype=np.float32)]
        for video in variants:
            with self.subTest(shape=video.shape, dtype=str(video.dtype)), self.assertRaises(ValueError):
                load_cached_clip(*self._clip_paths(video=video))
        for target in (np.zeros((8, 1)), np.zeros(7), np.full(8, np.inf)):
            with self.assertRaises(ValueError):
                load_cached_clip(*self._clip_paths(target=target))
        # A raw uint8 all-black diagnostic remains a valid cache representation.
        x, _ = load_cached_clip(*self._clip_paths(video=np.zeros((8, 5, 7, 3), dtype=np.uint8)))
        self.assertEqual(x.sum().item(), 0)

    def test_grid_trace_means_use_original_pixels_and_model_cell_order(self):
        video = torch.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7).float()
        global_trace, regional = extract_rgb_traces(video, grid_size=3)
        self.assertEqual(global_trace.shape, (2, 3))
        self.assertEqual(regional.shape, (9, 2, 3))
        torch.testing.assert_close(global_trace, video.mean(dim=(-1, -2)))
        areas = []
        for row in range(3):
            for col in range(3):
                top, bottom = row * 5 // 3, (row + 1) * 5 // 3
                left, right = col * 7 // 3, (col + 1) * 7 // 3
                torch.testing.assert_close(regional[row * 3 + col],
                                           video[:, :, top:bottom, left:right].mean(dim=(-1, -2)))
                areas.append((bottom - top) * (right - left))
        weighted = (regional * torch.tensor(areas)[:, None, None]).sum(dim=0) / (5 * 7)
        torch.testing.assert_close(weighted, global_trace)
        self.assertGreater(global_trace.max().item(), 1)  # no implicit /255

    def test_grid_trace_validation_and_uint8_support(self):
        x = torch.full((4, 3, 6, 6), 120, dtype=torch.uint8)
        global_trace, regional = extract_rgb_traces(x)
        self.assertEqual(global_trace.dtype, torch.float32)
        self.assertEqual(regional.unique().tolist(), [120.0])
        for bad, grid in ((x[:, :2], 3), (x, 0), (x, 7), (x.float() * 3, 3)):
            with self.assertRaises(ValueError):
                extract_rgb_traces(bad, grid)


if __name__ == "__main__":
    unittest.main()
