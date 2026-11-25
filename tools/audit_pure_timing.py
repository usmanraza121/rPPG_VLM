"""Read-only PURE raw timestamps versus the existing index-resampling convention.

No image decoding, face extraction, labels/cache writes, or physiological HR
corrections. Nanosecond timestamp units are inferred from the supplied clocks.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def inspect_record(folder):
    path = folder / (folder.name + '.json')
    payload = json.loads(path.read_text(encoding='utf-8'))
    images, ppg = payload['/Image'], payload['/FullPackage']
    image_ts = np.array([x['Timestamp'] for x in images], dtype=np.int64)
    ppg_ts = np.array([x['Timestamp'] for x in ppg], dtype=np.int64)
    image_dt, ppg_dt = np.diff(image_ts) * 1e-9, np.diff(ppg_ts) * 1e-9
    if not (np.all(image_dt > 0) and np.all(ppg_dt > 0)):
        raise ValueError(f'Nonmonotonic timestamps in {path}')
    png = sorted((folder / folder.name).glob('*.png'))
    png_ts = [int(re.fullmatch(r'Image(\d+)\.png', x.name)[1]) for x in png]
    waveform = np.asarray([x['Value']['waveform'] for x in ppg], dtype=np.float64)
    # Reproduce BaseLoader.resample_ppg, then compare with timestamp interpolation.
    index_based = np.interp(np.linspace(1, len(waveform), len(images)),
                            np.arange(1, len(waveform) + 1), waveform)
    origin = min(image_ts[0], ppg_ts[0])
    timestamp_based = np.interp((image_ts - origin) * 1e-9,
                               (ppg_ts - origin) * 1e-9, waveform)
    reference_std = float(np.std(timestamp_based))
    # How different are the actual PPG time coordinates selected by the legacy map?
    legacy_times = np.interp(np.linspace(1, len(waveform), len(images)),
                             np.arange(1, len(waveform) + 1), (ppg_ts - origin) * 1e-9)
    coordinate_error = legacy_times - (image_ts - origin) * 1e-9
    flags = sum(any(bool(x['Value'].get(k, False)) for k in
                    ('probeError', 'searching', 'searchingToLong', 'droppingo2Sat')) for x in ppg)
    return {'recording': folder.name, 'subject': folder.name[:2],
            'json_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'video_frames_json': len(images), 'png_files': len(png),
            'png_timestamps_match_json': png_ts == image_ts.tolist(),
            'ppg_samples': len(ppg),
            'video_duration_seconds': float((image_ts[-1] - image_ts[0]) * 1e-9),
            'effective_video_fps': float(1 / image_dt.mean()),
            'effective_ppg_hz': float(1 / ppg_dt.mean()),
            'video_interval_median_ms': float(np.median(image_dt) * 1e3),
            'video_interval_max_ms': float(image_dt.max() * 1e3),
            'video_gaps_over_1p5_median': int((image_dt > 1.5 * np.median(image_dt)).sum()),
            'ppg_gaps_over_1p5_median': int((ppg_dt > 1.5 * np.median(ppg_dt)).sum()),
            'first_image_minus_first_ppg_ms': float((image_ts[0] - ppg_ts[0]) * 1e-6),
            'last_image_minus_last_ppg_ms': float((image_ts[-1] - ppg_ts[-1]) * 1e-6),
            'images_outside_ppg_time_support': int(((image_ts < ppg_ts[0]) | (image_ts > ppg_ts[-1])).sum()),
            'legacy_coordinate_error_rms_ms': float(np.sqrt(np.mean(coordinate_error ** 2)) * 1e3),
            'legacy_coordinate_error_max_abs_ms': float(abs(coordinate_error).max() * 1e3),
            'index_vs_timestamp_waveform_corr': (float(np.corrcoef(index_based, timestamp_based)[0, 1])
                                                 if reference_std > 0 and np.std(index_based) > 0 else None),
            'index_vs_timestamp_normalized_rmse': (float(np.sqrt(np.mean((index_based - timestamp_based) ** 2)) / reference_std)
                                                    if reference_std > 0 else None),
            'device_flagged_ppg_samples': flags}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    folders = sorted(p for p in Path(args.raw_root).iterdir()
                     if p.is_dir() and re.fullmatch(r'\d\d-\d\d', p.name))
    if not folders:
        raise ValueError('No PURE subject-trial folders found.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for folder in folders:
        rows.append(inspect_record(folder))
        print('Checked:', folder.name, flush=True)
    with (output / 'recordings.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'raw_root': str(Path(args.raw_root).resolve()), 'recordings': len(rows),
               'subjects': sorted({r['subject'] for r in rows}),
               'all_png_timestamp_lists_match': all(r['png_timestamps_match_json'] for r in rows),
               'video_fps_minmax': [min(r['effective_video_fps'] for r in rows), max(r['effective_video_fps'] for r in rows)],
               'max_legacy_time_coordinate_error_ms': max(r['legacy_coordinate_error_max_abs_ms'] for r in rows),
               'recordings_with_device_flags': sum(r['device_flagged_ppg_samples'] > 0 for r in rows),
               'timestamp_units': '1e-9 seconds, inferred from supplied image/PPG timestamp clocks',
               'warning': 'Timing and interpolation checks are not certification of physiological alignment or reference correctness. No raw data/cache was modified.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
