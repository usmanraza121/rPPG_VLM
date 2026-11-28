"""Read-only source-validation fusion and reference-signal audit.

This tool does not train, change a checkpoint, repair labels, or use the
reference to select a deployable region. Reference-selected regions are
explicitly labelled oracle diagnostics. Run from the repository root.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config
from neural_methods.trainer.RegionalDINOTrainer import (
    build_model, model_arguments, validate_model_checkpoint,
)
from tools.regional_audit_data import (
    read_cache_records, assert_subject_disjoint, contiguous_segments,
    load_cached_clip, extract_rgb_traces,
)
from tools.regional_signal_diagnostics import (
    analyze_signal, classical_signals, preprocess_signal, spectrum,
)


def write_csv(path, rows):
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ['status']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (dict, list))
                             else value for key, value in row.items()})


def fusion_controls(output):
    """Use the same regional candidates for learned and uniform fusion.

    Each model clip already has its DC removed. Remove uniform fused DC too;
    no variance normalization or GT information enters either reconstruction.
    Individual candidates are centered to remove the model's shared fused DC.
    """
    regional = output['regional_waveforms']
    valid = output['valid_regions'].to(regional.dtype)
    uniform = (regional * valid[:, :, None]).sum(1) / valid.sum(1, keepdim=True)
    uniform = uniform - uniform.mean(-1, keepdim=True)
    candidates = regional - regional.mean(-1, keepdim=True)
    candidates = candidates.masked_fill(~output['valid_regions'][:, :, None], 0)
    return output['waveform'], uniform, candidates


def infer_segment(clips, model, grid_size, batch_size, device):
    """Stream images in batches; keep only waveforms and regional RGB means."""
    reference, global_rgb, region_rgb = [], [], []
    learned, uniform, candidates, weights = [], [], [], []
    lengths = set()
    shape = None
    for start in range(0, len(clips), batch_size):
        batch = [load_cached_clip(input_path, label_path)
                 for _, input_path, label_path in clips[start:start + batch_size]]
        videos, targets = zip(*batch)
        for video, target in batch:
            if shape is None:
                shape = tuple(video.shape)
            elif tuple(video.shape) != shape:
                raise ValueError('Cached clip shapes change within a recording; do not stack or silently resize.')
            whole, regions = extract_rgb_traces(video, grid_size)
            global_rgb.append(np.asarray(whole))
            region_rgb.append(np.asarray(regions))
            reference.append(target.numpy())
            lengths.add(len(target))
        if model is not None:
            with torch.no_grad():
                output = model(torch.stack(videos).to(device), return_aux=True)
                fused, equal, local = fusion_controls(output)
            learned.extend(fused.cpu().numpy())
            uniform.extend(equal.cpu().numpy())
            candidates.extend(local.cpu().numpy())
            weights.extend(output['region_weights'].cpu().numpy())
    result = {'reference': np.concatenate(reference), 'rgb_global': np.concatenate(global_rgb),
              'rgb_regions': np.concatenate(region_rgb, axis=1),
              'clip_length': next(iter(lengths)), 'clip_count': len(clips)}
    if model is not None:
        result.update(learned=np.concatenate(learned), uniform=np.concatenate(uniform),
                      candidates=np.concatenate(candidates, axis=1),
                      weights=np.concatenate(weights, axis=1))
    return result


def window_slices(length, fs, window_seconds):
    """Return full segment or nonoverlapping windows, never join missing clips."""
    width = length if window_seconds == 0 else int(round(window_seconds * fs))
    minimum = max(12, int(math.ceil(fs)))
    if window_seconds and width < minimum:
        raise ValueError('Evaluation windows must be at least one second.')
    if length < minimum:
        return []
    return [slice(start, min(start + width, length)) for start in range(0, length, width)
            if min(width, length - start) >= minimum]


def finite_mean(values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def summarize(rows, failure_penalty_bpm=105.0):
    result = {}
    for name in sorted({row['method'] for row in rows}):
        selected = [row for row in rows if row['method'] == name]
        ref_usable = [row for row in selected if row['reference_hr_bpm'] is not None]
        usable = [row for row in selected if row['hr_abs_error_bpm'] is not None]
        errors = [row['hr_abs_error_bpm'] for row in usable]
        prediction = [row['hr_bpm'] for row in usable]
        reference = [row['reference_hr_bpm'] for row in usable]
        corr = (float(np.corrcoef(reference, prediction)[0, 1])
                if len(usable) > 2 and np.std(prediction) > 1e-8 and np.std(reference) > 1e-8 else None)
        result[name] = {'windows_total': len(selected), 'windows_usable': len(usable),
                        'windows_unusable': len(selected) - len(usable),
                        'reference_usable_windows': len(ref_usable),
                        'mae_bpm': finite_mean(errors),
                        'mae_with_failure_penalty_bpm': (
                            (sum(errors) + (len(ref_usable) - len(usable)) * failure_penalty_bpm)
                            / len(ref_usable) if ref_usable else None),
                        'median_ae_bpm': float(np.median(errors)) if errors else None,
                        'rmse_bpm': float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                        'hr_pearson': corr,
                        'within_5_bpm': sum(error <= 5 for error in errors),
                        'near_double_count': sum(row['near_double_reference'] is True for row in usable),
                        'near_half_count': sum(row['near_half_reference'] is True for row in usable),
                        'waveform_corr': finite_mean([row['waveform_corr'] for row in usable]),
                        'snr_fundamental_db': finite_mean([row['snr_fundamental_db'] for row in usable]),
                        'snr_harmonics_db': finite_mean([row['snr_harmonics_db'] for row in usable]),
                        'reference_dependent_oracle': name.startswith('oracle_')}
    return result


def diagnostic_row(signal, reference, metadata, method, fs, min_hz, max_hz, **extra):
    result = analyze_signal(signal, reference, fs=fs, min_hz=min_hz, max_hz=max_hz)
    if result['reference_hr_bpm'] is None:
        # A flat prediction must not make a valid reference disappear from
        # coverage or failure-penalized comparisons.
        ref = analyze_signal(reference, reference, fs=fs, min_hz=min_hz, max_hz=max_hz)
        result['reference_hr_bpm'] = ref['hr_bpm']
        result['reference_status'] = ref['status']
        result['reference_reason'] = ref['reason']
        result['reference_peak_hz'] = ref['reference_peak_hz']
        result['reference_autocorr_hr_bpm'] = ref['reference_autocorr_hr_bpm']
        result['reference_std'] = ref['prediction_std']
        result['reference_flat'] = (ref['prediction_std'] <= 1e-12
                                    if ref['prediction_std'] is not None else None)
    return {**metadata, 'method': method, **result, **extra}


def analyze_segment(data, metadata, fs, min_hz, max_hz, window_seconds):
    """Return method, individual-region, reference and decision rows."""
    methods, regions, references, decisions, panels = [], [], [], [], []
    classical = classical_signals(data['rgb_global'], fs)
    regional_classical = [classical_signals(trace, fs) for trace in data['rgb_regions']]
    # A fixed coordinate control, selected independently of the reference.
    central = len(regional_classical) // 2
    for window_index, interval in enumerate(window_slices(len(data['reference']), fs, window_seconds)):
        info = {**metadata, 'window': window_index, 'start_seconds': interval.start / fs,
                'duration_seconds': (interval.stop - interval.start) / fs,
                'frames': interval.stop - interval.start}
        target = data['reference'][interval]
        ref_row = diagnostic_row(target, target, info, 'reference_self', fs, min_hz, max_hz)
        references.append(ref_row)
        local_model = []
        traces = {'reference': target}
        if 'learned' in data:
            fused_rows = {}
            for name in ('learned', 'uniform'):
                wave = data[name][interval]
                row = diagnostic_row(wave, target, info, name, fs, min_hz, max_hz)
                methods.append(row)
                fused_rows[name] = row
                traces[name] = wave
            for region, wave in enumerate(data['candidates']):
                mean_weight = float(data['weights'][region, interval].mean())
                row = diagnostic_row(wave[interval], target, info, 'regional_candidate', fs,
                                     min_hz, max_hz, region=region, mean_weight=mean_weight)
                regions.append(row)
                local_model.append(row)
            usable = [row for row in local_model if row['hr_abs_error_bpm'] is not None]
            oracle = None
            if usable:
                oracle = min(usable, key=lambda row: row['hr_abs_error_bpm'])
                oracle_row = {**oracle, 'method': 'oracle_region_hr', 'reference_dependent_oracle': True}
                methods.append(oracle_row)
                traces['oracle_region_hr'] = data['candidates'][oracle['region'], interval]
            else:
                methods.append({**local_model[0], 'method': 'oracle_region_hr', 'region': None,
                                'reference_dependent_oracle': True})
            learned_row, uniform_row = fused_rows['learned'], fused_rows['uniform']
            decisions.append({**info,
                'learned_ae_bpm': learned_row['hr_abs_error_bpm'],
                'uniform_ae_bpm': uniform_row['hr_abs_error_bpm'],
                'reference_hr_bpm': learned_row['reference_hr_bpm'],
                'oracle_region': oracle['region'] if oracle else None,
                'oracle_ae_bpm': oracle['hr_abs_error_bpm'] if oracle else None,
                'candidate_correct_but_fusion_wrong': bool(
                    oracle and oracle['hr_abs_error_bpm'] <= 5 and
                    (learned_row['hr_abs_error_bpm'] is None or learned_row['hr_abs_error_bpm'] > 10)),
                'no_candidate_within_5_bpm': (
                    oracle is None or oracle['hr_abs_error_bpm'] > 5
                    if learned_row['reference_hr_bpm'] is not None else None)})
        for name, wave in classical.items():
            methods.append(diagnostic_row(wave[interval], target, info, name, fs, min_hz, max_hz))
            methods.append(diagnostic_row(regional_classical[central][name][interval], target,
                                          info, name + '_central_cell', fs, min_hz, max_hz,
                                          region=central))
            traces[name] = wave[interval]
            classical_rows = []
            for region, regional_signals in enumerate(regional_classical):
                row = diagnostic_row(regional_signals[name][interval], target, info,
                                     name + '_regional', fs, min_hz, max_hz, region=region)
                regions.append(row)
                classical_rows.append(row)
            usable = [row for row in classical_rows if row['hr_abs_error_bpm'] is not None]
            if usable:
                oracle = min(usable, key=lambda row: row['hr_abs_error_bpm'])
                methods.append({**oracle, 'method': 'oracle_' + name + '_region_hr',
                                'reference_dependent_oracle': True})
            else:
                methods.append({**classical_rows[0], 'method': 'oracle_' + name + '_region_hr',
                                'region': None, 'reference_dependent_oracle': True})
        panels.append((info, traces, local_model,
                       data['candidates'][:, interval] if 'candidates' in data else None))
    return methods, regions, references, decisions, panels


def plot_panel(path, info, traces, region_rows, fs, min_hz, max_hz):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for name, wave in traces.items():
        try:
            filtered = preprocess_signal(wave, fs, min_hz, max_hz)
            freq, power = spectrum(wave, fs, min_hz, max_hz)
        except (TypeError, ValueError, OverflowError):
            continue  # Explicit unusable reasons remain in the CSV tables.
        scale = np.std(filtered)
        if scale > 1e-12:
            view = filtered / scale
            time = np.arange(min(len(view), int(fs * 10))) / fs
            axes[0, 0].plot(time, view[:len(time)], label=name, alpha=0.8)
        band = (freq >= min_hz) & (freq <= max_hz)
        peak = max(float(power[band].max()), 1e-12)
        axes[0, 1].plot(freq[band] * 60, power[band] / peak, label=name, alpha=0.8)
    axes[0, 0].set(xlabel='Time (s), first 10 seconds', ylabel='Z-score for display only')
    axes[0, 1].set(xlabel='Frequency (bpm)', ylabel='PSD / own peak, display only')
    axes[0, 1].legend(fontsize=7, ncol=2)
    if region_rows:
        labels = [str(row['region']) for row in region_rows]
        errors = [row['hr_abs_error_bpm'] if row['hr_abs_error_bpm'] is not None else np.nan
                  for row in region_rows]
        weights = [row['mean_weight'] for row in region_rows]
        axes[1, 0].bar(labels, errors)
        axes[1, 0].axhline(5, color='black', linestyle='--', linewidth=0.8)
        axes[1, 0].set(xlabel='Grid region (row-major)', ylabel='HR absolute error (bpm)')
        axes[1, 1].bar(labels, weights)
        axes[1, 1].set(xlabel='Grid region (row-major)', ylabel='Mean learned weight', ylim=(0, 1))
    else:
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
    fig.suptitle(f"{info['recording']} segment {info['segment']} window {info['window']}\n"
                 'Reference-based audit; no label correction or oracle inference')
    fig.tight_layout()
    fig.savefig(path, dpi=135)
    plt.close(fig)


def plot_regional_spectra(path, info, traces, region_rows, candidates, fs, min_hz, max_hz):
    """Show every fixed expert, including un-bandpassed second-harmonic power."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    columns = int(math.ceil(math.sqrt(len(candidates))))
    rows = int(math.ceil(len(candidates) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 2.9 * rows), squeeze=False)
    reference = analyze_signal(traces['reference'], fs=fs, min_hz=min_hz, max_hz=max_hz)
    f_ref = reference['hr_bpm'] / 60 if reference['hr_bpm'] is not None else None
    upper = min(fs / 2, 2 * max_hz + .2)
    for index, axis in enumerate(axes.flat):
        if index >= len(candidates):
            axis.axis('off')
            continue
        row = region_rows[index]
        try:
            freq, power = spectrum(candidates[index], fs, min_hz, max_hz, bandpass=False)
            band = (freq >= min_hz) & (freq <= upper)
            peak = max(float(power[band].max()), 1e-12)
            axis.plot(freq[band] * 60, power[band] / peak, color='tab:blue')
        except (TypeError, ValueError, OverflowError):
            axis.text(.5, .5, 'Unusable signal; see regional.csv', transform=axis.transAxes,
                      ha='center', fontsize=8)
        if f_ref is not None:
            axis.axvspan((f_ref - .1) * 60, (f_ref + .1) * 60, color='tab:green', alpha=.2)
            axis.axvspan((2 * f_ref - .1) * 60, (2 * f_ref + .1) * 60, color='tab:orange', alpha=.2)
        hr = f"{row['hr_bpm']:.1f}" if row['hr_bpm'] is not None else 'unusable'
        ratio = row['second_to_fundamental_ratio']
        ratio_text = f'{ratio:.2g}' if ratio is not None else 'unavailable'
        axis.set(title=f"Region {index}: HR {hr}, weight {row['mean_weight']:.3f}\n"
                       f'P(2 f_ref) / P(f_ref) = {ratio_text}',
                 xlabel='Frequency (bpm)', ylabel='PSD / own peak', xlim=(min_hz * 60, upper * 60))
        axis.tick_params(labelsize=8)
    ref_text = f'{f_ref * 60:.1f} bpm' if f_ref is not None else 'unusable reference'
    fig.suptitle(f"{info['recording']} segment {info['segment']} window {info['window']}\n"
                 f'Reference spectral peak: {ref_text}; green = f_ref, orange = 2 f_ref\n'
                 'Detrended without bandpass; normalization for display only; reference bands diagnostic only', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(path, dpi=135)
    plt.close(fig)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/train_configs/cross/MMPD_PURE_RegionalDINO.yaml')
    parser.add_argument('--checkpoint', help='RegionalDINO best.pth; omitted only with --classical-only.')
    parser.add_argument('--split', choices=['valid', 'train', 'test'], default='valid')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-recordings', type=int, default=0, help='0 means all; limited runs are smoke tests.')
    parser.add_argument('--window-seconds', type=float, default=0, help='0: whole contiguous segment; otherwise nonoverlapping windows.')
    parser.add_argument('--plot-recordings', type=int, default=6, help='Number of worst learned-fusion windows to plot.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--classical-only', action='store_true')
    parser.add_argument('--save-waveforms', action='store_true')
    parser.add_argument('--min-hz', type=float, default=0.75)
    parser.add_argument('--max-hz', type=float, default=2.5)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    if (args.batch_size < 1 or args.max_recordings < 0 or args.plot_recordings < 0
            or args.window_seconds < 0 or args.threads < 1):
        parser.error('Require positive batch-size/threads and nonnegative limits/window-seconds.')
    if not args.classical_only and not args.checkpoint:
        parser.error('--checkpoint is required unless --classical-only is used.')
    torch.set_num_threads(args.threads)
    config = get_config(argparse.Namespace(config_file=args.config))
    data_config = config[args.split.upper()].DATA
    if (list(data_config.PREPROCESS.DATA_TYPE) != ['Raw'] or data_config.DATA_FORMAT != 'NDCHW'
            or data_config.PREPROCESS.LABEL_TYPE != 'Standardized'):
        raise ValueError('Audit requires existing Raw RGB / Standardized labels / NDCHW cache configuration.')
    fs = float(data_config.FS)
    if not (0 < args.min_hz < args.max_hz < fs / 2):
        parser.error('Require 0 < min-hz < max-hz < sample-rate/2.')
    records = read_cache_records(config, args.split)
    train_records = None
    if args.split == 'valid':
        train_records = read_cache_records(config, 'train')
        assert_subject_disjoint(train_records, records)
        print('Verified training/validation subject lists do not overlap.')
    if not records:
        raise ValueError('No recording satisfies the split and configured filters.')
    total_records = len(records)
    if args.max_recordings:
        records = records[:args.max_recordings]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'manifest.json').exists():
        raise FileExistsError('Choose a new --output directory to preserve the previous audit.')
    architecture = model_arguments(config)
    model, checkpoint_epoch, checkpoint_training_seed, source_checkpoint = None, None, None, None
    if not args.classical_only:
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        validate_model_checkpoint(payload, config)
        model = build_model(config)
        model.load_state_dict(payload['model_state'], strict=True)
        model.to(args.device).eval()
        checkpoint_epoch = payload['epoch']
        checkpoint_training_seed = payload.get('random_seed')
        source_checkpoint = payload.get('source_checkpoint')
        print('Loaded checkpoint epoch:', checkpoint_epoch)
    manifest = {'command_arguments': vars(args), 'checkpoint_epoch': checkpoint_epoch,
                'checkpoint_training_seed': checkpoint_training_seed,
                'source_checkpoint': source_checkpoint,
                'checkpoint_sha256': file_hash(args.checkpoint) if model is not None else None,
                'file_list_sha256': file_hash(data_config.FILE_LIST_PATH),
                'train_file_list_sha256': (file_hash(config.TRAIN.DATA.FILE_LIST_PATH)
                                           if train_records is not None else None),
                'train_subjects': (sorted({r['subject'] for r in train_records})
                                   if train_records is not None else None),
                'audited_subjects': sorted({r['subject'] for r in records}),
                'model_name': config.MODEL.NAME,
                'model_arguments': architecture, 'resolved_config': config.dump(),
                'available_recordings': total_records, 'evaluated_recordings': len(records),
                'diagnostic_only': True, 'ground_truth_certified': False,
                'protocol': 'Tarvainen lambda100; Butterworth order1; Welch nfft3333; raw clip concatenation',
                'snr_definition': '10log10 power ratio, separate fundamental and two-harmonic scores',
                'oracle_warning': 'Reference-selected best region is not deployable and is not an accuracy claim.'}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    method_rows, region_rows, reference_rows, decision_rows, saved_panels, coverage_rows = [], [], [], [], [], []
    def panel_score(panel):
        value = analyze_signal(panel[1].get('learned', panel[1]['GREEN']), panel[1]['reference'],
                               fs=fs, min_hz=args.min_hz, max_hz=args.max_hz)['hr_abs_error_bpm']
        return value if value is not None else float('inf')
    for record in tqdm(records, desc=f'Audit {args.split}', ncols=95):
        for segment_index, clips in enumerate(contiguous_segments(record)):
            data = infer_segment(clips, model, architecture['grid_size'], args.batch_size, args.device)
            info = {'recording': record['recording'], 'subject': record['subject'],
                    'segment': segment_index, 'split': args.split,
                    'first_clip': clips[0][0], 'last_clip': clips[-1][0]}
            methods, regions, refs, decisions, panels = analyze_segment(
                data, info, fs, args.min_hz, args.max_hz, args.window_seconds)
            evaluated_frames = sum(row['frames'] for row in refs)
            coverage_rows.append({**info, 'available_frames': len(data['reference']),
                                  'evaluated_frames': evaluated_frames,
                                  'omitted_tail_frames': len(data['reference']) - evaluated_frames,
                                  'evaluated_windows': len(refs)})
            method_rows.extend(methods)
            region_rows.extend(regions)
            reference_rows.extend(refs)
            decision_rows.extend(decisions)
            if args.plot_recordings:
                saved_panels.extend(panels)
                saved_panels.sort(key=panel_score, reverse=True)
                saved_panels = saved_panels[:args.plot_recordings]
            if args.save_waveforms:
                directory = output_dir / 'waveforms'
                directory.mkdir(exist_ok=True)
                np.savez_compressed(directory / f"{record['recording']}_segment{segment_index}.npz", **data)
    failure_penalty = 60 * (args.max_hz - args.min_hz)
    method_summary = summarize(method_rows, failure_penalty)
    paired = [row for row in decision_rows if row['reference_hr_bpm'] is not None]
    paired_complete = [row for row in paired if row['learned_ae_bpm'] is not None
                       and row['uniform_ae_bpm'] is not None]
    paired_comparison = {
        'reference_usable_windows': len(paired), 'both_predictions_usable': len(paired_complete),
        'mean_learned_minus_uniform_ae_bpm_complete_pairs': finite_mean([
            row['learned_ae_bpm'] - row['uniform_ae_bpm'] for row in paired_complete]),
        'failure_penalty_bpm': failure_penalty,
        'mean_learned_minus_uniform_ae_bpm_with_failure_penalty': finite_mean([
            (row['learned_ae_bpm'] if row['learned_ae_bpm'] is not None else failure_penalty)
            - (row['uniform_ae_bpm'] if row['uniform_ae_bpm'] is not None else failure_penalty)
            for row in paired]),
    }
    summary = {'methods': method_summary,
               'paired_fusion': paired_comparison,
               'candidate_correct_but_fusion_wrong_windows': sum(
                   row['candidate_correct_but_fusion_wrong'] for row in decision_rows),
               'no_candidate_within_5_bpm_windows': sum(row['no_candidate_within_5_bpm'] is True
                                                       for row in decision_rows),
               'evaluated_model_windows': len(decision_rows),
               'model_windows_with_usable_reference': len(paired),
               'available_frames': sum(row['available_frames'] for row in coverage_rows),
               'evaluated_frames': sum(row['evaluated_frames'] for row in coverage_rows),
               'omitted_tail_frames': sum(row['omitted_tail_frames'] for row in coverage_rows),
               'evaluated_recordings': len(records), 'limited_smoke_run': len(records) < total_records,
               'warning': 'Agreement with video-derived methods does not prove reference correctness. Oracle scores use reference.'}
    write_csv(output_dir / 'methods.csv', [{'method': name, **values} for name, values in method_summary.items()])
    write_csv(output_dir / 'records.csv', method_rows)
    write_csv(output_dir / 'regional.csv', region_rows)
    write_csv(output_dir / 'reference_checks.csv', reference_rows)
    write_csv(output_dir / 'fusion_decisions.csv', decision_rows)
    write_csv(output_dir / 'coverage.csv', coverage_rows)
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    if saved_panels:
        directory = output_dir / 'plots'
        directory.mkdir(exist_ok=True)
        for index, (info, traces, local, candidates) in enumerate(saved_panels):
            plot_panel(directory / f'{index + 1:02d}_audit.png', info, traces, local,
                       fs, args.min_hz, args.max_hz)
            if candidates is not None:
                plot_regional_spectra(directory / f'{index + 1:02d}_regional_spectra.png',
                                      info, traces, local, candidates, fs, args.min_hz, args.max_hz)
    print(json.dumps(summary, indent=2, allow_nan=False))
    print('Saved audit:', output_dir.resolve())
    return summary


if __name__ == '__main__':
    main()
