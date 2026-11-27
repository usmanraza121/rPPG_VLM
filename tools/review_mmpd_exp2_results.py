"""Matched Exp2 MMPD source-validation review, without training or data changes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.regional_signal_diagnostics import detrend_signal, preprocess_signal


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def peak_hr(values):
    f, p = signal.welch(values, fs=30, window='hann', nperseg=256,
                         noverlap=128, nfft=3333, detrend=False, axis=-1)
    band = (f >= .75) & (f <= 2.5)
    return f[band][p[..., band].argmax(-1)] * 60


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=ROOT.parent)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'docs/experiments/results/exp2_mmpd_20261126')
    args = parser.parse_args()
    workspace, out = args.workspace, args.output
    out.mkdir(parents=True, exist_ok=True)
    folders = {'baseline': workspace / 'mmpd_no_dino_full',
               'exp1': workspace / 'mmpd_exp1_rate_aug_full',
               'exp2': workspace / 'Exp_data/exp2/mmpd_exp2_attention_full'}
    manifests = {k: json.loads((p / 'manifest.json').read_text()) for k, p in folders.items()}
    summaries = {k: json.loads((p / 'summary.json').read_text()) for k, p in folders.items()}
    frames = {k: pd.read_csv(p / 'records.csv') for k, p in folders.items()}
    keys = ['recording', 'segment', 'window', 'method']
    checks = {}
    for run in ['exp1', 'exp2']:
        for key in ['file_list_sha256', 'train_file_list_sha256', 'train_subjects',
                    'audited_subjects', 'protocol']:
            checks[f'{run}_{key}'] = manifests[run][key] == manifests['baseline'][key]
        for file in ['coverage.csv', 'reference_checks.csv']:
            checks[f'{run}_{file}_bytes'] = (folders[run] / file).read_bytes() == (folders['baseline'] / file).read_bytes()
        paired = frames['baseline'].merge(frames[run], on=keys,
                                          suffixes=('_base', '_new'), validate='one_to_one')
        assert len(paired) == len(frames[run]) == len(frames['baseline'])
        for col in ['reference_hr_bpm', 'reference_std', 'frames', 'sample_count']:
            checks[f'{run}_{col}'] = bool(np.allclose(paired[col + '_base'], paired[col + '_new'], equal_nan=True))
        classical = paired[paired.method.str.contains('GREEN|POS|CHROM')]
        for col in ['hr_bpm', 'hr_abs_error_bpm', 'waveform_corr']:
            checks[f'{run}_classical_{col}'] = bool(np.allclose(classical[col + '_base'], classical[col + '_new'], equal_nan=True))
    assert all(checks.values())
    learned = {}
    metrics, bins, subjects, candidates = [], [], [], {}
    for run, df in frames.items():
        p = df[df.method == 'learned'].copy().set_index('recording')
        assert len(p) == 137 and p.index.is_unique
        f = pd.read_csv(folders[run] / 'fusion_decisions.csv').set_index('recording')
        p['candidate_correct'] = ~f.no_candidate_within_5_bpm
        p['signed_error_bpm'] = p.hr_bpm - p.reference_hr_bpm
        p['hr_bin'] = pd.cut(p.reference_hr_bpm, [0, 60, 80, 100, np.inf], right=False,
                             labels=['<60', '60-80', '80-100', '>=100'])
        learned[run] = p
        point = summaries[run]['methods']['learned']
        assert abs(p.hr_abs_error_bpm.mean() - point['mae_bpm']) < 1e-10
        assert (p.hr_abs_error_bpm <= 5).sum() == point['within_5_bpm']
        metrics.append({'run': run, **point, 'within_10_bpm': int((p.hr_abs_error_bpm <= 10).sum()),
                        'over_10_bpm': int((p.hr_abs_error_bpm > 10).sum()),
                        'over_20_bpm': int((p.hr_abs_error_bpm > 20).sum()),
                        'bias_bpm': float(p.signed_error_bpm.mean()),
                        'uniform_mae_bpm': summaries[run]['methods']['uniform']['mae_bpm'],
                        'oracle_region_mae_bpm_diagnostic': summaries[run]['methods']['oracle_region_hr']['mae_bpm']})
        for label, g in p.groupby('hr_bin', observed=True):
            bins.append({'run': run, 'hr_bin': str(label), 'n': len(g),
                         'mae_bpm': float(g.hr_abs_error_bpm.mean()),
                         'bias_bpm': float(g.signed_error_bpm.mean()),
                         'correct_candidate': int(g.candidate_correct.sum())})
        for subject, g in p.groupby('subject'):
            subjects.append({'run': run, 'subject': subject, 'n': len(g),
                             'mae_bpm': float(g.hr_abs_error_bpm.mean()),
                             'waveform_corr': float(g.waveform_corr.mean())})
        rg = pd.read_csv(folders[run] / 'regional.csv')
        rg = rg[rg.method == 'regional_candidate']
        missing = p.index[~p.candidate_correct]
        near = set()
        for row in rg[rg.recording.isin(missing)].itertuples():
            if any(abs(z['hr_bpm'] - row.reference_hr_bpm) <= 5 for z in json.loads(row.dominant_peaks)):
                near.add(row.recording)
        big = p.hr_abs_error_bpm > 10
        candidates[run] = {'missing_correct_processed_candidate': len(missing),
                           'correct_processed_candidate': int(p.candidate_correct.sum()),
                           'candidate_present_error_over10': int((big & p.candidate_correct).sum()),
                           'candidate_missing_error_over10': int((big & ~p.candidate_correct).sum()),
                           'missing_with_reference_near_top3_diagnostic': len(near),
                           'regional_second_to_fundamental_power_over1': int((rg.second_to_fundamental_ratio > 1).sum())}
    matched = learned['baseline'][['subject', 'reference_hr_bpm', 'hr_bpm', 'hr_abs_error_bpm']].add_suffix('_baseline')
    for run in ['exp1', 'exp2']:
        matched = matched.join(learned[run][['hr_bpm', 'hr_abs_error_bpm']].add_suffix('_' + run), validate='one_to_one')
    pair_counts = {}
    for run in ['baseline', 'exp1']:
        delta = matched.hr_abs_error_bpm_exp2 - matched['hr_abs_error_bpm_' + run]
        pair_counts[run] = {'exp2_better': int((delta < -1e-6).sum()),
                            'exp2_worse': int((delta > 1e-6).sum()),
                            'ties': int((abs(delta) <= 1e-6).sum())}
    stage_rows, exact_references = [], 0
    for file in sorted((folders['exp2'] / 'waveforms').glob('*.npz')):
        rec = file.name.rsplit('_segment', 1)[0]
        with np.load(file) as x:
            reference = x['reference'].astype(float)
            values = np.vstack((x['learned'][None], x['candidates'])).astype(float)
            assert len(reference) == 1760 and values.shape == (10, 1760)
        with np.load(folders['exp1'] / 'waveforms' / file.name) as old:
            assert np.array_equal(reference, old['reference'])
        exact_references += 1
        h = float(learned['exp2'].loc[rec, 'reference_hr_bpm'])
        for stage in ['raw', 'detrended', 'filtered']:
            if stage == 'raw':
                data = values - values.mean(-1, keepdims=True)
            elif stage == 'detrended':
                data = np.stack([detrend_signal(v) for v in values])
            else:
                data = np.stack([preprocess_signal(v) for v in values])
            hrs = peak_hr(data)
            if stage == 'filtered':
                assert abs(hrs[0] - learned['exp2'].loc[rec, 'hr_bpm']) < 1e-8
            stage_rows.append({'recording': rec, 'stage': stage, 'reference_hr_bpm': h,
                               'learned_hr_bpm': float(hrs[0]), 'error_bpm': float(abs(hrs[0] - h)),
                               'correct_candidate': bool((abs(hrs[1:] - h) <= 5).any())})
    assert exact_references == 137
    stages = pd.DataFrame(stage_rows)
    stage_stats = {stage: {'mae_bpm': float(g.error_bpm.mean()),
                           'within5': int((g.error_bpm <= 5).sum()),
                           'within10': int((g.error_bpm <= 10).sum()),
                           'correct_candidate': int(g.correct_candidate.sum())}
                   for stage, g in stages.groupby('stage')}
    matched.to_csv(out / 'matched_recordings.csv')
    pd.DataFrame(metrics).to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(bins).to_csv(out / 'hr_bins.csv', index=False)
    pd.DataFrame(subjects).to_csv(out / 'subject_metrics.csv', index=False)
    stages.to_csv(out / 'processing_stages.csv', index=False)
    summary = {'review_date': '2026-11-26', 'matching_checks': checks,
               'recordings': 137, 'subjects': 7, 'frames': 241120,
               'exact_exp1_exp2_reference_recordings': exact_references,
               'exp2_checkpoint_manifest': {k: manifests['exp2'][k] for k in ['checkpoint_epoch', 'checkpoint_sha256', 'model_arguments']},
               'metrics': metrics, 'hr_bins': bins, 'subject_metrics': subjects,
               'candidate_diagnostics': candidates, 'paired_recordings_vs': pair_counts,
               'exp2_processing_sensitivity': stage_stats,
               'source_sha256': {run: {name: sha(p / name) for name in ['manifest.json', 'summary.json', 'records.csv', 'regional.csv', 'coverage.csv', 'reference_checks.csv']}
                                 for run, p in folders.items()},
               'limits': ['One run per setting; attention effect not isolated.',
                          'MMPD validation selected the checkpoints, so this is not independent source-test performance.',
                          'Threshold counts describe full-record HR errors, not confidence intervals or waveform fidelity.',
                          'Raw/detrended changes and reference-selected regional diagnostics do not replace primary scores.',
                          'Manifest checkpoint hashes were not rehashed against local checkpoint binaries.']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    bins_frame = pd.DataFrame(bins)
    x = np.arange(3)
    for index, (run, color) in enumerate([('baseline', '#777777'), ('exp1', '#c67b27'), ('exp2', '#3269a8')]):
        g = bins_frame[bins_frame.run == run]
        ax[0].bar(x + (index - 1) * .26, g.mae_bpm, .26, label=run, color=color)
    ax[0].set_xticks(x, ['60–80 (n=37)', '80–100 (n=79)', '≥100 (n=21)'])
    ax[0].set_ylabel('MMPD recording MAE (bpm)')
    ax[0].set_xlabel('Reference heart rate')
    ax[0].set_title('High-rate errors decrease; mid-range worsens')
    ax[0].legend(frameon=False)
    errors = np.sort(learned['exp2'].hr_abs_error_bpm.to_numpy())
    ax[1].step(errors, np.arange(1, 138) / 137 * 100, where='post', color='#3269a8')
    for threshold in [5, 10]:
        count = int((errors <= threshold).sum())
        ax[1].axvline(threshold, ls='--', lw=.9, color='#777777')
        ax[1].scatter([threshold], [100 * count / 137], color='#c67b27')
        ax[1].annotate(f'{count}/137 ({100 * count / 137:.1f}%)',
                       (threshold, 100 * count / 137), xytext=(8, -16 if threshold == 5 else 8),
                       textcoords='offset points', fontsize=9)
    ax[1].set_xlabel('Allowed absolute HR error (bpm)')
    ax[1].set_ylabel('Recordings within the tolerance (%)')
    ax[1].set_title('Exp2 tolerance coverage')
    ax[1].set_ylim(0, 103)
    for a in ax:
        a.grid(axis='y', alpha=.2)
        a.set_axisbelow(True)
    fig.savefig(out / 'rate_comparison_and_tolerances.png', dpi=170)
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ['metrics', 'hr_bins', 'subject_metrics',
                                            'candidate_diagnostics', 'paired_recordings_vs',
                                            'exp2_processing_sensitivity']}, indent=2))
    return summary


if __name__ == '__main__':
    main()
