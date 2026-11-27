"""Reproduce the matched Exp3 MMPD audit review from supplied artifacts.

Run from rPPG_VLM: python tools/review_mmpd_exp3_results.py
No model, dataset, labels or primary evaluator are changed. GT-selected regional
results and prediction processing sensitivity are diagnostics only.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.review_mmpd_exp2_results import peak_hr, sha
from tools.regional_signal_diagnostics import detrend_signal, preprocess_signal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=ROOT.parent)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'docs/experiments/results/exp3_mmpd_20261127')
    args = parser.parse_args()
    workspace, out = args.workspace, args.output
    out.mkdir(parents=True, exist_ok=True)
    folders = dict(baseline=workspace / 'mmpd_no_dino_full',
                   exp1=workspace / 'mmpd_exp1_rate_aug_full',
                   exp2=workspace / 'Exp_data/exp2/mmpd_exp2_attention_full',
                   exp3=workspace / 'Exp_data/exp3/mmpd_exp3_tcn_full')
    manifests = {run: json.loads((p / 'manifest.json').read_text())
                 for run, p in folders.items()}
    summaries = {run: json.loads((p / 'summary.json').read_text())
                for run, p in folders.items()}
    tables = {run: pd.read_csv(p / 'records.csv') for run, p in folders.items()}
    checks = {}
    keys = ['recording', 'segment', 'window', 'method']
    for run in ['baseline', 'exp1', 'exp2']:
        for key in ['file_list_sha256', 'train_file_list_sha256', 'train_subjects',
                    'audited_subjects', 'protocol']:
            checks[f'{run}_{key}'] = manifests[run][key] == manifests['exp3'][key]
        for key in ['split', 'window_seconds', 'max_recordings', 'min_hz', 'max_hz']:
            checks[f'{run}_argument_{key}'] = (
                manifests[run]['command_arguments'][key]
                == manifests['exp3']['command_arguments'][key])
        for name in ['coverage.csv', 'reference_checks.csv']:
            checks[f'{run}_{name}_bytes'] = ((folders[run] / name).read_bytes()
                                            == (folders['exp3'] / name).read_bytes())
        pairs = tables[run].merge(tables['exp3'], on=keys, how='outer',
                                  suffixes=('_old', '_new'), validate='one_to_one', indicator=True)
        checks[f'{run}_all_record_method_keys'] = bool((pairs._merge == 'both').all())
        for col in ['reference_hr_bpm', 'reference_std', 'frames', 'sample_count']:
            checks[f'{run}_{col}'] = bool(np.allclose(
                pairs[col + '_old'], pairs[col + '_new'], equal_nan=True))
        classical = pairs[pairs.method.str.contains('GREEN|POS|CHROM')]
        for col in ['hr_bpm', 'hr_abs_error_bpm', 'waveform_corr']:
            checks[f'{run}_classical_{col}'] = bool(np.allclose(
                classical[col + '_old'], classical[col + '_new'], equal_nan=True))
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}
    assert manifests['exp3']['model_arguments']['temporal_backend'] == 'tcn'
    assert manifests['exp3']['checkpoint_epoch'] == 38
    learned, metric_rows, bins, subjects, candidates = {}, [], [], [], {}
    for run, df in tables.items():
        p = df[df.method == 'learned'].copy().set_index('recording')
        assert len(p) == 137 and p.index.is_unique
        assert (p.status == 'ok').all() and (p.reference_status == 'ok').all()
        assert (p.segment == 0).all() and (p.window == 0).all()
        assert int(p.frames.sum()) == 241120
        decisions = pd.read_csv(folders[run] / 'fusion_decisions.csv').set_index('recording')
        p['candidate_correct'] = ~decisions.no_candidate_within_5_bpm
        p['signed_error_bpm'] = p.hr_bpm - p.reference_hr_bpm
        p['hr_bin'] = pd.cut(p.reference_hr_bpm, [0, 60, 80, 100, np.inf],
                             right=False, labels=['<60', '60-80', '80-100', '>=100'])
        learned[run] = p
        point = summaries[run]['methods']['learned']
        assert abs(p.hr_abs_error_bpm.mean() - point['mae_bpm']) < 1e-10
        assert (p.hr_abs_error_bpm <= 5).sum() == point['within_5_bpm']
        metric_rows.append(dict(run=run, **point,
                               within_10_bpm=int((p.hr_abs_error_bpm <= 10).sum()),
                               over_10_bpm=int((p.hr_abs_error_bpm > 10).sum()),
                               over_20_bpm=int((p.hr_abs_error_bpm > 20).sum()),
                               bias_bpm=float(p.signed_error_bpm.mean()),
                               predicted_min_bpm=float(p.hr_bpm.min()),
                               predicted_max_bpm=float(p.hr_bpm.max()),
                               uniform_mae_bpm=summaries[run]['methods']['uniform']['mae_bpm'],
                               oracle_region_mae_bpm_diagnostic=
                               summaries[run]['methods']['oracle_region_hr']['mae_bpm']))
        for label, g in p.groupby('hr_bin', observed=True):
            bins.append(dict(run=run, hr_bin=str(label), n=len(g),
                             mae_bpm=float(g.hr_abs_error_bpm.mean()),
                             bias_bpm=float(g.signed_error_bpm.mean()),
                             waveform_corr=float(g.waveform_corr.mean()),
                             correct_candidate=int(g.candidate_correct.sum())))
        for subject, g in p.groupby('subject'):
            subjects.append(dict(run=run, subject=subject, n=len(g),
                                 mae_bpm=float(g.hr_abs_error_bpm.mean()),
                                 waveform_corr=float(g.waveform_corr.mean())))
        rg = pd.read_csv(folders[run] / 'regional.csv')
        rg = rg[rg.method == 'regional_candidate']
        assert len(rg) == 137 * 9
        oracle = rg.groupby('recording').hr_abs_error_bpm.min()
        assert np.array_equal(p.candidate_correct, (oracle <= 5).reindex(p.index))
        missing, near = p.index[~p.candidate_correct], set()
        for row in rg[rg.recording.isin(missing)].itertuples():
            if any(abs(z['hr_bpm'] - row.reference_hr_bpm) <= 5
                   for z in json.loads(row.dominant_peaks)):
                near.add(row.recording)
        big = p.hr_abs_error_bpm > 10
        candidates[run] = dict(
            correct_processed_candidate=int(p.candidate_correct.sum()),
            missing_correct_processed_candidate=len(missing),
            candidate_present_error_over10=int((big & p.candidate_correct).sum()),
            candidate_missing_error_over10=int((big & ~p.candidate_correct).sum()),
            missing_with_reference_near_top3_diagnostic=len(near),
            regional_second_to_fundamental_power_over1=int((rg.second_to_fundamental_ratio > 1).sum()))
    joined = learned['exp3'][['subject', 'reference_hr_bpm', 'hr_bpm', 'hr_abs_error_bpm',
                              'waveform_corr', 'candidate_correct']].add_suffix('_exp3')
    pair_stats = {}
    for run in ['baseline', 'exp1', 'exp2']:
        p = learned[run]
        joined = joined.join(p[['hr_bpm', 'hr_abs_error_bpm', 'waveform_corr',
                                'candidate_correct']].add_suffix('_' + run), validate='one_to_one')
        delta = learned['exp3'].hr_abs_error_bpm - p.hr_abs_error_bpm
        subject_mae = pd.DataFrame(subjects).pivot(index='subject', columns='run', values='mae_bpm')
        pair_stats[run] = dict(
            exp3_better=int((delta < -1e-6).sum()),
            exp3_worse=int((delta > 1e-6).sum()), ties=int((abs(delta) <= 1e-6).sum()),
            exp3_subjects_better=int((subject_mae.exp3 < subject_mae[run]).sum()),
            exp3_subjects_worse=int((subject_mae.exp3 > subject_mae[run]).sum()),
            exp3_lost_correct_candidate=int((p.candidate_correct & ~learned['exp3'].candidate_correct).sum()),
            exp3_gained_correct_candidate=int((~p.candidate_correct & learned['exp3'].candidate_correct).sum()))
    stage_rows, exact_arrays = [], 0
    files = sorted((folders['exp3'] / 'waveforms').glob('*.npz'))
    assert len(files) == 137
    for file in files:
        rec = file.name.rsplit('_segment', 1)[0]
        with np.load(file) as x, np.load(folders['exp2'] / 'waveforms' / file.name) as old:
            for key in ['reference', 'rgb_global', 'rgb_regions']:
                assert np.array_equal(x[key], old[key])
            assert int(x['clip_length']) == 160 and int(x['clip_count']) == 11
            reference = x['reference'].astype(float)
            values = np.vstack((x['learned'][None], x['candidates'])).astype(float)
            assert len(reference) == 1760 and values.shape == (10, 1760)
            assert np.isfinite(values).all()
            assert np.allclose(x['weights'].sum(0), 1, atol=1e-6)
            # Audit fusion_controls centers each exported candidate per clip;
            # the model itself applies one shared fused DC correction instead.
            # Restore that common DC convention before comparing weighted sums.
            weighted = (x['weights'] * x['candidates']).sum(0).reshape(-1, 160)
            weighted = weighted - weighted.mean(-1, keepdims=True)
            assert np.allclose(weighted.reshape(-1), x['learned'], atol=1e-6)
            assert np.allclose(x['candidates'].mean(0), x['uniform'], atol=1e-6)
        exact_arrays += 1
        h = float(learned['exp3'].loc[rec, 'reference_hr_bpm'])
        assert abs(float(peak_hr(preprocess_signal(reference))) - h) < 1e-8
        for stage in ['raw', 'detrended', 'filtered']:
            data = (values - values.mean(-1, keepdims=True) if stage == 'raw' else
                    np.stack([(detrend_signal(v) if stage == 'detrended'
                               else preprocess_signal(v)) for v in values]))
            hrs = peak_hr(data)
            if stage == 'filtered':
                assert abs(hrs[0] - learned['exp3'].loc[rec, 'hr_bpm']) < 1e-8
                assert bool((abs(hrs[1:] - h) <= 5).any()) == bool(learned['exp3'].loc[rec, 'candidate_correct'])
            stage_rows.append(dict(recording=rec, stage=stage, reference_hr_bpm=h,
                                   learned_hr_bpm=float(hrs[0]), error_bpm=float(abs(hrs[0] - h)),
                                   correct_candidate=bool((abs(hrs[1:] - h) <= 5).any())))
        if exact_arrays % 40 == 0:
            print(f'Checked {exact_arrays}/137 saved source recordings', flush=True)
    stages = pd.DataFrame(stage_rows)
    stage_stats = {stage: dict(mae_bpm=float(g.error_bpm.mean()),
                              within5=int((g.error_bpm <= 5).sum()),
                              within10=int((g.error_bpm <= 10).sum()),
                              correct_candidate=int(g.correct_candidate.sum()))
                   for stage, g in stages.groupby('stage')}
    summary = dict(
        review_date='2026-11-27', matching_checks=checks,
        recordings=137, subjects=7, frames=241120,
        exact_exp2_exp3_reference_and_rgb_recordings=exact_arrays,
        checkpoint_manifest={k: manifests['exp3'][k] for k in
                             ['checkpoint_epoch', 'checkpoint_sha256', 'model_name', 'model_arguments']},
        metrics=metric_rows, hr_bins=bins, subject_metrics=subjects,
        candidate_diagnostics=candidates, paired_recordings_vs=pair_stats,
        exp3_processing_sensitivity=stage_stats,
        source_sha256={run: {name: sha(p / name) for name in
                            ['manifest.json', 'summary.json', 'records.csv', 'regional.csv',
                             'coverage.csv', 'reference_checks.csv', 'fusion_decisions.csv']}
                       for run, p in folders.items()},
        limits=[
            'One run per setting; temporal operator, context and optimization differ.',
            'MMPD validation selected checkpoints; this is not independent source-test performance.',
            '137 recordings come from seven subjects and are not independent replication counts.',
            'Source waveform r/SNR improve while dominant-HR error worsens; no blanket waveform superiority.',
            'GT-selected regions and prediction processing sensitivity are diagnostic only.',
            'Reported checkpoint hash is from the manifest, not a local checkpoint binary rehash.',
        ])
    for name, table in [('matched_recordings', joined), ('metrics', pd.DataFrame(metric_rows)),
                         ('hr_bins', pd.DataFrame(bins)), ('subject_metrics', pd.DataFrame(subjects)),
                         ('processing_stages', stages)]:
        table.to_csv(out / (name + '.csv'), index=name == 'matched_recordings')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout='constrained')
    bins_df, subjects_df = pd.DataFrame(bins), pd.DataFrame(subjects)
    colors = dict(baseline='#888888', exp2='#3269a8', exp3='#c66532')
    x = np.arange(3)
    for j, run in enumerate(colors):
        g = bins_df[bins_df.run == run]
        axes[0].bar(x + (j - 1) * .25, g.mae_bpm, .25, color=colors[run], label=run)
    axes[0].set_xticks(x, ['60–80 (n=37)', '80–100 (n=79)', '≥100 (n=21)'])
    axes[0].set(xlabel='Reference HR (bpm)', ylabel='MMPD recording MAE (bpm)',
                title='High-rate regression in Exp3')
    x = np.arange(7)
    for j, run in enumerate(['exp2', 'exp3']):
        g = subjects_df[subjects_df.run == run].sort_values('subject')
        axes[1].bar(x + (j - .5) * .35, g.mae_bpm, .35, color=colors[run], label=run)
    axes[1].set_xticks(x, [str(i) for i in range(27, 34)])
    axes[1].set(xlabel='MMPD validation subject', ylabel='Recording MAE (bpm)',
                title='Variation across seven held-out subjects')
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
    fig.savefig(out / 'source_rate_and_subject_comparison.png', dpi=170)
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ['metrics', 'hr_bins', 'subject_metrics',
                                            'candidate_diagnostics', 'paired_recordings_vs',
                                            'exp3_processing_sensitivity']}, indent=2))
    return summary


if __name__ == '__main__':
    main()
