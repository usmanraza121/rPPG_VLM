"""Review the supplied Exp3 TCN control using the fixed earlier evaluator.

Run from rPPG_VLM: python tools/review_exp3_results.py
Reads supplied artifacts only; writes derived tables/figures to docs/experiments.
Reference-selected regions and preprocessing sensitivity are diagnostics only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.review_exp2_results import (
    differences, parse_log, reconstruct, sha, stage_hrs,
)
from tools.regional_signal_diagnostics import analyze_signal, detrend_signal


def history(path):
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    assert [r['epoch'] for r in rows] == list(range(40))
    selected = min((r for r in rows if r['checkpoint_eligible']),
                   key=lambda r: r['selection_score'])
    assert all(r['selection_metric'] == 'loss' for r in rows)
    return rows, selected


def metrics(reference, predicted):
    error = predicted - reference
    ae = abs(error)
    return dict(MAE=float(ae.mean()), RMSE=float(np.sqrt((error ** 2).mean())),
                Pearson=float(np.corrcoef(reference, predicted)[0, 1]),
                within5=int((ae <= 5).sum()), within10=int((ae <= 10).sum()),
                over10=int((ae > 10).sum()), over20=int((ae > 20).sum()),
                bias_bpm=float(error.mean()), minimum_hr=float(predicted.min()),
                maximum_hr=float(predicted.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=ROOT.parent)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'docs/experiments/results/exp3_20261127')
    args = parser.parse_args()
    workspace, out = args.workspace, args.output
    out.mkdir(parents=True, exist_ok=True)
    paths = {'baseline': workspace / 'training.log', **{
        run: workspace / 'Exp_data' / run / 'training.log'
        for run in ['exp1', 'exp2', 'exp3']}}
    runs = {run: parse_log(p) for run, p in paths.items()}
    histories = {run: history(workspace / 'Exp_data' / run / 'regional_history.jsonl')
                 for run in ['exp2', 'exp3']}
    for run, r in runs.items():
        assert len(r['reference_hr']) == 59
        assert np.array_equal(r['reference_hr'], runs['exp3']['reference_hr'])
        assert abs(metrics(r['reference_hr'], r['predicted_hr'])['MAE']
                   - r['metrics']['MAE']) < 1e-7
    payloads = {run: torch.load(workspace / 'Exp_data' / run / 'regional_waveforms.pt',
                               map_location='cpu', weights_only=True)
                for run in ['exp2', 'exp3']}
    exp2, exp3 = payloads['exp2'], payloads['exp3']
    assert exp3['model_name'] == 'RegionalSTAttention'
    assert exp3['model_arguments']['temporal_backend'] == 'tcn'
    assert exp3['fs'] == 30 and exp3['grid_size'] == 3
    assert set(exp2['gt']) == set(exp3['gt']) and len(exp3['gt']) == 59
    records, regional, stages = [], [], []
    clips, frames = 0, 0
    for index, rec in enumerate(exp3['pred_raw']):
        assert set(exp3['gt'][rec]) == set(exp2['gt'][rec])
        for clip, gt in exp3['gt'][rec].items():
            assert torch.equal(gt, exp2['gt'][rec][clip])
        data = reconstruct(exp3, rec)
        clips += data['clips']
        frames += len(data['reference'])
        subject = f'subject{int(rec) // 100:02d}'
        for method in ['learned', 'uniform']:
            row = analyze_signal(data[method], data['reference'], fs=30,
                                 max_lag_seconds=0)
            assert row['status'] == row['reference_status'] == 'ok'
            records.append(dict(recording=rec, subject=subject, method=method,
                                clips=data['clips'], **row))
        learned = records[-2]
        assert abs(learned['reference_hr_bpm'] - runs['exp3']['reference_hr'][index]) < 1e-7
        assert abs(learned['hr_bpm'] - runs['exp3']['predicted_hr'][index]) < 1e-7
        regional_hrs = []
        for region in range(9):
            row = analyze_signal(data['regions'][region], data['reference'], fs=30,
                                 max_lag_seconds=0)
            assert row['status'] == row['reference_status'] == 'ok'
            regional.append(dict(recording=rec, subject=subject, region=region,
                                 mean_weight=float(data['weights'][region].mean()), **row))
            regional_hrs.append(row['hr_bpm'])
        h = learned['reference_hr_bpm']
        for stage in ['raw', 'detrended', 'filtered']:
            if stage == 'filtered':
                hrs = np.array([learned['hr_bpm'], *regional_hrs])
            else:
                values = np.vstack((data['learned'][None], data['regions']))
                values = (values - values.mean(-1, keepdims=True) if stage == 'raw'
                          else np.stack([detrend_signal(v) for v in values]))
                hrs = stage_hrs(values)
            stages.append(dict(recording=rec, subject=subject, stage=stage,
                               reference_hr_bpm=h, learned_hr_bpm=float(hrs[0]),
                               learned_ae_bpm=float(abs(hrs[0] - h)),
                               candidate_correct=bool((abs(hrs[1:] - h) <= 5).any())))
        if (index + 1) % 10 == 0:
            print(f'Analyzed {index + 1}/59 recordings', flush=True)
    assert clips == 750
    df, rg, stage_df = pd.DataFrame(records), pd.DataFrame(regional), pd.DataFrame(stages)
    learned = df[df.method == 'learned'].copy()
    uniform = df[df.method == 'uniform']
    assert abs(learned.hr_abs_error_bpm.mean() - runs['exp3']['metrics']['MAE']) < 1e-8
    oracle = rg.groupby('recording').hr_abs_error_bpm.min()
    candidate = oracle <= 5
    learned['candidate_correct'] = learned.recording.map(candidate)
    learned['hr_bin'] = pd.cut(learned.reference_hr_bpm, [0, 60, 80, 100, np.inf],
                               right=False, labels=['<60', '60-80', '80-100', '>=100'])
    bins, subjects = [], []
    for label, g in learned.groupby('hr_bin', observed=True):
        ids = learned.recording.isin(g.recording).to_numpy()
        bins.append({'hr_bin': str(label), 'n': len(g), **{
            run + '_mae': float(abs(r['predicted_hr'] - r['reference_hr'])[ids].mean())
            for run, r in runs.items()}, 'exp3_candidates': int(g.candidate_correct.sum())})
    for subject, g in learned.groupby('subject'):
        ids = learned.recording.isin(g.recording).to_numpy()
        subjects.append({'subject': subject, 'n': len(g), **{
            run + '_mae': float(abs(r['predicted_hr'] - r['reference_hr'])[ids].mean())
            for run, r in runs.items()}, 'exp3_candidates': int(g.candidate_correct.sum()),
                         'exp3_waveform_corr': float(g.waveform_corr.mean())})
    previous_out = ROOT / 'docs/experiments/results/exp2_20261126'
    previous = pd.read_csv(previous_out / 'model_recordings.csv', dtype={'recording': str})
    previous = previous[previous.method == 'learned'].set_index('recording')
    current = learned.set_index('recording')
    columns = ['reference_hr_bpm', 'hr_bpm', 'hr_abs_error_bpm', 'waveform_corr']
    pairs = previous[columns].merge(current[columns], left_index=True, right_index=True,
                                    suffixes=('_exp2', '_exp3'), validate='one_to_one')
    assert len(pairs) == 59 and np.array_equal(pairs.reference_hr_bpm_exp2, pairs.reference_hr_bpm_exp3)
    pairs['error_delta_bpm'] = pairs.hr_abs_error_bpm_exp3 - pairs.hr_abs_error_bpm_exp2
    old_regions = pd.read_csv(previous_out / 'regional_candidates.csv', dtype={'recording': str})
    old_candidate = old_regions.groupby('recording').hr_abs_error_bpm.min() <= 5
    lost = old_candidate & ~candidate
    gained = ~old_candidate & candidate
    big = learned.hr_abs_error_bpm > 10
    filtered_errors = stage_df.pivot(index='recording', columns='stage', values='learned_ae_bpm')
    stage_stats = {stage: dict(mae_bpm=float(g.learned_ae_bpm.mean()),
                              within5=int((g.learned_ae_bpm <= 5).sum()),
                              candidate_correct=int(g.candidate_correct.sum()))
                   for stage, g in stage_df.groupby('stage')}
    log_text = paths['exp3'].read_text(encoding='utf-8', errors='replace')
    checkpoint_lines = [s.strip() for s in log_text.splitlines()
                        if 'best.pth' in s or 'selected checkpoint' in s.lower()]
    history_rows = []
    for run, (rows, _) in histories.items():
        for row in rows:
            for split in ['train', 'valid']:
                history_rows.append(dict(run=run, epoch=row['epoch'], split=split,
                                         selection_score=row['selection_score'],
                                         learning_rate=row['learning_rates'][0], **row[split]))
    summary = dict(
        review_date='2026-11-27',
        source_hashes={p.name: sha(p) for p in (workspace / 'Exp_data/exp3').iterdir() if p.is_file()},
        metadata={k: v for k, v in exp3.items() if k not in ['gt', 'pred_raw', 'regional']},
        resolved_config_differences_exp2_to_exp3=[
            dict(key=d['key'], exp2=d['baseline'], exp3=d['exp2'])
            for d in differences(runs['exp2']['config'], runs['exp3']['config'])],
        checkpoint_log_lines=checkpoint_lines, selected_epoch_zero_based=histories['exp3'][1]['epoch'],
        epochs=40, recordings=59, clips=clips, frames=frames,
        exact_reference_clips_exp2_exp3=True, reference_log_vectors_all_runs_equal=True,
        official_metrics={run: {**metrics(r['reference_hr'], r['predicted_hr']), **r['metrics']}
                          for run, r in runs.items()},
        hr_bins=bins, subjects=subjects,
        selected_source_validation_clip={run: dict(epoch=selected['epoch'],
                                                   train=selected['train'], valid=selected['valid'])
                                         for run, (_, selected) in histories.items()},
        saved_waveform_analysis=dict(
            uniform_mae_bpm=float(uniform.hr_abs_error_bpm.mean()),
            oracle_region_mae_bpm_diagnostic=float(oracle.mean()),
            correct_processed_regional_candidate=int(candidate.sum()),
            lost_correct_candidate_from_exp2=int(lost.sum()),
            lost_candidate_recordings=list(lost[lost].index),
            gained_correct_candidate_from_exp2=int(gained.sum()),
            large_fused_error_over10=int(big.sum()),
            correct_candidate_large_fused_error=int((big & learned.candidate_correct).sum()),
            no_candidate_large_fused_error=int((big & ~learned.candidate_correct).sum()),
            near_double_fused=int(learned.near_double_reference.sum()),
            near_half_fused=int(learned.near_half_reference.sum()),
            processed_signed_waveform_corr=float(learned.waveform_corr.mean()),
            mean_absolute_unshifted_r=float(abs(learned.waveform_corr).mean()),
            regional_second_to_fundamental_gt1=int((rg.second_to_fundamental_ratio > 1).sum())),
        paired_exp2_exp3=dict(worse=int((pairs.error_delta_bpm > 1e-7).sum()),
                              better=int((pairs.error_delta_bpm < -1e-7).sum()),
                              tied=int((abs(pairs.error_delta_bpm) <= 1e-7).sum())),
        processing_stage_diagnostics=stage_stats,
        detrended_correct_filtered_error_over10_recordings=list(filtered_errors[
            (filtered_errors.detrended <= 5) & (filtered_errors.filtered > 10)].index),
        limits=[
            'One run per architecture; no repeated-seed or causal attention claim.',
            'Source clip metrics differ from full-record audit metrics; Exp3 source-record audit is missing.',
            'PURE has guided development and is exploratory, not fresh confirmatory data.',
            'GT-selected regions are diagnostic; sensitivity scores do not replace the primary evaluator.',
            'No checkpoint binary or server cache hashes supplied with Exp3.',
            'Uniform fusion is post-hoc on jointly learned candidates, not a separately trained model.',
        ])
    # Nested list fields remain JSON rather than Python repr in CSV.
    for table in [df, rg]:
        for col in ['dominant_peaks', 'fundamental_band_hz', 'second_harmonic_band_hz']:
            table[col] = table[col].map(json.dumps)
    df.to_csv(out / 'model_recordings.csv', index=False)
    rg.to_csv(out / 'regional_candidates.csv', index=False)
    stage_df.to_csv(out / 'processing_stages.csv', index=False)
    pd.DataFrame(bins).to_csv(out / 'hr_bins.csv', index=False)
    pd.DataFrame(subjects).to_csv(out / 'subject_metrics.csv', index=False)
    pd.DataFrame(history_rows).to_csv(out / 'training_history.csv', index=False)
    pairs.to_csv(out / 'exp2_exp3_pairs.csv')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    colors = {'exp2': '#3269a8', 'exp3': '#c66532'}
    for run, (rows, _) in histories.items():
        epochs = [r['epoch'] + 1 for r in rows]
        for split, linestyle in [('train', '--'), ('valid', '-')]:
            axes[0, 0].plot(epochs, [r[split]['total'] for r in rows], ls=linestyle,
                            color=colors[run], label=f'{run} {split}')
        axes[0, 1].plot(epochs, [r['valid']['waveform_corr'] for r in rows],
                        color=colors[run], label=run)
    axes[0, 0].set(xlabel='Completed epoch', ylabel='Composite loss', title='Source loss improves in Exp3')
    axes[0, 1].set(xlabel='Completed epoch', ylabel='Mean clip waveform r', title='MMPD validation clips')
    x = np.arange(len(bins))
    for j, run in enumerate(['baseline', 'exp2', 'exp3']):
        axes[1, 0].bar(x + (j - 1) * .25, [r[run + '_mae'] for r in bins], .25,
                       label=run, color=colors.get(run, '#888888'))
    axes[1, 0].set_xticks(x, [f"{r['hr_bin']}\n(n={r['n']})" for r in bins])
    axes[1, 0].set(xlabel='Reference HR (bpm)', ylabel='PURE recording MAE (bpm)', title='Target HR range comparison')
    for run in ['exp2', 'exp3']:
        axes[1, 1].scatter(runs[run]['reference_hr'], runs[run]['predicted_hr'],
                           label=run, alpha=.65, s=24, color=colors[run])
    axes[1, 1].plot([40, 150], [40, 150], 'k--', lw=1)
    axes[1, 1].set(xlabel='Reference HR (bpm)', ylabel='Predicted HR (bpm)', title='Fixed primary PURE evaluator')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / 'exp2_exp3_comparison.png', dpi=170)
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in [
        'official_metrics', 'hr_bins', 'subjects', 'selected_source_validation_clip',
        'saved_waveform_analysis', 'processing_stage_diagnostics', 'paired_exp2_exp3',
        'resolved_config_differences_exp2_to_exp3']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
