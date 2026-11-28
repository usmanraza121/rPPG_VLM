"""Review both seed101 runs against the original attention/TCN pair.

Read-only on supplied results. Run: python tools/review_seed101_results.py
Reference-selected regional results and processing sensitivity are diagnostic.
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
from tools.review_exp2_results import differences, parse_log, reconstruct, sha, stage_hrs
from tools.review_exp3_results import history, metrics
from tools.regional_signal_diagnostics import analyze_signal, detrend_signal, preprocess_signal


def grouped_rows(frame, run, dataset):
    rows = []
    for column in ['hr_bin', 'subject']:
        for label, g in frame.groupby(column, observed=True):
            rows.append(dict(run=run, dataset=dataset, grouping=column, group=str(label),
                             n=len(g), mae_bpm=float(g.hr_abs_error_bpm.mean()),
                             bias_bpm=float((g.hr_bpm-g.reference_hr_bpm).mean()),
                             within5=int((g.hr_abs_error_bpm <= 5).sum()),
                             candidate_correct=int(g.candidate_correct.sum()),
                             waveform_corr=float(g.waveform_corr.mean())))
    return rows


def waveform_stats(frame, regional, uniform):
    candidate = regional.groupby('recording').hr_abs_error_bpm.min() <= 5
    frame['candidate_correct'] = frame.recording.map(candidate)
    big = frame.hr_abs_error_bpm > 10
    missing = set(candidate[~candidate].index)
    near = set()
    for row in regional[regional.recording.isin(missing)].itertuples():
        peaks = json.loads(row.dominant_peaks) if isinstance(row.dominant_peaks, str) else row.dominant_peaks
        if any(abs(p['hr_bpm'] - row.reference_hr_bpm) <= 5 for p in peaks):
            near.add(row.recording)
    return dict(uniform_mae_bpm=float(uniform.hr_abs_error_bpm.mean()),
                oracle_region_mae_bpm_diagnostic=float(regional.groupby('recording').hr_abs_error_bpm.min().mean()),
                correct_processed_candidate=int(candidate.sum()),
                large_fused_error_over10=int(big.sum()),
                correct_candidate_large_fused_error=int((big & frame.candidate_correct).sum()),
                missing_candidate_large_fused_error=int((big & ~frame.candidate_correct).sum()),
                missing_candidate_with_reference_near_top3_diagnostic=len(near),
                waveform_corr=float(frame.waveform_corr.mean()),
                mean_absolute_unshifted_r=float(abs(frame.waveform_corr).mean()),
                snr_fundamental_db=float(frame.snr_fundamental_db.mean()),
                snr_harmonics_db=float(frame.snr_harmonics_db.mean()),
                near_double=int(frame.near_double_reference.sum()),
                near_half=int(frame.near_half_reference.sum()))


def stages_for(values, reference_hr, run, rec, dataset):
    rows = []
    for stage in ['raw', 'detrended', 'filtered']:
        x = (values-values.mean(-1, keepdims=True) if stage == 'raw' else
             np.stack([(detrend_signal(v) if stage == 'detrended' else preprocess_signal(v)) for v in values]))
        hrs = stage_hrs(x)
        rows.append(dict(run=run, recording=rec, dataset=dataset, stage=stage,
                         reference_hr_bpm=reference_hr, hr_bpm=float(hrs[0]),
                         hr_abs_error_bpm=float(abs(hrs[0]-reference_hr)),
                         candidate_correct=bool((abs(hrs[1:]-reference_hr) <= 5).any())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=ROOT.parent)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/experiments/results/seed101_20261127')
    args = parser.parse_args()
    w, out = args.workspace, args.output
    out.mkdir(parents=True, exist_ok=True)
    new = w/'Exp_data/expt4_st_attention_tcn'
    dirs = dict(attention100=w/'Exp_data/exp2', tcn100=w/'Exp_data/exp3',
                attention101=new/'st_attention', tcn101=new/'st_tcn')
    audits = dict(attention100=dirs['attention100']/'mmpd_exp2_attention_full',
                  tcn100=dirs['tcn100']/'mmpd_exp3_tcn_full',
                  attention101=dirs['attention101']/'mmpd_attention_seed101_full',
                  tcn101=dirs['tcn101']/'mmpd_tcn_seed101_full')
    logs = {run: parse_log(p/'training.log') for run, p in dirs.items()}
    histories = {run: history(p/'regional_history.jsonl') for run,p in dirs.items()}
    manifests = {run: json.loads((p/'manifest.json').read_text()) for run,p in audits.items()}
    checks, config_changes = {}, {}
    for run in dirs:
        assert np.array_equal(logs[run]['reference_hr'],logs['attention100']['reference_hr'])
        assert len(logs[run]['reference_hr']) == 59
        manifest = manifests[run]
        for key in ['file_list_sha256','train_file_list_sha256','train_subjects','audited_subjects','protocol']:
            checks[f'{run}_{key}'] = manifest[key] == manifests['attention100'][key]
        for key in ['split','window_seconds','min_hz','max_hz','max_recordings']:
            checks[f'{run}_argument_{key}'] = manifest['command_arguments'][key] == manifests['attention100']['command_arguments'][key]
        for name in ['coverage.csv','reference_checks.csv']:
            checks[f'{run}_{name}_bytes'] = (audits[run]/name).read_bytes() == (audits['attention100']/name).read_bytes()
        assert manifest['checkpoint_epoch'] == histories[run][1]['epoch']
        assert manifest['model_arguments']['temporal_backend'] == ('attention' if run.startswith('attention') else 'tcn')
        if run.endswith('101'):
            assert logs[run]['config']['RANDOM_SEED'] == 101
            assert manifest['checkpoint_training_seed'] == 101
            assert all(x['random_seed'] == 101 for x in histories[run][0])
            changes = differences(logs[run.replace('101','100')]['config'], logs[run]['config'])
            config_changes[run] = [dict(key=d['key'], seed100=d['baseline'], seed101=d['exp2']) for d in changes]
            assert {d['key'] for d in changes} == {'LOG.PATH','MODEL.MODEL_DIR','RANDOM_SEED'}
    assert all(checks.values())
    records, regions, stages, grouped, stats = [], [], [], [], []
    datasets = {}
    # Existing reproducible PURE tables avoid redoing the completed seed100 audit.
    for run, folder in [('attention100','exp2_20261126'),('tcn100','exp3_20261127')]:
        p = ROOT/'docs/experiments/results'/folder
        table = pd.read_csv(p/'model_recordings.csv',dtype={'recording':str})
        rg = pd.read_csv(p/'regional_candidates.csv',dtype={'recording':str})
        learned = table[table.method == 'learned'].copy()
        info = waveform_stats(learned,rg,table[table.method == 'uniform'])
        learned['hr_bin'] = pd.cut(learned.reference_hr_bpm,[0,60,80,100,np.inf],right=False,labels=['<60','60-80','80-100','>=100'])
        datasets[(run,'PURE')] = learned
        grouped += grouped_rows(learned,run,'PURE')
        stats.append(dict(run=run,dataset='PURE',**metrics(learned.reference_hr_bpm,learned.hr_bpm),**info))
    ref_payload = torch.load(dirs['attention100']/'regional_waveforms.pt',map_location='cpu',weights_only=True)
    new_metadata = {}
    for run in ['attention101','tcn101']:
        payload = torch.load(dirs[run]/'regional_waveforms.pt',map_location='cpu',weights_only=True)
        assert payload['training_seed'] == payload['inference_seed'] == 101
        assert payload['model_arguments'] == manifests[run]['model_arguments']
        assert payload['fs'] == 30 and payload['normalization'] == 'raw' and payload['label_type'] == 'Standardized'
        assert set(payload['gt']) == set(ref_payload['gt'])
        new_metadata[run] = {k:v for k,v in payload.items() if k not in ['pred_raw','gt','regional']}
        local, local_regions, clip_count = [], [], 0
        for index,rec in enumerate(payload['pred_raw']):
            assert set(payload['gt'][rec]) == set(ref_payload['gt'][rec])
            for clip,gt in payload['gt'][rec].items():
                assert torch.equal(gt,ref_payload['gt'][rec][clip])
            data = reconstruct(payload,rec)
            clip_count += data['clips']
            subject = f'subject{int(rec)//100:02d}'
            for method in ['learned','uniform']:
                row = analyze_signal(data[method],data['reference'],fs=30,max_lag_seconds=0)
                assert row['status'] == row['reference_status'] == 'ok'
                local.append(dict(run=run,recording=rec,subject=subject,method=method,**row))
            learned = local[-2]
            assert abs(learned['reference_hr_bpm']-logs[run]['reference_hr'][index]) < 1e-7
            assert abs(learned['hr_bpm']-logs[run]['predicted_hr'][index]) < 1e-7
            for region in range(9):
                row = analyze_signal(data['regions'][region],data['reference'],fs=30,max_lag_seconds=0)
                assert row['status'] == row['reference_status'] == 'ok'
                local_regions.append(dict(run=run,recording=rec,subject=subject,region=region,**row))
            stage = stages_for(np.vstack((data['learned'][None],data['regions'])),learned['reference_hr_bpm'],run,rec,'PURE')
            assert abs(stage[-1]['hr_bpm']-learned['hr_bpm']) < 1e-8
            stages += stage
        assert clip_count == 750
        table, rg = pd.DataFrame(local), pd.DataFrame(local_regions)
        learned = table[table.method == 'learned'].copy()
        info = waveform_stats(learned,rg,table[table.method == 'uniform'])
        assert abs(learned.hr_abs_error_bpm.mean()-logs[run]['metrics']['MAE']) < 1e-8
        learned['hr_bin'] = pd.cut(learned.reference_hr_bpm,[0,60,80,100,np.inf],right=False,labels=['<60','60-80','80-100','>=100'])
        datasets[(run,'PURE')] = learned
        grouped += grouped_rows(learned,run,'PURE')
        stats.append(dict(run=run,dataset='PURE',**metrics(learned.reference_hr_bpm,learned.hr_bpm),**info))
        records += local
        regions += local_regions
        print('Reconstructed PURE',run,flush=True)
    source_tables = {run:pd.read_csv(p/'records.csv') for run,p in audits.items()}
    for run, table in source_tables.items():
        pair = source_tables['attention100'].merge(table,on=['recording','segment','window','method'],how='outer',
                                                   suffixes=('_old','_new'),validate='one_to_one',indicator=True)
        assert (pair._merge == 'both').all()
        for col in ['reference_hr_bpm','reference_std','frames','sample_count']:
            assert np.allclose(pair[col+'_old'],pair[col+'_new'],equal_nan=True)
        classic = pair[pair.method.str.contains('GREEN|POS|CHROM')]
        for col in ['hr_bpm','hr_abs_error_bpm','waveform_corr']:
            assert np.allclose(classic[col+'_old'],classic[col+'_new'],equal_nan=True)
        rg = pd.read_csv(audits[run]/'regional.csv')
        rg = rg[rg.method == 'regional_candidate']
        learned = table[table.method == 'learned'].copy()
        assert len(learned) == 137 and learned.recording.is_unique
        assert int(learned.frames.sum()) == 241120 and (learned.status == 'ok').all()
        info = waveform_stats(learned,rg,table[table.method == 'uniform'])
        learned['hr_bin'] = pd.cut(learned.reference_hr_bpm,[0,60,80,100,np.inf],right=False,labels=['<60','60-80','80-100','>=100'])
        datasets[(run,'MMPD')] = learned
        grouped += grouped_rows(learned,run,'MMPD')
        stats.append(dict(run=run,dataset='MMPD',**metrics(learned.reference_hr_bpm,learned.hr_bpm),**info))
        if not run.endswith('101'):
            continue
        count = 0
        for file in sorted((audits[run]/'waveforms').glob('*.npz')):
            rec = file.name.rsplit('_segment',1)[0]
            with np.load(file) as x, np.load(audits['attention100']/'waveforms'/file.name) as old:
                for key in ['reference','rgb_global','rgb_regions']:
                    assert np.array_equal(x[key],old[key])
                assert int(x['clip_length']) == 160 and int(x['clip_count']) == 11
                assert np.isfinite(x['learned']).all() and x['candidates'].shape == (9,1760)
                assert np.allclose(x['weights'].sum(0),1,atol=1e-6)
                weighted = (x['weights']*x['candidates']).sum(0).reshape(-1,160)
                assert np.allclose((weighted-weighted.mean(-1,keepdims=True)).reshape(-1),x['learned'],atol=1e-6)
                assert np.allclose(x['candidates'].mean(0),x['uniform'],atol=1e-6)
                values = np.vstack((x['learned'][None],x['candidates'])).astype(float)
                h = float(stage_hrs(preprocess_signal(x['reference'].astype(float))))
            expected = learned[learned.recording == rec].iloc[0]
            assert abs(h-expected.reference_hr_bpm) < 1e-8
            stage = stages_for(values,h,run,rec,'MMPD')
            assert abs(stage[-1]['hr_bpm']-expected.hr_bpm) < 1e-8
            assert stage[-1]['candidate_correct'] == expected.candidate_correct
            stages += stage
            count += 1
        assert count == 137
        checks[run+'_137_exact_source_reference_RGB_arrays'] = True
        print('Checked MMPD',run,flush=True)
    df, group_df, stage_df = pd.DataFrame(stats),pd.DataFrame(grouped),pd.DataFrame(stages)
    pair_rows, seed_means = [], []
    for dataset in ['PURE','MMPD']:
        for seed in [100,101]:
            a,b = [datasets[(f'{backend}{seed}',dataset)].set_index('recording') for backend in ['attention','tcn']]
            p = a[['subject','reference_hr_bpm','hr_bpm','hr_abs_error_bpm','candidate_correct']].join(
                b[['hr_bpm','hr_abs_error_bpm','candidate_correct']],lsuffix='_attention',rsuffix='_tcn',validate='one_to_one')
            delta = p.hr_abs_error_bpm_attention-p.hr_abs_error_bpm_tcn
            pair_rows.append(dict(dataset=dataset,seed=seed,attention_minus_tcn_mae=float(delta.mean()),
                                  attention_better=int((delta < -1e-7).sum()),attention_worse=int((delta > 1e-7).sum()),
                                  ties=int((abs(delta) <= 1e-7).sum()),
                                  only_attention_correct_candidate=int((p.candidate_correct_attention & ~p.candidate_correct_tcn).sum()),
                                  only_tcn_correct_candidate=int((~p.candidate_correct_attention & p.candidate_correct_tcn).sum())))
            p.to_csv(out/f'{dataset.lower()}_paired_seed{seed}.csv')
        for backend in ['attention','tcn']:
            rows = df[(df.dataset == dataset)&df.run.str.startswith(backend)]
            seed_means.append(dict(dataset=dataset,backend=backend,seeds=[100,101],
                                   mae_mean=float(rows.MAE.mean()),mae_min=float(rows.MAE.min()),mae_max=float(rows.MAE.max()),
                                   mae_sample_sd_descriptive_n2=float(rows.MAE.std(ddof=1))))
    stage_stats = [dict(run=run,dataset=dataset,stage=stage,MAE=float(g.hr_abs_error_bpm.mean()),
                        within5=int((g.hr_abs_error_bpm <= 5).sum()),candidate_correct=int(g.candidate_correct.sum()))
                   for (run,dataset,stage),g in stage_df.groupby(['run','dataset','stage'])]
    selected = {run:dict(epoch=best['epoch'],train=best['train'],valid=best['valid'])
                for run,(_,best) in histories.items()}
    fingerprints = {run:{name:sha(p/name) for name in ['training.log','regional_history.jsonl','regional_waveforms.pt']}
                    for run,p in dirs.items()}
    for run,p in audits.items():
        fingerprints[run]['audit'] = {name:sha(p/name) for name in ['manifest.json','summary.json','records.csv','regional.csv','coverage.csv','reference_checks.csv']}
    summary = dict(review_date='2026-11-27',matching_checks=checks,config_changes=config_changes,
                   new_tensor_metadata=new_metadata,
                   audit_provenance={run:{k:m.get(k) for k in ['checkpoint_epoch','checkpoint_training_seed','checkpoint_sha256','model_arguments']}
                                     for run,m in manifests.items()},
                   metrics=stats,grouped_metrics=grouped,paired_seed_results=pair_rows,
                   two_seed_descriptive_summary=seed_means,selected_source_clip=selected,
                   processing_sensitivity=stage_stats,source_sha256=fingerprints,
                   limits=['Two paired seeds are insufficient for a stable variance or universal mechanism claim.',
                           'Seed100 provenance comes from the historical fixed entry point, not embedded checkpoint metadata.',
                           'Checkpoint hashes are supplied manifest values, not local checkpoint rehashes.',
                           'MMPD validation selected checkpoints; PURE has guided development and is exploratory.',
                           'Reference-selected regions and processing sensitivity are diagnostics, not alternative inference scores.',
                           'Mean HR accuracy and waveform fidelity are distinct; correlated recordings are not independent subjects.'])
    df.to_csv(out/'metrics.csv',index=False)
    group_df.to_csv(out/'groups.csv',index=False)
    stage_df.to_csv(out/'processing_stages.csv',index=False)
    pd.DataFrame(seed_means).to_csv(out/'two_seed_summary.csv',index=False)
    pd.DataFrame(pair_rows).to_csv(out/'paired_seed_summary.csv',index=False)
    for filename,rows in [('pure_new_recordings.csv',records),('pure_new_regional.csv',regions)]:
        table = pd.DataFrame(rows)
        for column in ['dominant_peaks','fundamental_band_hz','second_harmonic_band_hz']:
            table[column] = table[column].map(json.dumps)
        table.to_csv(out/filename,index=False)
    history_rows = [dict(run=run,epoch=row['epoch'],split=split,**row[split])
                    for run,(rows,_) in histories.items() for row in rows for split in ['train','valid']]
    pd.DataFrame(history_rows).to_csv(out/'training_history.csv',index=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,dataset in zip(axes,['MMPD','PURE']):
        for backend,color in [('attention','#3269a8'),('tcn','#c66532')]:
            g=df[(df.dataset == dataset)&df.run.str.startswith(backend)].sort_values('run')
            ax.plot([100,101],g.MAE,'o-',label=backend,color=color)
        ax.set(xlabel='Training seed',ylabel='Recording MAE (bpm)',title=f'{dataset}: paired seed comparison',xticks=[100,101])
        ax.legend(frameon=False);ax.grid(alpha=.2)
    fig.savefig(out/'paired_seed_comparison.png',dpi=170);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ['metrics','paired_seed_results','two_seed_descriptive_summary','processing_sensitivity']},indent=2))


if __name__ == '__main__':
    main()
