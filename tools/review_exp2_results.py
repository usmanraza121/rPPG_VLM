"""Reproduce the Exp2 artifact review without a dataset or checkpoint model.

Read-only with respect to supplied logs/tensors. Uses the unchanged diagnostic
signal pipeline. GT-selected regions/secondary peaks are diagnostics only.
Run from rPPG_VLM: python tools/review_exp2_results.py
"""
import argparse,hashlib,json,re,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch,yaml
from scipy import signal
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.regional_signal_diagnostics import analyze_signal,detrend_signal,preprocess_signal


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def parse_log(path):
    text=path.read_text(encoding='utf-8',errors='replace')
    cfg=yaml.safe_load(text.split('Configuration:',1)[1].split('\nCached Data Path',1)[0])
    metrics={name:float(re.search(r'FFT '+name+r' \(FFT Label\): ([^\s]+)',text).group(1)) for name in ['MAE','RMSE','MAPE','Pearson','SNR']}
    vectors={k:np.fromstring(re.search(label+r':\[([^]]+)\]',text,re.S).group(1),sep=' ') for k,label in [('reference_hr','gt_hr_peak_all'),('predicted_hr','predict_hr_peak_all')]}
    return {'config':cfg,'metrics':metrics,**vectors}

def differences(a,b,p=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return [d for k in sorted(set(a)|set(b)) for d in differences(a.get(k,'<absent>'),b.get(k,'<absent>'),p+'.'+k if p else k)]
    return [] if a==b else [{'key':p,'baseline':a,'exp2':b}]

def array(t):return t.detach().cpu().numpy().astype(np.float64)

def reconstruct(payload,rec):
    clips=sorted(payload['pred_raw'][rec],key=int)
    assert [int(c) for c in clips]==list(range(len(clips)))
    learned=[];reference=[];uniform=[];regions=[];weights=[]
    for c in clips:
        pred=array(payload['pred_raw'][rec][c]).reshape(-1)
        gt=array(payload['gt'][rec][c]).reshape(-1)
        aux=payload['regional'][rec][c]
        region=array(aux['regional_waveforms']);weight=array(aux['region_weights'])
        valid=aux['valid_regions'].cpu().numpy()
        assert pred.shape==gt.shape==(160,) and region.shape==weight.shape==(9,160)
        assert valid.all() and np.isfinite(pred).all() and np.isfinite(region).all()
        assert np.allclose(weight.sum(0),1,atol=1e-6)
        assert np.allclose((weight*region).sum(0),pred,atol=1e-6)
        equal=region.mean(0);equal-=equal.mean()
        region=region-region.mean(-1,keepdims=True)
        learned.append(pred);reference.append(gt);uniform.append(equal);regions.append(region);weights.append(weight)
    return {'learned':np.concatenate(learned),'reference':np.concatenate(reference),'uniform':np.concatenate(uniform),
            'regions':np.concatenate(regions,axis=1),'weights':np.concatenate(weights,axis=1),'clips':len(clips)}

def stage_hrs(values):
    f,p=signal.welch(values,fs=30,window='hann',nperseg=256,noverlap=128,nfft=3333,detrend=False,axis=-1,scaling='density')
    band=(f>=.75)&(f<=2.5)
    return f[band][p[...,band].argmax(-1)]*60


def sensitivity_review(exp1, exp2, learned, summary, out):
    """Fixed-reference, fixed-search secondary analysis; never a new score."""
    rows, spectra, references = [], {}, {}
    for run, payload in [('exp1', exp1), ('exp2', exp2)]:
        for rec in payload['pred_raw']:
            data = reconstruct(payload, rec)
            h = float(stage_hrs(preprocess_signal(data['reference'])))
            values = np.vstack((data['learned'][None], data['regions']))
            variants = {
                'raw': values - values.mean(-1, keepdims=True),
                'detrended': np.stack([detrend_signal(v) for v in values]),
                'filtered': np.stack([preprocess_signal(v) for v in values]),
            }
            for stage, values in variants.items():
                hrs = stage_hrs(values)
                f, p = signal.welch(values[0], fs=30, window='hann', nperseg=256,
                                     noverlap=128, nfft=3333, detrend=False)
                first = p[abs(f - h / 60) <= .1].sum()
                second = p[abs(f - 2 * h / 60) <= .1].sum()
                rows.append({'run': run, 'recording': rec,
                             'subject': f'subject{int(rec) // 100:02d}', 'stage': stage,
                             'reference_hr_bpm': h, 'learned_hr_bpm': float(hrs[0]),
                             'learned_ae_bpm': float(abs(hrs[0] - h)),
                             'candidate_correct': bool((abs(hrs[1:] - h) <= 5).any()),
                             'second_to_fundamental_power': float(second / max(first, 1e-20))})
                if run == 'exp2':
                    spectra[(rec, stage)], references[rec] = (f, p), h
    df = pd.DataFrame(rows)
    df.to_csv(out / 'exp1_exp2_processing_sensitivity.csv', index=False)
    current = df[df.run == 'exp2']
    error = current.pivot(index='recording', columns='stage', values='learned_ae_bpm')
    candidates = current.pivot(index='recording', columns='stage', values='candidate_correct')
    changed = error[(error.detrended <= 5) & (error.filtered > 10)].copy()
    changed['delta'] = changed.filtered - changed.detrended
    changed = changed.sort_values('delta', ascending=False)
    summary['detrended_correct_filtered_error_over10_recordings'] = list(changed.index)
    summary['candidate_no_correct_any_stage'] = int((~candidates.any(axis=1)).sum())
    summary['detrended_candidate_lost_after_filter'] = int((candidates.detrended & ~candidates.filtered).sum())
    summary['detrended_missing_candidate_gained_after_filter'] = int((~candidates.detrended & candidates.filtered).sum())
    summary['subject05_09_absolute_error_fraction'] = float(
        learned[learned.subject.isin(['subject05', 'subject09'])].hr_abs_error_bpm.sum()
        / learned.hr_abs_error_bpm.sum())
    summary['waveform_diagnostics'] = {
        'mean_absolute_unshifted_r': float(abs(learned.waveform_corr).mean()),
        'mean_bounded_lagged_absolute_r_diagnostic': float(learned.lagged_abs_corr.mean()),
        'negative_signed_correlation_recordings': int((learned.waveform_corr < 0).sum()),
        'lag_search_bound_recordings': int((abs(learned.best_lag_seconds) >= 1).sum()),
    }
    summary['processing_sensitivity_both_runs'] = [
        {'run': run, 'stage': stage, 'mae_bpm': float(g.learned_ae_bpm.mean()),
         'candidate_correct': int(g.candidate_correct.sum())}
        for (run, stage), g in df.groupby(['run', 'stage'])]
    for stage in ['raw', 'detrended']:
        summary['processing_stage_diagnostics'][stage]['within5'] = int((error[stage] <= 5).sum())
    import matplotlib.pyplot as plt
    cases = list(changed.index[:3])
    fig, axes = plt.subplots(len(cases), 2, figsize=(11, 3 * len(cases)), layout='constrained')
    for row, rec in enumerate(cases):
        h = references[rec]
        for col, stage in enumerate(['detrended', 'filtered']):
            f, p = spectra[(rec, stage)]
            mask = (f >= .55) & (f <= 2.6)
            p = p / max(p[mask].max(), 1e-20)
            ax = axes[row, col]
            ax.plot(f[mask] * 60, p[mask], color='#3269a8')
            ax.axvline(h, color='#3d9140', ls='--', label='Reference fundamental')
            ax.axvline(2 * h, color='#c67b27', ls=':', label='Reference second harmonic')
            ax.set_title(f'Recording {rec}: {stage}')
            ax.set_ylabel('Relative PSD')
            ax.grid(alpha=.2)
            if row == len(cases) - 1:
                ax.set_xlabel('Frequency (bpm)')
            if row == 0:
                ax.legend(fontsize=8)
    fig.savefig(out / 'filter_harmonic_sensitivity.png', dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=ROOT.parent)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/experiments/results/exp2_20261126')
    args=parser.parse_args();workspace=args.workspace;out=args.output;out.mkdir(parents=True,exist_ok=True)
    runs={k:parse_log(p) for k,p in {'baseline':workspace/'training.log','exp1':workspace/'Exp_data/exp1/training.log','exp2':workspace/'Exp_data/exp2/training.log'}.items()}
    for r in runs.values():assert len(r['reference_hr'])==59 and np.array_equal(r['reference_hr'],runs['baseline']['reference_hr'])
    config_differences=differences(runs['baseline']['config'],runs['exp2']['config'])
    exp2=torch.load(workspace/'Exp_data/exp2/regional_waveforms.pt',map_location='cpu',weights_only=True)
    exp1=torch.load(workspace/'Exp_data/exp1/regional_waveforms.pt',map_location='cpu',weights_only=True)
    assert set(exp2['gt'])==set(exp1['gt']) and len(exp2['gt'])==59
    for rec in exp2['gt']:
        assert set(exp2['gt'][rec])==set(exp1['gt'][rec])
        for clip,gt in exp2['gt'][rec].items():assert torch.equal(gt,exp1['gt'][rec][clip])
    records=[];regional=[];stages=[];matched_arrays=0;total_clips=0
    for index,rec in enumerate(exp2['pred_raw']):
        data=reconstruct(exp2,rec);total_clips+=data['clips'];ref=data['reference']
        subject=f'subject{int(rec)//100:02d}'
        for method in ['learned','uniform']:
            row=analyze_signal(data[method],ref,fs=30)
            assert row['status']=='ok' and row['reference_status']=='ok'
            records.append({'recording':rec,'subject':subject,'method':method,'clips':data['clips'],**row})
        learned=records[-2]
        assert abs(learned['reference_hr_bpm']-runs['exp2']['reference_hr'][index])<1e-7
        assert abs(learned['hr_bpm']-runs['exp2']['predicted_hr'][index])<1e-7
        for r in range(9):
            row=analyze_signal(data['regions'][r],ref,fs=30,max_lag_seconds=0)
            regional.append({'recording':rec,'subject':subject,'region':r,'mean_weight':float(data['weights'][r].mean()),**row})
        for stage in ['raw','detrended','filtered']:
            vals=np.vstack((data['learned'][None],data['regions']))
            if stage=='raw':vals=vals-vals.mean(-1,keepdims=True)
            elif stage=='detrended':vals=np.stack([detrend_signal(v) for v in vals])
            else:vals=np.stack([preprocess_signal(v) for v in vals])
            hrs=stage_hrs(vals);h=learned['reference_hr_bpm']
            stages.append({'recording':rec,'subject':subject,'stage':stage,'reference_hr_bpm':h,'learned_hr_bpm':float(hrs[0]),'learned_ae_bpm':float(abs(hrs[0]-h)),'candidate_correct':bool((abs(hrs[1:]-h)<=5).any())})
        old=workspace/'pure_cross_matched_test_full/waveforms'/f'{rec}_segment0.npz'
        if old.is_file():
            with np.load(old) as x:assert np.array_equal(x['reference'],ref)
            matched_arrays+=1
        if (index+1)%10==0:print('Analyzed',index+1,'of 59 saved recordings',flush=True)
    df=pd.DataFrame(records);rg=pd.DataFrame(regional);stage_df=pd.DataFrame(stages)
    learned=df[df.method=='learned'].copy();equal=df[df.method=='uniform']
    assert total_clips==750 and matched_arrays==12
    assert abs(learned.hr_abs_error_bpm.mean()-runs['exp2']['metrics']['MAE'])<1e-8
    oracle=rg.groupby('recording').hr_abs_error_bpm.min()
    candidate=oracle<=5;learned['candidate_correct']=learned.recording.map(candidate)
    learned['bin']=pd.cut(learned.reference_hr_bpm,[0,60,80,100,np.inf],right=False,labels=['<60','60-80','80-100','>=100'])
    bins=[]
    for label,g in learned.groupby('bin',observed=True):
        mask={'<60':runs['baseline']['reference_hr']<60,'60-80':(runs['baseline']['reference_hr']>=60)&(runs['baseline']['reference_hr']<80),'80-100':(runs['baseline']['reference_hr']>=80)&(runs['baseline']['reference_hr']<100),'>=100':runs['baseline']['reference_hr']>=100}[str(label)]
        bins.append({'hr_bin':str(label),'n':len(g),'baseline_mae':float(np.abs(runs['baseline']['predicted_hr']-runs['baseline']['reference_hr'])[mask].mean()),'exp1_mae':float(np.abs(runs['exp1']['predicted_hr']-runs['exp1']['reference_hr'])[mask].mean()),'exp2_mae':float(g.hr_abs_error_bpm.mean()),'exp2_candidate_correct':int(g.candidate_correct.sum())})
    subjects=learned.groupby('subject').agg(n=('recording','size'),mae_bpm=('hr_abs_error_bpm','mean'),waveform_corr=('waveform_corr','mean'),candidate_correct=('candidate_correct','sum')).reset_index()
    previous=pd.read_csv(workspace/'diagnostics/exp1_review_20261125/model_recordings.csv',dtype={'recording':str})
    previous=previous[previous.method=='learned'].set_index('recording')
    byid=learned.set_index('recording')
    exp1_pairs=previous[['reference_hr_bpm','hr_bpm','hr_abs_error_bpm','waveform_corr']].merge(byid[['reference_hr_bpm','hr_bpm','hr_abs_error_bpm','waveform_corr']],left_index=True,right_index=True,suffixes=('_exp1','_exp2'),how='inner',validate='one_to_one')
    assert len(exp1_pairs)==59 and np.allclose(exp1_pairs.reference_hr_bpm_exp1,exp1_pairs.reference_hr_bpm_exp2)
    old=pd.read_csv(workspace/'pure_cross_matched_test_full/records.csv',dtype={'recording':str});old=old[old.method=='learned'].set_index('recording')
    matched=old[['subject','reference_hr_bpm','hr_bpm','hr_abs_error_bpm','waveform_corr']].merge(byid[['hr_bpm','hr_abs_error_bpm','waveform_corr']],left_index=True,right_index=True,suffixes=('_baseline','_exp2'),how='inner',validate='one_to_one')
    matched=matched.join(previous[['hr_bpm','hr_abs_error_bpm','waveform_corr']].add_suffix('_exp1'))
    assert len(matched)==12
    # Dominant candidates with a reference-near secondary peak remain GT-assisted.
    missing=set(candidate[~candidate].index);near=set()
    for row in rg[rg.recording.isin(missing)].itertuples():
        if any(abs(x['hr_bpm']-row.reference_hr_bpm)<=5 for x in row.dominant_peaks):near.add(row.recording)
    stats={}
    for run,r in runs.items():
        ae=abs(r['predicted_hr']-r['reference_hr']);stats[run]={**r['metrics'],'within5':int((ae<=5).sum()),'over10':int((ae>10).sum()),'over20':int((ae>20).sum()),'bias_bpm':float((r['predicted_hr']-r['reference_hr']).mean())}
    stage_stats={k:{'mae_bpm':float(g.learned_ae_bpm.mean()),'candidate_correct':int(g.candidate_correct.sum())} for k,g in stage_df.groupby('stage')}
    history=[json.loads(s) for s in (workspace/'Exp_data/exp2/regional_history.jsonl').read_text().splitlines()]
    assert [x['epoch'] for x in history]==list(range(40))
    best=min((x for x in history if x['checkpoint_eligible']),key=lambda x:x['selection_score'])
    baseline_valid=json.loads((workspace/'diagnostics/exp1_review_20261125/summary.json').read_text())['selected_source_validation']['baseline_epoch37']
    exp1_history=[json.loads(s) for s in (workspace/'Exp_data/exp1/regional_history.jsonl').read_text().splitlines()]
    exp1_best=min((x for x in exp1_history if x['checkpoint_eligible']),key=lambda x:x['selection_score'])
    big=learned.hr_abs_error_bpm>10
    summary={'review_date':'2026-11-26','source_hashes':{f.name:sha(f) for f in (workspace/'Exp_data/exp2').iterdir() if f.is_file()},'run_directory':runs['exp2']['config']['LOG']['PATH'],'loaded_checkpoint':'runs/RegionalSTAttention_20261126_093244/checkpoints/MMPD_PURE_RegionalSTAttention_best.pth','selected_epoch_zero_based':best['epoch'],'model_metadata':{k:v for k,v in exp2.items() if k not in ['gt','pred_raw','regional']},'epochs':len(history),'recordings':59,'clips':total_clips,'frames':int(learned.sample_count.sum()),'all_exp1_exp2_reference_clips_exactly_equal':True,'exact_baseline_reference_recordings':matched_arrays,'reference_log_vectors_equal':True,'resolved_config_differences':config_differences,'metrics':stats,'hr_bins':bins,'subject_metrics':subjects.to_dict('records'),'selected_source_validation_clip':{'baseline':baseline_valid,'exp1':exp1_best['valid'],'exp2':best['valid']},'saved_waveform_analysis':{'uniform_mae_bpm':float(equal.hr_abs_error_bpm.mean()),'processed_signed_waveform_corr':float(learned.waveform_corr.mean()),'oracle_region_mae_bpm_diagnostic':float(oracle.mean()),'correct_processed_regional_candidate':int(candidate.sum()),'large_fused_error_over10':int(big.sum()),'correct_candidate_large_fused_error':int((big&learned.candidate_correct).sum()),'no_candidate_large_fused_error':int((big&~learned.candidate_correct).sum()),'missing_candidate_with_reference_near_top3':len(near),'near_double_fused':int(learned.near_double_reference.sum()),'near_half_fused':int(learned.near_half_reference.sum()),'regional_second_to_fundamental_power_gt1':int((rg.second_to_fundamental_ratio>1).sum())},'processing_stage_diagnostics':stage_stats,'matched12':{'baseline_mae':float(matched.hr_abs_error_bpm_baseline.mean()),'exp1_mae':float(matched.hr_abs_error_bpm_exp1.mean()),'exp2_mae':float(matched.hr_abs_error_bpm_exp2.mean()),'baseline_waveform_corr':float(matched.waveform_corr_baseline.mean()),'exp2_waveform_corr':float(matched.waveform_corr_exp2.mean())},'limits':['Single exploratory run; architecture and capacity changed, attention effect not isolated.','No full-record Exp2 MMPD source-validation audit supplied.','Full59 baseline comparisons use consistent log order/reference vectors; baseline arrays independently verified only for12recordings.','No trained checkpoint or server cache hashes supplied; resolved settings and saved references support comparison.','Processed dominant or secondary spectral match does not establish physiology; oracle is diagnostic only.','PURE test already informed development; fresh data required for confirmatory claims.']}
    from tools.audit_regional_dino import write_csv
    write_csv(out/'model_recordings.csv',records);write_csv(out/'regional_candidates.csv',regional)
    stage_df.to_csv(out/'processing_stages.csv',index=False);subjects.to_csv(out/'subject_metrics.csv',index=False)
    pd.DataFrame(bins).to_csv(out/'hr_bins.csv',index=False);matched.to_csv(out/'matched12_recordings.csv');exp1_pairs.to_csv(out/'exp1_exp2_pairs.csv')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained');x=np.arange(4);w=.26
    for j,(run,color) in enumerate([('baseline','#777777'),('exp1','#c67b27'),('exp2','#3269a8')]):
        axes[0].bar(x+(j-1)*w,[r[run+'_mae'] for r in bins],w,label=run,color=color)
    axes[0].set_xticks(x,[f"{r['hr_bin']}\n(n={r['n']})" for r in bins]);axes[0].set_ylabel('PURE recording MAE (bpm)');axes[0].set_xlabel('Reference HR (bpm)');axes[0].legend(frameon=False);axes[0].set_title('High-rate recovery; low-rate failures remain')
    axes[1].plot([r['epoch']+1 for r in history],[r['valid']['clip_hr_mae'] for r in history],label='Exp2 attention',color='#3269a8')
    axes[1].plot([r['epoch']+1 for r in exp1_history],[r['valid']['clip_hr_mae'] for r in exp1_history],label='Exp1 slowing',color='#c67b27')
    axes[1].axhline(baseline_valid['clip_hr_mae'],label='Original selected checkpoint',color='#777777',ls='--');axes[1].set_ylabel('MMPD validation clip HR MAE (bpm)');axes[1].set_xlabel('Completed training epoch');axes[1].set_title('Source clip accuracy still trails baseline');axes[1].legend(frameon=False)
    for a in axes:a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
    fig.savefig(out/'rate_recovery_and_source_validation.png',dpi=170);plt.close(fig)
    sensitivity_review(exp1, exp2, learned, summary, out)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['metrics','hr_bins','subject_metrics','selected_source_validation_clip','saved_waveform_analysis','processing_stage_diagnostics','matched12','resolved_config_differences']},indent=2))
    return summary

if __name__=='__main__':main()
