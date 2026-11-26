# Exp1 full MMPD validation audit: source regression confirmed

Reviewed 2026-11-26. This completes the source-domain audit requested after the first paired-slowing experiment. Inputs are `mmpd_exp1_rate_aug_full/` and the original `mmpd_no_dino_full/`. Derived artifacts are in `diagnostics/exp1_mmpd_review_20261126/`. This review updates documentation and derived analysis; it does not change training, labels, preprocessing or evaluation.

## Main result

Slowing-only augmentation improves the previously examined PURE aggregate while worsening full-record MMPD validation. Keep Exp1 as an ablation rather than the default architecture comparator.

| MMPD validation, 137 recordings | No augmentation | Exp1 slowing |
|---|---:|---:|
| Learned-fusion MAE (bpm) | 8.5660 | 11.4239 |
| RMSE (bpm) | 13.5393 | 17.0337 |
| Median absolute error (bpm) | 3.7804 | 5.4005 |
| HR Pearson | .4981 | .3306 |
| Within 5 bpm | 83/137 | 68/137 |
| Errors over 10 bpm | 43 | 57 |
| Errors over 20 bpm | 22 | 32 |
| Mean signed waveform correlation | .3565 | .3409 |
| Signed HR bias (prediction minus reference, bpm) | -7.1390 | -10.7380 |
| Uniform fusion MAE, identical candidates | 9.3228 | 11.5501 |
| Reference-selected regional oracle MAE, diagnostic only | 4.2455 | 5.6686 |

Source MAE rises **2.8580 bpm (33.4%)**. By matched recording, Exp1 improves 23, worsens 79 and ties 35; six of seven subject MAEs worsen. The prior PURE result remains 18.2611 to 13.5837 bpm (25.6% lower), with low-rate gain and high-rate regression. These results establish a trade-off in these runs, not a universally useful augmentation or a uniquely identified rate-prior mechanism. Resampling also changes motion, interpolation and available temporal context.

## Comparison validity

The supplied manifests have identical training and audited file-list SHA256 values, subject lists, model arguments and evaluation protocol. Coverage and reference-check files are byte-identical. All recording/method keys match one to one; reference HR, autocorrelation HR, reference standard deviation, sample counts and frame counts match. Classical GREEN/POS/CHROM results are unchanged. All 137 recordings and 241,120 frames are evaluated, with no omitted tails.

Resolved configurations differ only in base-YAML lineage, experiment name and the opt-in RATE_AUG block. Exp1 manifest identifies epoch39 and checkpoint SHA256 `8f781050861031da303f01bdb865086fed2d75778df54bd0e78d418ff9ea87e3`; baseline epoch37 SHA256 `6e9bcdf82a62ff3397bdf5810c601440e0d101195840837313d9ec3b8d93cc83`. These are manifest values, not locally rehashed checkpoint binaries. Exact source artifact fingerprints are in the derived summary. Matching lists and diagnostics do not constitute a local full-cache rehash.

MMPD validation already selected each checkpoint; this is source-validation diagnosis, not an independent source-test score. These recording scores are distinct from the previous 12.47/15.57 clip diagnostics. There are seven people, not 137 independent participants, and only one training run per setting.

## Heart-rate and subject breakdown

| Reference HR (bpm) | Recordings | Baseline MAE | Exp1 MAE |
|---|---:|---:|---:|
| 60–80 | 37 | 2.6127 | 1.9559 |
| 80–100 | 79 | 6.9728 | 11.3685 |
| >=100 | 21 | 25.0482 | 28.3143 |

The largest bin-level deterioration is 80–100 bpm, not only the high-rate tail. This validation set has no references below60 bpm; its reference range is 61.57–126.91. It cannot independently confirm the low-HR PURE benefit.

| MMPD subject | Recordings | Baseline MAE | Exp1 MAE |
|---|---:|---:|---:|
| 27 | 20 | 3.3213 | 4.8335 |
| 28 | 20 | 11.5032 | 17.0117 |
| 29 | 20 | 9.6670 | 12.9343 |
| 30 | 20 | 8.9109 | 13.5554 |
| 31 | 20 | 18.4698 | 21.6292 |
| 32 | 19 | 2.6434 | 4.6331 |
| 33 | 18 | 4.7705 | 4.3204 |

![MMPD rate and subject trade-off](../../../diagnostics/exp1_mmpd_review_20261126/source_rate_and_subject_tradeoff.png)

## Extraction versus fusion

Recordings with any processed regional dominant HR within5 bpm fall from **110 to97**; missing-candidate recordings increase **27 to40**. Of Exp1's57 fused errors over10 bpm,19 retain a correct dominant candidate and38 do not. Baseline has19 and24 respectively. The oracle and uniform scores also worsen, so changed weighting alone does not explain the degradation. The learned-over-uniform MAE benefit shrinks from .7569 to .1261 bpm for the respective checkpoints' candidates; this does not isolate a causal change in gating because the learned candidates also changed.

Among the40 missing-candidate recordings,32 have a reference-near peak among a region's three reported peaks. This suggests competing spectral components and motivates improving discrimination, but reference-assisted secondary-peak selection is diagnostic only. It does not establish usable physiological recovery or justify selecting the peak nearest GT at inference.

This source failure is not principally second-harmonic dominance: learned outputs have zero near-double/near-half classifications, and only4/1,233 regional spectra have second/reference fundamental band-power ratio above1 (baseline8/1,233). These bands use the reference and the detrended, un-bandpassed spectrum; they are diagnostic quantities.

## Saved waveform checks: filtering does not explain the regression

All137 saved NPZ files were analyzed with the same reference HR and fixed .75–2.5Hz HR search. Raw means only DC removal; detrended adds the existing Tarvainen operation; filtered adds the current bandpass. Filtered learned and candidate HRs reproduce the supplied tables.

| Exp1 processing stage, diagnostic only | Learned MAE | Recordings with a correct dominant regional candidate |
|---|---:|---:|
| Raw | 13.0323 | 96/137 |
| Detrended, without bandpass | 12.9652 | 96/137 |
| Current filtered pipeline | 11.4239 | 97/137 |

Only one recording has a correct raw dominant regional candidate that disappears after filtering; two gain one. **39 recordings lack a correct dominant candidate at every checked stage.** Raw outputs can still contain weak non-dominant pulse components, and all stage analyses retain the fixed search band. These results distinguish this source-wide regression from the earlier individual PURE filter-edge failure; they do not replace primary scores or prove raw pulse absence.

## Next development decision

The requested audit is complete. No additional identical server audit is needed before architecture development. Preserve the original no-augmentation model as the comparison anchor; keep Exp1 separate.

Proceed with the previously discussed pulse-extractor hypothesis: preserve spatial features during temporal processing and provide broader temporal context. Include a context-only TCN control to distinguish longer support from joint spatiotemporal processing; account for capacity and compute. Retain regional heads/fusion, loss, schedule, split, checkpoint selection and evaluator initially, with rate augmentation disabled. The 61-frame convolutional receptive field and early spatial pooling are plausible limitations, not proven causes: PURE intra-dataset success shows the current model can learn pulse signals.

The research target is better discrimination of weak pulse from competing rhythms across rates and conditions. A later nuisance-conditioned mechanism must beat those architectural controls and relevant factorization/spectral baselines before a novelty claim. Test architecture and augmentation as separate factors after establishing the new extractor. Use predefined development criteria and fresh confirmatory data because PURE test results have already informed design.

[Summary and fingerprints](../../../diagnostics/exp1_mmpd_review_20261126/summary.json), [matched recordings](../../../diagnostics/exp1_mmpd_review_20261126/matched_recordings.csv), [rate bins](../../../diagnostics/exp1_mmpd_review_20261126/hr_bins.csv), [subjects](../../../diagnostics/exp1_mmpd_review_20261126/subject_metrics.csv), [processing-stage diagnostics](../../../diagnostics/exp1_mmpd_review_20261126/exp1_processing_stages.csv), [table-analysis script](../../../diagnostics/exp1_mmpd_review_20261126/analyze_mmpd_exp1.py).
