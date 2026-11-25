# Matched PURE audits: transfer and filter-edge failure

Reviewed 2026-11-25. Inputs: workspace `pure_cross_matched_test_full/` and `pure_intra_test_full/`. Both are read-only supplied audit artifacts. No model, trainer, loss, label, cache or official evaluator was modified in this review.

## Main conclusions

1. Changing training data from MMPD to PURE substantially improves this unchanged no-DINO model on the same twelve PURE recordings. The result establishes a matched transfer/learning gap under these training setups; it does not isolate HR coverage, visual domain, optimization or reference conventions as its cause.
2. Learned regional weighting gives no measurable HR gain here relative to uniform fusion of the same outputs. A separately trained uniform model would answer a different question.
3. The remaining PURE-trained near-doubling case **contains the correct dominant raw fundamental**. The shared evaluation bandpass changes relative peak power and selects the second harmonic near the lower cutoff. This is different from the MMPD-trained waveform, which already favors the wrong rhythm before filtering.
4. Fixed full-face POS is a stronger HR baseline than either model on these twelve recordings. The intra model demonstrates learning but does not establish superiority over a classical baseline.

## Matching and provenance

The test CSV hash is identical: `f6a04525365f1a6c3a73e780a18f9b857f1a4edc0693946a3429024d01d24150`. Both audits contain subjects09 and10, all twelve recording IDs901–906/1001–1006, identical model arguments,24640evaluatedframes and zero omitted frames. The saved reference arrays and both global/regional RGB traces are **exactly equal for every recording**. Reference-check CSVs are byte-identical; classical-control results also match. Both use the same Tarvainen100, first-order zero-phase Butterworth.75–2.5Hz and Welch3333 pipeline. This is now a verified paired evaluation, unlike comparing the previous59recording aggregate with the12recording intra score.

Cross-trained checkpoint: `runs/RegionalDINO_20261124_121357/checkpoints/MMPD_PURE_RegionalShallowLearned_best.pth`,epoch37,SHA256 `6e9bcdf82a62ff3397bdf5810c601440e0d101195840837313d9ec3b8d93cc83`.

Intra-trained checkpoint: `runs/RegionalDINO_20261125_113336/checkpoints/PURE_PURE_RegionalShallowLearned_6_2_2_best.pth`,epoch25,SHA256 `93261aafa026fbc14b2a83e1e63bc607511e7583ec06f19b2537a1f89821c14f`.

The shared PURE config specifies evaluation data and architecture; it does not mean the first checkpoint was trained on PURE. Both test-mode manifests have null training-subject metadata. Actual PURE training/validation CSV contents were not supplied, so this comparison verifies the test identities but does not independently certify the training split membership. Intended subjects01–06/07–08/09–10 remain established by the config/loader, pending CSV verification.

## Official common-protocol results

| Method | MAE (bpm) | RMSE (bpm) | Within5 | Mean signed waveform r |
|---|---:|---:|---:|---:|
| MMPD-trained learned fusion |20.7021|28.4568|5/12|.0493|
| PURE-trained learned fusion |4.0504|12.9463|11/12|.8332|
| MMPD-trained uniform fusion |20.5671|28.3451|5/12|.0474|
| PURE-trained uniform fusion |4.0504|12.9463|11/12|.8313|
| Full-face POS |.3150|.5621|12/12|-.6747|
| Central-cell POS |.4500|.6236|12/12|-.6318|
| Full-face GREEN |3.9154|11.1106|11/12|-.4969|
| Full-face CHROM |19.8470|30.2024|7/12|.6675|

Matched learned MAE decreases16.6517bpm,80.4%. The intra checkpoint wins seven recordings, ties three, and loses two by approximately one.54bpm spectral bin. HR Pearson changes-.3366→.7486; near-double count3→1. These are descriptive results on two previously examined subjects, not a population confidence interval or a new contribution claim.

| Subject | Recordings | Reference HR range | Cross MAE | Intra MAE |
|---|---:|---|---:|---:|
|09|6|45.36–52.93|39.5140|7.6508|
|10|6|71.83–100.45|1.8902|.4500|

Subject09 supplies about95.7%of the aggregate absolute-error improvement. Identity and HR are confounded: we cannot claim that low HR alone caused transfer failure. The intra model succeeds on five of six low-HR recordings; the remaining failure dominates its aggregate. No test reference is above100.45, so this does not settle the120–135bpm failure in other experiments.

The negative classical waveform correlations can reflect optical/PPG polarity; they do not imply inaccurate HR. Conversely, high signed waveform correlation does not guarantee the same selected dominant HR. All waveform correlations here are computed after the common processing, without a GT-selected sign flip. Reference-aware best-lag/sign metrics remain diagnostic.

## Candidate versus fusion results

Under the **official processed-spectrum** definition, the MMPD-trained model has a correct dominant regional candidate within5bpm on6/12recordings; PURE-trained has11/12. Reference-selected regional oracle MAEs are12.1962 and2.8803 respectively. These oracle values useGT and are not deployable scores. Neither audit has an error>10 with a regional dominant candidate within5; note this threshold does not count the cross model's9.72bpm error on1002.

MMPD-trained outputs fail on all six subject09recordings before and after spectral processing. Its subject10 errors are mostly small. Changing scalar weights among those low-HR outputs cannot reliably select a correct dominant candidate that is absent under the current definition. A spectral/phase-aware combiner could have different behavior, but this audit does not demonstrate that intervention.

PURE-trained weights are close to uniform: average per-record spread between largest/smallest mean weights is about.0145, versus.0991 for cross-trained. Its learned and uniform **HR predictions** are identical on all twelve cases; their waveforms are not exactly identical. A stronger gate is not the first priority suggested by these results.

## Recording901: the key revised diagnosis

This is PURE09-01, reference45.3645bpm. Both models report90.1890bpm after the official evaluation pipeline. The reference autocorrelation estimate is45bpm; fixed GREEN and POS return45.3645. This supports reference HR consistency, without certifying sensor timing or raw reference validity.

| PURE-trained output processing stage | Dominant HR | Second/fundamental **peak-power** ratio |
|---|---:|---:|
| Raw, mean removed |45.3645|.318|
| Tarvainen detrend only |45.3645|.420|
| Detrend + current.75–2.5Hz filter |90.1890|1.567|

Ratios use maximum PSD within broad fundamental/second neighborhoods for this mechanism check; they differ from the audit CSV's integrated±.1Hz-band power ratio. The intra901 integrated second/fundamental ratio is.4255 **before bandpass**, and processed waveform correlation is.9157. The model therefore retained a strong fundamental. The.7561Hz pulse sits just above the.75Hz lower filter cutoff: forward/reverse filtering attenuates it more than the~1.50Hz second harmonic. The reference is also filtered, but its stronger original fundamental remains dominant; its postfilter harmonic/fundamental peak-power ratio is.895. Identical filtering can flip one waveform and not another because their spectral mixtures differ.

For the MMPD-trained901 output, raw second/fundamental peak-power ratio is17.15 and after detrending22.38. It already favors90.189 before the filter; the evaluator is not the sole cause of the cross-trained failure. Lowering the diagnostic cutoff to.65Hz preserves45.3645 in the intra output but leaves the cross output at90.189. This is a post-hoc sensitivity experiment, **not a validated new evaluation protocol or corrected benchmark score**.

Across all twelve cases, raw/detrended regional candidate availability is6/12for cross and12/12for intra; after the official filter it is6/12and11/12. Thus the earlier statement 'no correct regional dominant candidate' must specify preprocessing stage. It does not mean the raw video/model output contains no recoverable pulse. The saved pre-bandpass regional spectra already show several reference-near fundamental peaks in intra901 despite postfilterHR labels near90.

A common detrend-only HR diagnostic gives intraMAE.3150 versus official4.0504; this exploratory number must not replace the official result after observing test failures. Raw/detrended spectra and per-record changes are saved as diagnostic evidence, with the official scores preserved. No labels, cutoff, checkpoint or output waveforms were changed.

![Filter effect on901](../../../diagnostics/pure_matched_comparison_20261125/recording901_filter_effect.png)

![Matched cross/intra and POS](../../../diagnostics/pure_matched_comparison_20261125/matched_results.png)

[Fingerprint and numeric evidence](../../../diagnostics/pure_matched_comparison_20261125/summary.json), [paired records](../../../diagnostics/pure_matched_comparison_20261125/paired_recordings.csv), [raw/processed checks](../../../diagnostics/pure_matched_comparison_20261125/raw_vs_processed_spectra.csv).

## Next steps and scope

1. **Verify evaluation behavior before retraining.** Specify the intended physiological search range, distinguish search bounds from filtering passband edges, and check synthetic known-frequency signals with realistic harmonic mixtures around both edges. Validate any revised common decoder on training/validation data and apply it equally to learned/classical methods. Freeze the protocol before future held-out evaluation; retain current scores as historical comparators. Do not automatically halve peaks or choose per-record cutoffs usingGT.
2. **Audit source-training rate coverage and nuisance conditions.** Derive per-subject HR histograms and timestamp/crop/reference quality from MMPDtraining and PUREtraining/validation. Current evidence points to transfer/learned spectral emphasis; it does not identify rate imbalance or camera representation as the cause. The160frame/30Hz training spectrum uses nfft256: bins within.75–2.5Hz start at49.21875bpm, a further reason to inspect low-rate supervision and leakage around the lower bound on training/validation. Zero-padding increases bin density, not information beyond the clip duration.
3. **Keep POS as a required fixed baseline**, measured on full datasets and the same subject splits, including MMPD where it previously performed much worse. POS success on two subjects is neither universal superiority nor independent GT certification.
4. Once the evaluation protocol is settled, test one controlled rate-coverage/temporal-equivariance or nuisance-robust pulse-path intervention. Hold fusion/architecture/loss changes separate. Rate augmentation and spectral attention already have prior art; novelty needs a narrower demonstrated mechanism and controls. Keep DINOv3/gate redesign deferred until evidence justifies them.

Existing PUREtest has guided development, so use rotated subject evaluations and a future untouched confirmatory dataset for generalization claims. Two subjects are too few for persuasive subject-level uncertainty estimates. These findings support a focused signal/transfer investigation, not an architecture expansion or a claim that physiology is solved.
