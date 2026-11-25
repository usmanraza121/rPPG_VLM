# PURE-trained no-DINO model — result review, 2026-11-25

This run supports the existing pulse path learning substantially better within PURE. Eleven of12reported test recordings have HR errors no larger than.5401bpm. Overall MAE4.0504/RMSE12.9463 remain the headline results: one45.3645→90.1890near-doubling failure accounts for92.22%of total absolute error. Do not remove that failure or present the remaining11MAE.3437 as the main result.

## Verified from the supplied log

Source: workspace `training_PURE_PURE`, SHA256 `20996a7db9d9aa1cbbd6856ad77ed7b55f779298974a2fcfebde3057849f453a`.

Resolved config: PUREfortrain/validation/test; noDINO,learnedfusion,RawRGB/StandardizedPPG,160frames,128x128,30fps,batch16,40epochs,peakLR1e-4. Same750cachedclips now split443train/153validation/154test. New split CSVs have ranges0–.6/.6–.8/.8–1 under `data/file_lists/PURE_RegionalShallowLearned_6_2_2/`; loader semantics imply subjects01–06/07–08/09–10 if all10subjects exist. The log shows these three lists being generated and loaded, but does not supply their contents or hashes. Subject overlap and memberships remain to be independently checked from those CSVs. The progress label 'MMPD validation' is hardcoded trainer wording; the resolved validation dataset isPURE. This is not evidence that validation used MMPD.

Selected zero-based epoch25 by PUREvalidationloss. Correct checkpoint was loaded: `runs/RegionalDINO_20261125_113336/checkpoints/PURE_PURE_RegionalShallowLearned_6_2_2_best.pth`. Fortyepochs contain28trainingbatches each,1120optimizerupdates versus14280in the MMPD run. Holding epochs/LR/schedule shape fixed did not equalize the number of updates.

## Metrics and failure pattern

| Measure | Result |
|---|---:|
| Test recordings |12|
| MAE (bpm) |4.0504|
| RMSE (bpm) |12.9463|
| MAPE (%) |8.7004|
| HR Pearson |.7486|
| Legacy SNR (dB) |9.9931|
| Within1bpm / within5bpm |11/12 /11/12|

Seven recordings have about.5401bpm error and four have zero HR-bin error. These are spectral HR estimates; exact bin agreement does not establish beat timing or reconstructed waveform morphology. One low-HR recording is nearly doubled:45.3645→90.1890,error44.8245bpm. Its array index is6(zero-based); the log does not identify the recording. Do not assign a subject/trial solely from array position. A secondary-harmonic selection is plausible, but extraction versus fusion versus evaluation peak selection needs regional spectra and saved waveforms.

The12test references cover45.36–100.45bpm: six<60,four60–80,one80–100,one≥100. The single100.45bpm case is not adequate validation of the120–135bpm failures seen in other runs. Six low-HR cases include five accurate results and one large error; this is improved low-rate behavior, not universal harmonic robustness. Two expected test subjects cannot establish general population robustness or a contribution claim.

## Learning behavior

| Metric | Epoch0 validation | Selected epoch25 validation |
|---|---:|---:|
| Total loss |1.5805|.8414|
| Mean clip waveform correlation |.0980|.5779|
| Diagnostic clip HRMAE |36.8015|5.2390|

Selected-epoch training waveformcorrelation=.6840; finalepoch=.6918. Validation waveformcorrelation ends=.5752, close to the selected result. Validationloss broadly plateaus after~20–25epochs with mild fluctuations, not wholesale failure to learn. Selected rawMSE=.9382 versuszero=.9997; predictionstd=.05255versustarget=.97306. Raw waveform amplitude remains weak even while most whole-record HR peaks are accurate. This further shows that low output amplitude alone did not explain the previous HR failure. Do not rescale/z-score outputs and call it restored physiology.

Gateentropy at selectedvalidation=2.1963 versusmaximumln9=2.1972, close to uniform. This could mean similar candidates or weak gate discrimination; it does not prove learnedfusion contributes. Compare its same-candidate uniform control in the audit.

![PURE learning and held-out HR](../../../diagnostics/pure_intra_log_20261125/learning_and_hr.png)

[Full parsed epochs, arrays and source fingerprint](../../../diagnostics/pure_intra_log_20261125/summary.json).

## Interpretation and next action

The model can learn useful waveform timing and HRwithinPURE with its existing raw-input/standardized-label setup. This supports prioritizing transfer, trainingHRcoverage, nuisance/preprocessing differences and reference alignment over addingDINOv3 or a larger network. It does not isolate which cause matters: train distributions, subjects, rate coverage, recording conditions and optimization budgets differ.

Do not call18.26→4.05a matched cross/intra gain:18.26used59PURErecordings;4.05uses12. Their GTvalues match the first6/last6entries of the earlier log, which is consistent with loader order, but GTmatching does not independently certify recording identities. Obtain a matched comparison with the exact test CSV.

Next, audit both trained checkpoints on the **same PUREtest split**, no retraining or model changes. From the server `rPPG_VLM/`directory, with the config and existing split CSVs copied/available:

```bash
python tools/audit_regional_dino.py \
  --config configs/train_configs/intra/PURE_PURE_RegionalShallowLearned.yaml \
  --checkpoint runs/RegionalDINO_20261125_113336/checkpoints/PURE_PURE_RegionalShallowLearned_6_2_2_best.pth \
  --split test \
  --save-waveforms \
  --output diagnostics/pure_intra_test_full
```

```bash
python tools/audit_regional_dino.py \
  --config configs/train_configs/intra/PURE_PURE_RegionalShallowLearned.yaml \
  --checkpoint runs/RegionalDINO_20261124_121357/checkpoints/MMPD_PURE_RegionalShallowLearned_best.pth \
  --split test \
  --save-waveforms \
  --output diagnostics/pure_cross_matched_test_full
```

The model architecture arguments are identical, so the audit can use the same test configuration with either checkpoint. A shared config does not change either checkpoint's historical training dataset; label outputs by actual training source. Both commands use `--split test`; the current valid-split disjointness helper is MMPD-specific and should not be used to certify PUREvalidation. Compare the two manifests' test-list hashes, recordingIDs, referenceHRvalues, evaluated frames and evaluator settings. Inspect all three source CSVs separately to verify actual6/2/2subjectmembership and no overlap; the test-only audit does not check training membership.

These audit scores use the same Welch/powerSNR protocol for both models, which differs from the legacy training-log FFT/SNR report. Keep the two metric protocols separate. The audit will identify the doubled recording, whether a correct regional fundamental exists, learned-versus-uniform differences and whether the cross-trained failure persists on these exact records. Reference-based oracle/peak comparisons are diagnostic only. Preserve the outlier and do not tune a half-frequency rule from this test example. No new model change or training run is justified before the matched evaluation.
