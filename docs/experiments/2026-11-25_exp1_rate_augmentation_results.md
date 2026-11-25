# Exp1 paired slowing: transfer gain with a rate trade-off

Reviewed 2026-11-25. Inputs: `Exp_data/exp1/{training.log,regional_history.jsonl,rate_augmentation.jsonl,regional_waveforms.pt}`. This review adds derived artifacts and documentation; training code, evaluation code, labels and supplied artifacts are unchanged.

## Result

The first slowing-only augmentation improves overall MMPD-to-PURE HR estimation, particularly at low rates, but worsens high-rate target errors and source-validation clip metrics. This single run supports further investigation; it does not establish a generally better model or a scientific contribution.

| Full 59-recording PURE test | No-DINO baseline | Exp1 slowing |
|---|---:|---:|
| MAE (bpm) | 18.2611 | 13.5837 |
| RMSE (bpm) | 26.1051 | 22.8694 |
| MAPE (%) | 30.6074 | 20.9427 |
| HR Pearson | .0843 | .2679 |
| Within 5 bpm | 27/59 | 35/59 |
| Errors over 20 bpm | 25/59 | 18/59 |

MAE improves by **4.6774 bpm (25.6%)**. Legacy printed SNR improves from -3.368 to -1.965 dB; it uses a different definition from the audit's power SNR. The historical printed RMSE uncertainty is not treated as a credible confidence interval.

| Reference HR (bpm) | Recordings | Baseline MAE | Exp1 MAE |
|---|---:|---:|---:|
| <60 | 25 | 31.1071 | 19.1611 |
| 60–80 | 26 | 1.7032 | 1.1009 |
| 80–100 | 1 | .5401 | 0 |
| >=100 | 7 | 36.4151 | 41.9699 |

High-rate error increases by **5.55 bpm**. Subject07 has Exp1 MAE 47.345 bpm: one high-rate recording is accurate, while several estimates fall toward 63–83 bpm. This is consistent with an asymmetric rate-exposure effect, but slowing also changes motion, interpolation smoothing and available context. It does not isolate pulse-rate exposure as the sole cause.

## Run and augmentation verification

Resolved configuration matches the baseline for data lists/paths, Raw RGB, standardized PPG, model/loss, batch size 16, 40 epochs, peak LR 1e-4 and source-validation-loss checkpoint selection. The intended difference is RATE_AUG, plus experiment names and base-YAML lineage. There are 62,800 trainable parameters and 5,709/1,507/750 training/validation/test clips.

The run selected and loaded zero-based epoch 39:
`runs/RegionalDINO_20261125_182240/checkpoints/MMPD_PURE_RegionalShallowRateAug_best.pth`.
The baseline selected epoch 37 using the same rule. Selection at the last epoch alone is not evidence that longer training will fix the trade-off.

The training log SHA256 is `ddb3308aefed6aea7e2ef6a165d2cf4c342b93e22f28f4eb506f10bd21871282`. All 59 reference HR values match the baseline log, and saved Exp1 arrays reproduce its reported MAE. Exact baseline/Exp1 reference-array identity is independently verified for the historical twelve recordings. The remaining full-test pairing uses consistent log order and reference values; no baseline per-record identity manifest or full cache hashes were exported in the supplied log.

All **228,360 training presentations** (40 x 5,709) are logged, with 5,709 unique recording/clip keys per epoch:

- 107,800 augmented: **47.206%**.
- 114,534 skipped by probability.
- 6,026 skipped by reference/speed guards: 877 without autocorrelation peaks, 2,408 reference disagreements, 1,033 without admissible speed, 1,708 boundary peaks.
- Zero logged speed, coordinate or expected-rate bound violations; no duplicate epoch/clip keys.
- Mean accepted speed .8297; range approximately .6000–1.0000.

On accepted transforms, mean source/expected HR proxies are 83.10/68.39 bpm. Low-rate proxy events increase from 6,881 to 36,332; high-rate events decrease from 16,767 to 4,034. Generated low-rate views involve all 26 training identities. These counts represent repeated synthetic presentations, not new independent physiological recordings. Safe coordinates establish indexing correctness; planned HR proxies do not establish physiological validity or actual recovered pulse.

## Source validation and waveform findings

| Selected-checkpoint MMPD validation **clip** metric | Baseline epoch37 | Exp1 epoch39 |
|---|---:|---:|
| Loss | 1.3192 | 1.3731 |
| Diagnostic clip HR MAE (bpm) | 12.4730 | 15.5679 |
| Mean waveform correlation | .2885 | .2710 |
| Raw MSE | .9868 | .9911 |

These are source-validation clip diagnostics, not recording-level audit errors. **Exp1 full-record MMPD validation is still missing.** Augmented training losses use altered examples and should not be compared directly with baseline training losses.

Saved PURE outputs were reassembled in numeric, contiguous clip order and analyzed under the existing common pipeline. No new server PURE audit is needed:

- Learned fusion MAE is **13.5837**, versus **14.4442** for uniform fusion of identical candidates.
- Processed mean signed waveform correlation is only **.0660**; improved HR does not establish waveform, phase, amplitude or HRV fidelity.
- 37/59 recordings have an individual processed regional dominant HR within 5 bpm. Reference-selected best-region MAE is 9.3914, an oracle diagnostic only.
- Of 22 fused errors over 10 bpm, 20 lack a correct processed dominant regional candidate. Candidate extraction/spectral behavior remains a major limitation. This does not prove the raw fundamental is absent, given the previously demonstrated filter-edge effect.

For the twelve historical recordings from subjects09/10, all saved reference arrays exactly match the baseline audit. MAE improves **20.7021 to 17.1017**, while candidate availability remains **6/12**. All six subject09 recordings still lack a correct processed dominant regional candidate. Some wrong predictions move closer without recovering the correct pulse: recording903 changes 93.43 to 69.67 bpm against 47.52; recording905 changes 93.97 to 78.85 against 47.52. Recording901 remains 90.19 against 45.36. Subject10 MAE remains 1.8902. Processed waveform correlation remains weak (.0493 baseline, .0657 Exp1).

![Target rate gains and source-validation trade-off](../../../diagnostics/exp1_review_20261125/rate_gain_and_source_tradeoff.png)

[Numeric summary and fingerprints](../../../diagnostics/exp1_review_20261125/summary.json), [recording analyses](../../../diagnostics/exp1_review_20261125/model_recordings.csv), [regional analyses](../../../diagnostics/exp1_review_20261125/regional_candidates.csv), [matched twelve](../../../diagnostics/exp1_review_20261125/matched12_recordings.csv).

## Next action

Audit the Exp1 checkpoint on **full MMPD validation** using the existing protocol. Run from the server repository:

```bash
python tools/audit_regional_dino.py \
  --config configs/train_configs/ablations/regional_shallow_rate_aug.yaml \
  --checkpoint runs/RegionalDINO_20261125_182240/checkpoints/MMPD_PURE_RegionalShallowRateAug_best.pth \
  --split valid \
  --save-waveforms \
  --output diagnostics/mmpd_exp1_rate_aug_full
```

Compare membership and source-file hashes with `mmpd_no_dino_full/`, then inspect HR bins, subjects, candidate availability and raw-versus-filtered fundamental/harmonic behavior. Baseline recording MAE is 8.5660 bpm; compare it only with an equivalent recording-level result.

Keep architecture, evaluator and augmentation strength fixed until this comparison is available. If the trade-off is confirmed, the next training comparison can test one less aggressive slowing setting or a balanced bidirectional design with sufficient **real contiguous source context**. Faster reads cannot be implemented correctly by wrapping or padding the 160-frame cache. Do not combine augmentation changes or select them using the repeatedly examined PURE test score. Predefine development criteria and reserve fresh data and matched seeds for confirmatory claims. A timestamp-based label reconstruction remains a separate preprocessing experiment.
