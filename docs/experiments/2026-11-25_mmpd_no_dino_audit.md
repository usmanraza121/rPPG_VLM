# Matched MMPD audit and intra-dataset decision — 2026-11-25

The new no-DINO audit does not show a meaningful aggregate HR advantage for either architecture. It confirms that high-rate extraction failures also occur within MMPD, while the poor PURE result still warrants a controlled PURE-trained diagnostic. Training within PURE is now a reasonable next experiment before redesigning the network.

## Comparison provenance

New input: workspace `mmpd_no_dino_full/`; comparison: `diagnostics/mmpd_full/`. Both use epoch37, the same train/validation file-list hashes, training subjects1–26, audited subjects27–33,137recordings,241120frames,zero omitted frames, whole contiguous recordings, Tarvainen100/order1bandpass/Welch3333 and the same powerSNR definition. Architecture arguments differ only in `use_dino`. Checkpoints have different hashes, as expected. Reference-check CSVs are byte-identical;822paired fixed classical-control results have zero AE difference. This verifies a strong matched evaluation, not identical stochastic training or repeated seeds. The audited seven subjects were used for checkpoint selection, so this is source-validation diagnosis, not independent source-test evidence.

## Results

| Measure | DINO context | No DINO |
|---|---:|---:|
| Learned MAE (bpm) |8.6724|8.5660|
| Learned RMSE (bpm) |14.7486|13.5393|
| HR Pearson |.3863|.4981|
| Within5bpm |88/137|83/137|
| Median absolute error (bpm) |2.1602|3.7804|
| Mean signed waveform correlation |.4026|.3565|
| Uniform, same model candidates, MAE |10.0127|9.3228|
| Fixed center candidate, descriptive MAE |8.5660|8.6685|
| GT-selected regional oracle MAE |4.2652|4.2455|
| No region dominant HR within5 |29|27|

No-DINO MAE differs by-.1064bpm (~1.2%). It wins43recordings, loses55 and ties39. Subject changes are mixed: no-DINO helps subject30 by4.293bpm but hurts subject28 by2.862. An exploratory10,000replicate subject-cluster bootstrap gives no-DINO-minus-DINO MAEdifference95percentile[-1.765,1.288]bpm; seven clusters provide limited precision, and this does not account for training-seed/model-selection uncertainty. Neither architecture wins across all waveform/HR metrics.

Both learned gates help their own candidates: DINO improves on uniform by1.3403bpm, no-DINO by.7569bpm. Candidate extractors were trained separately, so these gains are not an isolated measurement of DINO's gate contribution. The no-DINO center8.6685 versus fused8.5660 is also a small descriptive difference. Do not choose a fixed region from these validation outcomes and call that a general solution.

## Failure mechanism

| Reference HR | Recordings | DINO MAE | No-DINO MAE | No-DINO missing correct dominant candidate |
|---|---:|---:|---:|---:|
| 60–80 |37|1.737|2.613|1|
| 80–100 |79|7.212|6.973|10|
| ≥100 |21|26.385|25.048|16|

No-DINO predictions range59.95–102.07bpm despite references up to126.91. In the high-HR bin its signed bias is-24.997bpm. This is HR-range compression within the source dataset as well as poor transfer; it cannot all be attributed to MMPD/PURE differences. Its causes remain unproven: rate coverage, visual signal quality, motion/illumination, timing, learning bias or extraction can be involved.

There are43no-DINO fused errors>10bpm:19have an individual dominant regional candidate within5,24do not. DINO has39such errors:13with/26without a correct candidate. Thus fusion and extraction both fail. Twenty-four recordings lack a correct dominant candidate in **both** models; no-DINO recovers five former failures but loses three former successes. Of27no-DINO missing-candidate cases,21have a reference-near frequency among a region's stored top3peaks (11/16high-HR cases). Those are GT-assisted diagnostics, not proof of clean physiology or inference-time peak selection.

No learned near-double/near-half cases are flagged; only8/1233regional outputs have second/reference band power>1. A blanket second-harmonic correction is not supported. Worst-case plots show dominant~70bpm components while reference peaks are~111–112bpm, sometimes with weak reference-near secondary energy. Classical agreement at a different rate is not proof the reference is wrong. The waveform displays are z-scored/PSD-normalized for viewing, not an amplitude recovery result.

![Paired MMPD HR results](../../../diagnostics/dino_ablation_comparison_20261125/comparison.png)

[Fingerprint, paired uncertainty and numeric summary](../../../diagnostics/dino_ablation_comparison_20261125/summary.json), [paired records](../../../diagnostics/dino_ablation_comparison_20261125/paired_records.csv).

## Next experiment: PURE-trained subject-disjoint diagnostic

The user's intra-dataset proposal is useful **now**, as PURE→PURE with an unchanged model. Repeating the existing MMPD1–26training/27–33validation and relabelling it intra-dataset would not add evidence. A separate MMPDtest split would require a new training/validation/test partition and retraining; PUREtraining more directly addresses why PURE currently fails.

Start with the no-DINO learned-fusion model for lower cost; this is a development choice, not proof DINO is unnecessary. Keep pulse path, losses,160frames/30fps,RawRGB128²,standardizedlabels,batch16,peakLR1e-4,40epochs and the validation-loss checkpoint rule. Reuse the same PUREraw cache/representation as the cross-dataset test, with separate subject-list files and outputs. Training data changes, so the number of optimizer updates under40epochs changes; record that and do not describe optimization budgets as identical.

Before training, enumerate available PUREsubjects and choose a fixed subject-disjoint split (e.g.6train/2validation/2test if all10subjects are available), retaining all sessions/tasks of a person together. Freeze IDs before examining results; inspect **training/validation** rate coverage and reference quality, and predefine HR-bin reporting so absent training rates are not mistaken for visual-domain failure. Subject count/rate confounding limits any single split. If promising, rotate held-out subject pairs with validation nested inside each training fold rather than choosing the most favorable split.

Train from scratch on PUREtrain, select checkpoint using PUREvalidation only. Evaluate the existing MMPD-trained no-DINO checkpoint and the new PURE-trained model on **exactly the same held-out PUREtest recordings**, using identical reconstruction, band, estimator and candidate diagnostics. Their paired difference estimates the consequence of changing training data/domain under that protocol; it does not isolate camera representation from training HR/subject coverage. Do not compare a new12recording subset directly with the earlier59recording18.26aggregate.

If PUREtraining yields good held-out HR/waveforms while MMPDtraining fails on those same subjects, prioritize transfer, reference/preprocessing differences and source HR coverage. If both fail, test pipeline integrity and pulse extraction with a known baseline on the same split; a failed PURErun alone does not prove architecture failure if coverage/reference/optimization are inadequate. If both work only at midHR, prioritize rate-conditioned diagnostics. Preserve a future untouched confirmatory dataset because PURE has already informed development.

`dataset/data_loader/PURELoader.py` groups by integer subject ID for fraction-based splitting, but cache-derived list generation must still be verified. Do **not** directly reuse `configs/train_configs/intra/1PURE_SSMamba.yaml`: it is an old only_test/SSMamba configuration with Standardized **video** inputs and identical validation/test ranges. The regional model requires Raw video, three disjoint subject groups and its own model/loss configuration. The PURE training configuration was subsequently prepared at `configs/train_configs/intra/PURE_PURE_RegionalShallowLearned.yaml`; model/loss/trainer code remains unchanged. Its actual resolved configuration and PURELoader subject-splitting function were checked locally. No server training job has been launched.

## Prepared PURE training command

Copy `configs/train_configs/intra/PURE_PURE_RegionalShallowLearned.yaml` to the same path in the server repository, then run from `rPPG_VLM/`:

```bash
python main.py --config_file configs/train_configs/intra/PURE_PURE_RegionalShallowLearned.yaml
```

The configuration inherits the no-DINO learned-fusion experiment:40epochs,batch16,peakLR1e-4,same pulse path/loss/checkpoint policy. It changes the dataset to PURE for all splits, reuses `/raid/muhammadu/pre_proc_datasets/MMPD_PURE_cross_VLM/PURE_test/Data_cross_MMPD_PURE`, keeps `DO_PREPROCESS:False`, and generates distinct CSVs under `data/file_lists/PURE_RegionalShallowLearned_6_2_2/`. It requires the existing server raw-data directory for subject enumeration when these new CSVs are generated. With all ten PURE subjects present, sorted IDs01–06 train,07–08 validate,09–10 test. All sessions/chunks follow subject membership. Do not reuse or hand-copy the full59recording CSV into these split paths. The final test loads this run's validation-best checkpoint automatically.

Local validation confirms no DINO, learned fusion, RawRGB/StandardizedPPG,160frames,30fps,128x128 and separate subject partitions/list paths. Server data content/availability cannot be checked locally. The labels and cached arrays are reused without modification.
