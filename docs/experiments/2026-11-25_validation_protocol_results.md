# Validation protocol and source-rate results

Reviewed 2026-11-25. Supplied artifacts: workspace `pure_validation_protocol/` and `mmpd_validation_protocol/`. Both used existing checkpoints; neither trained nor loaded test waveforms. Actual split CSV membership is now verified in the manifests. Model/loss/trainer/official evaluator were not changed during this review.

## Decision

Keep the current common HR protocol as the primary comparator for the next training ablation. The fixed alternatives do not improve learned-model validation performance on either dataset. Retain the documented filter-edge mechanism as a sensitivity limit rather than replacing the evaluator after seeing a test example.

The next **proposed** training intervention is paired temporal-rate augmentation on MMPD training data, with architecture, losses, schedule and evaluation held fixed. This is a mechanism test, not a contribution claim or guaranteed improvement. It has not been implemented or launched. Prefer it over naive rate balancing because low-rate recordings come from very few identities; changing training speed can expose more identities at different temporal rates. It also changes motion dynamics and interpolates frames, so these confounds require explicit controls and mechanical validation.

## Protocol validation

| Learned model, validation MAE (bpm) | Current | Detrend only | Wider .6–3.3Hz filter |
|---|---:|---:|---:|
| PURE-trained,12PUREvalidation recordings |.3150|.5401|.3600|
| MMPD-trained,137MMPDvalidation recordings |8.5660|9.3425|9.0942|

All candidate predictions use the same45–150bpm search and one fixed current-protocol referenceHR for scoring. This prevents moving the reference along with a candidate decoder. It does not certify that the current reference decoder is physiologically correct in every case. MMPD's current-protocol results reproduce the prior no-DINO audit exactly. PUREvalidation is subjects07/08, not the earlier test subjects09/10.

PURElearned/uniform both achieve.3150MAE with all12within5 under the current pipeline. POScurrent4.4554→.5851detrend/.4050wider, and CHROM4.3654→.2700/.4050. Thus alternative filtering helps certain classical outputs but not the learned model on aggregate. The currentPOS failure is704(reference124.75,prediction74.53); currentCHROMfailure801(reference46.44,prediction91.81). Changing a protocol per method or per recording would defeat a common comparison.

On MMPD, current filtering also beats alternatives for POS, CHROM and GREEN. Learned high-HRMAE remains25.048current,25.305detrend,25.074wider. The source high-rate failure therefore persists across these decoders; it is not mainly a filter-edge artifact. No universal filtering fix is supported. Preserve the prior recording901 test diagnosis: raw/detrendedPUREoutput contains45.36; currentbandpass can promote its second harmonic. Dataset-average validation rankings do not invalidate that individual mechanism.

## Verified subject membership

PUREactual CSVs contain training01–06,validation07–08,test09–10, with no overlap. This resolves the earlier missing-CSV-membership uncertainty; the testCSV hash matches the matched-test audits. Source MMPDtrain1–26/validation27–33 also match previous file-list hashes and checkpoint. Test waveforms were not read by either diagnostic. MMPDconfig's test metadata names allPUREsubjects because that historical run tested the full target dataset; it is not part of validation inference here.

## Training rate support

| Training clip category | PURE | MMPD |
|---|---:|---:|
| <60bpm |152/443(34.3%)|422/5709(7.4%)|
| 60–80 |272|2388|
| 80–100 |19|2017|
| ≥100 |0|875(15.3%)|
| UndefinedHR |0|7|

Whole-record proxies: PURE35recordings,13below60 and22at60–80; MMPD519recordings,34below60,223at60–80,203at80–100,59at≥100. These are HR estimates from preprocessed references, not independently verified clinical HR. Clip/record estimates differ with nonstationarity, window resolution and noise.

All34low-rate MMPDtraining recordings are concentrated in subjects6(9),11(11),12(14). Below50bpm there are only four MMPDrecordings from one subject, versus twoPURErecordings from one subject. MMPDlow-rate clips exist; they are not absent. The useful hypothesis is limited frequency/identity/condition diversity, not a proven missing-HR cause. A uniform four-bin sampler would heavily repeat those three low-rate identities and could create an identity shortcut.

Substantial high-rate support exists in MMPD:875trainingclips and59recordings≥100;9recordings≥120 across4subjects. Lack of high-rate examples alone cannot explain its validation failure. High-HR validation occurs in stationary(n12,MAE23.76) and walking(n9,26.76) conditions, so the error is not confined to walking. Across all rates walkingMAE16.34 versusstationary8.81 is descriptive and confounded by subject/rate distribution, not causal motion evidence.

## Important generalization observation

PUREtraining has **no estimated clips≥100bpm** and whole-record training HR spans45.90–79.93. Nevertheless, its six subject07validation recordings have references124.75–134.47 and current-protocol MAE.5401bpm. The six low-rate subject08validation recordings haveMAE.0900. These validation subjects selected the checkpoint; this is not an untouched high-rate test or a clinical generalization claim. It does demonstrate that the architecture need not predict only rates observed in training, weakening a simple 'model memorized the training HR range' explanation. It can follow unseen rates in favorablePUREconditions; MMPDsignal quality, nuisance structure, reference quality and training dynamics remain plausible contributors.

## Reference and synthetic checks

Seven MMPDtraining clips with undefined HR all belong to `subject10_L4_MO1_E2_S4_GE2_GL2_H2_MA2`,clips1–7. Inspect their original PPG/cache quality before treating them as clean supervision; undefined estimatedHR does not by itself authorize deleting or rewriting labels.

Changing reference preprocessing shifts HR by>5bpm on17/519MMPDtraining recordings and101/5709trainingclips. It affects fiveMMPDvalidation records with detrend-only, threewithwider filtering; PURErecord-reference changes>5 arezero, with sixtrainingclip changes. This establishes some spectral ambiguity, not ground-truth error. Rate-stratification weights should not blindly rely on unstable clip peaks. Preserve original references and log uncertainty.

Both synthetic CSVs are byte-identical outputs of the same deterministic480case sweep, not two independent experiments. Pure sinusoids are recovered within.405bpm for all three processing choices. For64second mixtures with second-harmonic amplitude≤.65, the current pipeline has10/64errors>5bpm, widerfilter4/64, detrend-only0/64; these errors concentrate near45–48bpm. Equal-amplitude mixtures have additional peak ambiguity. These controlled cases corroborate edge/harmonic sensitivity, but omit real motion/noise and cannot rank real-world protocols by themselves. Short160framewindows and long64secondwindows were both included.

![Validation protocol and training rates](../../../diagnostics/validation_protocol_review_20261125/validation_and_rate_coverage.png)

[Fingerprinted numeric summary](../../../diagnostics/validation_protocol_review_20261125/summary.json), [MMPDsubject-by-rate support](../../../diagnostics/validation_protocol_review_20261125/mmpd_subject_rate_support.csv).

## Next controlled experiment specification

1. Keep the no-DINO learned-fusion model, rawRGB/standardizedPPG representation, current primary evaluator, existing subject splits, optimizer, losses and checkpoint-selection rule. Do not upgrade DINO or add a spectral gate simultaneously.
2. Add **training-only paired temporal-rate augmentation**, changing video and reference together. Use bounded transforms and retain transformed rates inside the declared HRsearch range; verify actual timestamps/effective rate and joint phase alignment with known signals. Handle fixed-length clips without invented cyclic boundaries or mislabeled padding; accelerating requires appropriate resampling/anti-aliasing. An independent augmentation RNG should preserve baseline data ordering. Define augmentation probability/range before the run, based on training support rather than PUREtest error.
3. Match model initialization, number of optimizer updates, schedule and seed to the baseline. Treat variable output length, overlap sampling or longer context as separate changes if used. Do not add an equivariance loss in the same first ablation; augmentation alone is the causal factor.
4. Evaluate sourcevalidation by HR/subject/condition and preregister a target-domain development protocol. PURE has already guided development, so target gains there remain exploratory; use future untouched data and repeated matched seeds for a contribution claim. Spectral-rate augmentation already has close prior art in the reviewed corpus.

A no-retraining rate-equivariance probe can be used as a mechanical check during implementation, measuring whether the frozen model follows a known joint video/PPG speed change. It should precede—not replace—the controlled training comparison. The purpose is to test rate sensitivity across identities; it will not by itself distinguish pulse from co-transformed motion.

These results are sufficient to move beyond repeated aggregate audits. The proposed experiment is the next research step, with the decoder policy frozen and the remaining reference-quality cases explicitly tracked.
