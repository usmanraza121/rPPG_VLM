# First paired-rate augmentation experiment

Prepared2026-11-25. This is an opt-in, **slowing-only** training ablation on MMPD. It tests whether adding lower-rate appearances across more source identities helps. It does not test a new backbone, a new loss, bidirectional rate invariance or a proven scientific contribution.

## Implementation and controls

New package `neural_methods/augmentation/paired_rate.py` samples `speed` in[.6,1.0] and reads both video andPPG at identical coordinates `start + speed*i`. Source span is inside the same cached clip. Output remains160frames at30Hz; no extrapolation, wrap, tiled beats, invented boundary frames or FPS-metadata substitution. Slower reads interpolate/upsample, so this experiment does not perform a decimation needing an anti-alias prefilter. It also smooths video and slows motion, and uses a shorter original interval; gains cannot be uniquely attributed to pulse-rate invariance.

Probability of attempting a transform is.5. A source-training label proxy uses a Hann-windowed in-band FFT plus autocorrelation agreement to skip flat/ambiguous/boundary clips. It samples uniformly within admissible speed bounds so expected transformed proxyHR stays50–140bpm. These are conservative bounds, not exact physiological guarantees:160frames contain only5.33seconds and zero-padding does not remove spectral uncertainty. Actual application rate is below.5 when references are ineligible. Skipped clips remain in training unchanged; this is not data deletion or label repair.

RawRGB/StandardizedPPG remains unchanged. Interpolated reference amplitudes are retained without new per-clip normalization. Model architecture, objective, baseline augmentationflag, training split, batch16,40epochs,OneCyclepeak1e-4 and validation-loss checkpoint selection stay fixed. Number of optimizer steps remains40×357=14280 for the same5709samplecache. An independentCPUgenerator(seed1701) leaves model/dropout/data-order RNG streams untouched by augmentation draws. Training records exact sourceclip/speed/start/eligibility in `rate_augmentation.jsonl`, aggregate effective rates in `regional_history.jsonl`, and augmentation settings/RNGstate in best/last checkpoints. Existing trainer does not implement resume; saved state is provenance, not a claimed tested resume path.

`config.py`adds disabled-by-defaultsettings; `RegionalDINOTrainer.py`invokesaugmentation only on trainingbatches whenenabled. Sourcevalidationprogresslabel was corrected fromhardcodedMMPDwording. Existing no-augmentation configs remain disabled. Model/loss/officialevaluator source files were not changed. The augmented checkpoint uses a unique `MMPD_PURE_RegionalShallowRateAug` name; baseline checkpoints are not reused as initialization. Train from scratch to compare with the existing baseline.

## Server transfer

Copy these paths from this repository to the same server repository paths, or extract the supplied update ZIP **inside `rPPG_VLM/`**:

- `config.py`
- `neural_methods/trainer/RegionalDINOTrainer.py`
- `neural_methods/augmentation/__init__.py`
- `neural_methods/augmentation/paired_rate.py`
- `configs/train_configs/ablations/regional_shallow_rate_aug.yaml`
- `tools/check_rate_augmentation.py`

The archive also includes tests, the raw PURE timing tool and this note. Existing regional/audit modules and base YAMLs are reused; no dependency installation or preprocessing rebuild is required.

## Commands

From server `rPPG_VLM/`, first run the source-cache mechanical check:

```bash
python tools/check_rate_augmentation.py \
  --config configs/train_configs/ablations/regional_shallow_rate_aug.yaml \
  --output diagnostics/rate_aug_preflight
```

This reads64evenlydistributed source-training clips, uses CPU, checks known-signal joint interpolation, verifies dimensions/finiteness and reports application/rejection plans. It reads no validation/test arrays, builds no model and changes no cache. It refuses an existing output directory and stops if no sampled transform is eligible. Inspect `summary.json`; numerical success is a mechanics check, not physiological validation.

If that command succeeds, start the training ablation:

```bash
python main.py --config_file configs/train_configs/ablations/regional_shallow_rate_aug.yaml
```

The inherited pipeline automatically tests the existing fullPUREtarget after selecting the MMPDvalidation-bestcheckpoint. Those target results remain exploratory because PURE has already informed development; do not choose the checkpoint on target performance. Share the new `training.log`, `regional_history.jsonl` and `rate_augmentation.jsonl` after completion. A later source/matched-target audit will compare HRranges/subjects and augmentation behavior; no performance improvement is promised.

## Local validation

Five new tests passed for sharedcoordinates/frequency response, speed1identity/probability0bypass, RNGisolation/deterministicplans/state roundtrip, flat/invalidreference handling, and trainer training-only application/stepcount/checkpoint provenance. Seven existing trainer tests passed. A synthetic cached-data CLIpreflight applied3of8clips and verified all shape/finiteness checks. Modelargs/data/lossweights/schedule match baseline config. FullserverMMPDcache preflight and realtraining have not been run locally. Additional training memory comes from a copied batch and per-clip interpolation; retain batch16 if possible for the matched comparison.

## Raw PURE access and timestamp finding

User supplied `F:/Datasets/PURE`; workspace `datasets/PURE - Shortcut.lnk` is aWindows shortcut. Directrawfolder access succeeded. A separate read-only audit checked all59recordings andall10subjects: PNGtimestamp lists exactly match `/Image`JSONtimestamps. Recording06-02 is absent, consistent with59ratherthan60recordings. EffectivevideoFPS29.8477–30.0000.

The existing PUREloader ignores timestamp coordinates when it maps PPG by samplecount. Timestamp-based interpolation differs from that legacy map by up to550ms in recording04-01 (RMS242ms), and367.8ms in06-01. Acrossrecords the two reconstructed reference waveforms have median correlation.9444,minimum.0998(07-01). These are interpolation-coordinate differences, **not proof of a measured camera–PPG offset or wrongGT**. Device flags occur in05-02(259samples)and08-06(62samples); no automaticexclusion/relabeling was done. Noimages were decoded in this timing scan, and no rawfiles/caches were modified.

Results are in workspace `diagnostics/pure_raw_timing_20261125/{summary.json,recordings.csv}`. Timestamp/cache rebuilding is a separate controlled preprocessing experiment; do not combine it with this MMPDaugmentation comparison. RawMMPD is not required locally for this run because it uses the servercache, but later would help inspect the flagged referenceclips/timestampquality.
