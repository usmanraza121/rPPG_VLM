# Diagnose a saved RegionalDINO checkpoint

Use this sequence to distinguish waveform estimation, regional fusion, and data-quality problems before another architecture change. The checkpoint audit performs inference and writes diagnostic artifacts; it does not retrain the model or replace ground-truth PPG. Use the training Python environment with PyTorch, NumPy, SciPy, YACS, PyYAML, tqdm, and Matplotlib installed. DINO-enabled inference needs access to the official hub code and pretrained backbone weights if they are not already cached. Run commands from the `rPPG_VLM` repository root on the host that has the cached MMPD data and saved checkpoint.

## 1. Check a small MMPD validation subset

Keep the original model configuration when loading the checkpoint. In particular, do not load a learned-fusion checkpoint into a separately configured uniform-fusion model: use the audit's learned-versus-uniform comparison on the same returned regional waveforms.

```bash
python tools/audit_regional_dino.py --config configs/train_configs/cross/MMPD_PURE_RegionalDINO.yaml --checkpoint runs/RegionalDINO_20261123_235243/checkpoints/MMPD_PURE_RegionalDINO_best.pth --split valid --device cuda:0 --batch-size 8 --max-recordings 4 --output diagnostics/mmpd_smoke
```

First check the printed checkpoint epoch and source split, then review the manifest's cached file list and evaluated-recording count. The tool checks raw RGB scale and numeric clip order as it reads each pair; inspect a real cached input/label pair if that audit fails. Check the recording names in the output tables against the intended run. The checkpoint and cache paths are relative to the current repository unless absolute paths are supplied. `cuda:0` is the first GPU visible to the process; set `CUDA_VISIBLE_DEVICES` before launching Python when a specific physical GPU is required.

Recordings are ordered deterministically; the first four are used by `--max-recordings 4`. MMPD validation audit also checks that training and validation subject IDs do not overlap.

The default `--window-seconds 0` evaluates each full contiguous recording span. It is not a single 160-frame clip. Missing chunks must not be joined as though they were adjacent. A diagnostic `--window-seconds 10` creates shorter evaluation windows and therefore changes the protocol; label those results separately and do not compare them directly with full-recording scores. Trailing windows shorter than one second are skipped; a retained shorter remainder is evaluated separately, so review the `duration_seconds` column.

The common HR pipeline uses the legacy smoothness-prior detrend with lambda 100 (computed by a sparse solve), a first-order zero-phase physiological bandpass, and Welch PSD with at least 3333 FFT points (approximately 0.54 bpm bin spacing at 30 Hz). Welch segment length is `min(256, T)` rather than the legacy `min(256, T-1)`, a small difference on short inputs. Harmonic diagnostics also examine un-bandpassed spectra, and power SNR uses `10*log10`, while the historical evaluator uses a different spectrum and dB scale. Compare learned, uniform, regional, and classical methods within the same audit. Do not treat its SNR or a changed window/reconstruction setting as directly interchangeable with a previous training-test printout.

A four-recording run is a smoke check, not a representative performance estimate. If it loads and completes correctly, continue to the complete source-validation audit.

## 2. Audit the full validation set

Remove `--max-recordings` and use a separate output directory. Despite its name, `--plot-recordings 6` saves up to six worst learned-fusion **windows**, which may come from the same recording; it does not restrict which recordings enter the metrics. Those plots are selected failures, not a representative sample. Add `--save-waveforms` only when retaining the diagnostic waveform arrays is useful, since this takes additional storage. Existing output directories containing a manifest are rejected to preserve previous audits; choose a new directory for a repeat.

```bash
python tools/audit_regional_dino.py --config configs/train_configs/cross/MMPD_PURE_RegionalDINO.yaml --checkpoint runs/RegionalDINO_20261123_235243/checkpoints/MMPD_PURE_RegionalDINO_best.pth --split valid --device cuda:0 --batch-size 8 --plot-recordings 6 --output diagnostics/mmpd_full
```

Read `summary.json` and `methods.csv` first, then locate failing recordings in `records.csv`. The outputs are:

| Artifact | Contents |
| --- | --- |
| `manifest.json` | Command, resolved config, architecture, checkpoint epoch/hash, source and training file-list hashes, audited/training subject lists, recording counts, and protocol. |
| `summary.json` | Aggregate method scores and fusion-decision counts; includes usable and unusable window counts. |
| `methods.csv` | One summary row per method, with reference-dependent oracles flagged. |
| `records.csv` | Method diagnostics for every recording segment/window, including HR, waveform, lag, and harmonic checks. |
| `regional.csv` | Model and classical diagnostics for every region/window; model rows include mean learned weights. |
| `reference_checks.csv` | Reference waveform self-diagnostics, variance, spectral peak, and autocorrelation-derived HR; these do not certify the sensor. |
| `fusion_decisions.csv` | Rule-based flags such as a region within 5 bpm while learned fusion is over 10 bpm; these are diagnostic heuristics. |
| `coverage.csv` | Available/evaluated frames and windows per contiguous span; explicitly reports any omitted tail shorter than one second. |
| `plots/01_audit.png`, ... | Filtered waveforms, PSDs, regional errors, and weights for selected failing windows. Waveforms are z-scored and spectra rescaled for display only. |
| `plots/01_regional_spectra.png`, ... | Each individual expert's detrended, un-bandpassed spectrum, with reference peak and doubled-peak bands; these remain visible above the HR search band. |
| `waveforms/<recording>_segment<number>.npz` | Optional raw model predictions, regional outputs, weights, reference, and RGB traces with `--save-waveforms`. |

Keep the manifest with the tables. A successful run makes no edits to the checkpoint, caches, or labels.

## 3. Interpret the regional comparisons

The useful comparison holds the trained regional waveforms fixed and changes only the fusion calculation:

- **Learned fusion:** the checkpoint's video-dependent weights combine its regional predictions.
- **Uniform fusion:** the same predictions receive equal weight over the valid regions. This isolates the effect of learned weighting without changing the experts or loading another checkpoint.
- **Reference-selected best region:** a diagnostic oracle selects a region using ground-truth HR. This uses information unavailable at deployment and must never be presented as a deployable model or a replacement checkpoint-selection rule.

A lower learned-fusion error than uniform fusion supports a benefit from the trained gate on that set. It does not establish that DINO caused the benefit: the gate also uses local video statistics. If both are poor but the oracle is better, a useful regional signal may exist while fusion fails to select it. If every candidate is poor, investigate the shared pulse path, preprocessing, and data alignment before attributing the result to weighting. HR agreement alone can hide a wrong waveform, so also inspect correlation, spectra, raw prediction variance, and visible boundary artifacts. Inspect unusable/flat estimates and their counts before interpreting an average: omission of failed estimates can make a failing method appear more accurate. Best-lag absolute correlation tolerates waveform inversion and searches over delays; it is a diagnostic, not the unshifted signed waveform correlation or an automatic alignment correction.


| Regional model result | Working interpretation and next step |
| --- | --- |
| Learned clearly improves on uniform across the complete source-validation set | The trained weights help these fixed experts; compare no-DINO training before crediting DINO. |
| Uniform improves on learned and some individual regions have useful waveforms | Inspect gate ranking and teacher targets; the pulse experts may contain usable information that weighting suppresses. |
| Learned and uniform are poor, while the reference-selected region is much better | Regional selection may be a bottleneck; the oracle is only a diagnostic and its favorable HR can be coincidental. |
| Learned, uniform, and nearly every region are poor | Focus on shared pulse estimation, temporal scale, preprocessing, and alignment; a different gate alone may not solve it. |
| HR error is small but waveform correlation/variance or spectral shape is poor | Check a dominant-frequency shortcut, phase or polarity differences, harmonics, and flat-output amplification. |
Weights are relative coefficients, not absolute confidence. They sum to one even when all regional estimates fail. The 3 by 3 cells refer to crop coordinates; they are not anatomical skin regions, perfusion measurements, or validated causal explanations.

## 4. Use classical rPPG to check the signal path

Compare GREEN, POS, and CHROM with the same cached video and reference PPG. The full-face and central-grid-cell choices must be fixed before examining ground-truth errors. In a 3 by 3 grid the central cell is the middle crop rectangle, not an anatomically aligned skin mask; it may include the nose, eyes, or surrounding tissue depending on cropping and motion.

The audit implements documented diagnostic variants: POS and CHROM use 1.6-second, Hann-weighted, 50%-overlapping windows with edge padding. POS therefore differs from the toolbox's stride-one rectangular overlap-add implementation. These are sanity checks, not an exact reproduction of a published classical baseline. Raw variance and numerical flatness thresholds are not calibrated signal-quality measurements.


If the saved checkpoint is unavailable, perform the video/reference sanity checks alone:

```bash
python tools/audit_regional_dino.py --config configs/train_configs/cross/MMPD_PURE_RegionalDINO.yaml --split valid --classical-only --max-recordings 4 --plot-recordings 6 --output diagnostics/mmpd_classical_smoke
```

This omits model loading and learned/uniform comparisons. Remove `--max-recordings` for all source-validation recordings and choose a new output directory. It uses the existing cache and does not generate pseudo-labels or correct GT.
The audit also evaluates grid-cell classical estimates. Any reference-selected best classical region is a diagnostic oracle, not a valid automatic ROI selection method. Report those separately from the preselected full-face and central-cell results.

Useful patterns are:

| Observation | Next check |
| --- | --- |
| Classical estimates agree with reference HR while the trained model fails | Inspect the learned pulse path, spectral peaks, output variance, and fusion. |
| One fixed classical ROI is useful and the full-face estimate is poor | Inspect crop/background contamination and regional signal quality. |
| Classical methods and the trained model all fail | Inspect motion, illumination, clipping, crop quality, and the raw reference/video alignment. This does not establish that GT is wrong. |
| Several video methods agree with each other but disagree with reference HR | Investigate shared lighting/motion rhythms, harmonics, and reference integrity. Agreement from the same video is not independent ground-truth confirmation. |

POS, CHROM, and GREEN can fail under motion, changing illumination, weak skin coverage, or exposure problems. Conversely, a clean reference PPG can correctly disagree with all video-based methods. Do not choose a favored algorithm or region after viewing the reference and label it an unbiased baseline.

## 5. Verify suspicious ground truth at the source

For a small number of clear failures, trace the cached input/label pair back to its raw recording. Keep the original labels intact while investigating.

1. Check the raw MMPD `GT_ppg` contact waveform, independently recorded sensor HR if available (the original MMPD file is not assumed to contain an HR field), recording identifiers, sample rate, units, and quality metadata. Check missing values, flat segments, saturation, and unexpected discontinuities. Cached PPG was standardized per recording, so its scale cannot establish sensor gain or physical amplitude and is insufficient to diagnose clipping or dropouts at the source.
2. Compare video and contact-sensor timestamps, start times, durations, missing frames, and resampling. The current cache loader matches waveform length to video length; equal lengths alone do not establish synchronized start times or correct sensor delay.
3. Inspect at least a 10-second contact-signal interval, and longer when the rhythm is irregular, then count plausible pulses as an independent sanity check. Compare pulse count and sensor HR with the dominant spectral peak, and inspect both the fundamental and doubled/halved candidates. An isolated strongest FFT peak may be a harmonic.
   The default HR search is 45–150 bpm for consistency with the existing evaluation. Peaks near either boundary require particular care: the search range can exclude the actual fundamental. The shaded `f_ref` band is the reference's selected spectral peak, not proof of the physiological fundamental.
4. Compare several longer contiguous windows. Distinguish a stable offset or lag from time-varying misalignment and changing HR. A consistent delay can preserve spectral HR while reducing waveform correlation; a wrong time span can affect both.
5. Record the evidence and suspected cause. If a label or alignment issue is confirmed, define and validate a reproducible preprocessing correction before regenerating caches and repeating the same evaluation protocol.

The audit does not automatically replace GT, shift it to maximize agreement, select the best classical estimate as a new label, or repair synchronization. Such changes would alter the dataset and could bias the evaluation if chosen from the test outcomes.

## 6. Test whether DINO contributes

The same-checkpoint uniform comparison tests learned fusion, not the benefit of DINO. That requires training a model without DINO context. The existing learned-fusion ablation keeps the shallow regional pulse path and learned gate and disables DINO:

```bash
python main.py --config_file configs/train_configs/ablations/regional_shallow_learned.yaml
```

Use the same source subject split, cache contents, label representation, loss coefficients, batch size, training budget, seeds, checkpoint eligibility, and raw-waveform reconstruction as the RegionalDINO run. The ablation YAML inherits those settings and changes `USE_DINO` and the experiment name; inspect the resolved configuration before launch. Several matched seeds are preferable before claiming a reliable difference.

After training, run the audit with that ablation configuration and its own saved checkpoint. Do not reuse a DINO checkpoint for the no-DINO model. Compare full-validation learned-fusion HR, waveform quality, failure distribution, throughput, and memory under the same window protocol.

If the no-DINO model matches or improves on RegionalDINO, the evidence does not support a DINO contribution in this setup. If RegionalDINO improves consistently, further controls such as frozen versus partially tuned context, context sampling rate, and controlled regional corruption can test why. Neither outcome alone establishes novelty.

MMPD validation has already been used for model selection. Reusing it here diagnoses source learning; it is not an independent source-test estimate. Keep PURE out of these design choices and tuning decisions, and use it for the final cross-dataset report after the method is fixed.
