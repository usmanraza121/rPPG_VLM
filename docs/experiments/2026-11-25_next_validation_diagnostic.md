# Next action: validation-only decoder and training-rate audit

No new training is needed yet. The matched PURE audits demonstrate both a training-domain gap and a lower-filter-edge interaction on recording901. Determine decoder sensitivity and source HR coverage before changing architecture or loss.

A new standalone tool, `tools/check_regional_protocol.py`, reuses the existing cache reader, strict checkpoint loader/model construction, validation inference and classical RGB extractors. Model, trainer, loss and existing evaluation files are unchanged. Four focused tests verify known-frequency behavior, a fixed reference across candidate protocols, undefined-output accounting/subject-overlap rejection, and an end-to-end CLI run where training RGB and test arrays deliberately do not exist.

## What it reads and measures

- All three split CSVs for subject-membership/overlap checks, separately for each dataset. It reads no test waveform/image arrays. This independently checks the actual PURE6/2/2membership when its CSVs are present.
- Source-training **PPG labels only** for clip-level HR proxies, contiguous-record HR proxies, per-subject HR bins and disagreement between the current and detrend-only reference estimates. It never joins missing clip IDs across a gap. Cache labels remain unchanged. These are reference-derived HR estimates, not independent physiological truth; short160frame clips have limited spectral information.
- Validation RGB/PPG with the specified existing checkpoint. Learned/uniform predictions and fixed GREEN/POS/CHROM are compared under three common postprocessing candidates: current Tarvainen100 + .75–2.5Hz bandpass; Tarvainen100 without bandpass; Tarvainen100 + .6–3.3Hz bandpass. Every candidate uses the same45–150bpm HRsearch and Welch256/50%overlap/nfft3333. Classical methods retain their own internal extraction steps; only common downstream processing changes.
- Independent known-frequency waveforms across45–150bpm, two durations and several harmonic amplitudes/phases. The actual synthetic input frequency is the reference. Equal/strong harmonics and short windows can be intrinsically ambiguous; a wider filter is not guaranteed to recover a fundamental.

All primary validation errors use a **fixed reference HR from the current pipeline**. Alternative per-protocol reference HR is exported separately for sensitivity checks, so a changed reference cannot silently improve MAE. This fixed reference can itself have filtering bias; inspect `validation_reference_hr_changes` rather than certifying it as truth. The tool does not select a winning protocol or checkpoint.

The wider-passband candidate was motivated after test-case inspection. This is a validation investigation, not an independent confirmatory protocol choice. Freeze any eventual protocol using physiological requirements, synthetic controls and validation evidence, apply it equally to all methods, and use future untouched data for confirmation. Keep all earlier official scores.

## Server commands

Copy the new `tools/check_regional_protocol.py` file into the same location in the server repository. Existing model and audit helper files must be the same versions used for the prior audits. From `rPPG_VLM/`, run PURE first:

```bash
python tools/check_regional_protocol.py \
  --config configs/train_configs/intra/PURE_PURE_RegionalShallowLearned.yaml \
  --checkpoint runs/RegionalDINO_20261125_113336/checkpoints/PURE_PURE_RegionalShallowLearned_6_2_2_best.pth \
  --output diagnostics/pure_validation_protocol
```

Then run the MMPD-trained model on its source validation split:

```bash
python tools/check_regional_protocol.py \
  --config configs/train_configs/ablations/regional_shallow_learned.yaml \
  --checkpoint runs/RegionalDINO_20261124_121357/checkpoints/MMPD_PURE_RegionalShallowLearned_best.pth \
  --output diagnostics/mmpd_validation_protocol
```

Outputs are new directories; the command refuses to overwrite an existing directory. Default GPU visibility matches the existing audit (`CUDA_VISIBLE_DEVICES=1` unless already set). `--device cpu` is available; MMPD validation inference can be substantially slower there.

## What to inspect next

Return both output directories. `summary.json` provides validation comparison and training HR-bin counts; `validation_protocols.csv` preserves every recording/method/candidate estimate, including undefined cases. `training_recording_hr.csv` and `training_clip_hr.csv` reveal rate coverage, while `synthetic_frequency_response.csv` shows whether the decoder changes a known rhythm near its band edges. `manifest.json` records actual subjects, CSV hashes, checkpoint hash/epoch and resolved config.

If decoder behavior is sound but MMPDtraining lacks low-rate examples, a controlled paired video/PPG rate-coverage intervention becomes a reasonable next experiment. If rates are covered yet transfer still fails, investigate visual/reference/nuisance differences and pulse-path generalization. If the filter alone changes dominant peaks on validation, settle the common measurement protocol first. These are decision branches, not established causes or promised improvements. No newDINOversion, per-recording halving rule, or simultaneous architecture/loss changes are warranted at this stage.
