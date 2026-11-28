# Frozen attention extractor: pulse-summary fusion experiment

Implemented directly in `rPPG_VLM` on2026-11-28. No new data preprocessing is needed. This is a hypothesis test of additional candidate information at the fusion gate, not a guaranteed accuracy gain or novelty claim.

## Fixed source and controlled comparison

Both runs load the original Exp2 attention checkpoint selected using MMPD validation loss:

```text
runs/RegionalSTAttention_20261126_093244/checkpoints/MMPD_PURE_RegionalSTAttention_best.pth
```

The configured SHA256 is `38d73b5a4568d9d4ac65c8fb388d560f52dcde5e5a243ff28737a571ee5e5a4e`, from the supplied source audit. Training/preflight rehashes the actual server file and rejects a mismatch. The source is the better source-recording attention run (MMPD6.8709), not the run with the best observed PURE score. Its zero-based epoch is39. Neither training nor the source checkpoint is available on this local data host; the server must retain it at this path. If moved, update `TRAIN.FROZEN_FUSION.SOURCE_CHECKPOINT` in the moments config to its actual location, preserving the locked hash.

The extractor, temporal attention, regional waveform heads and trace skip are frozen and kept in eval mode. Their dropout stays off during gate training. Per-region DC centering removes the old gate's shared DC correction to recover the unchanged candidate signals. New weights can alter only their mixture and the common fused DC. No temporal alignment, amplitude standardization of the fused output or new signal decoder is introduced.

Compare two fresh gates with the same initialization seed100 and tensor shapes:

- **Moments control:** existing twelve RGB/difference first/second moments, plus six zero channels.
- **Waveform-aware:** same twelve moments plus six detached, GT-free candidate summaries: relative log-RMS, in-band power fraction, normalized in-band spectral entropy, peak concentration, leave-one-out spectral cosine agreement and signed leave-one-out waveform correlation.

The summaries are computed per160-frame clip and broadcast over time, avoiding direct injection of instantaneous pulse phase into the added features. They use the existing30Hz/.75–2.5Hz bounds and no GT/selected reference peak. Common motion or illumination can also produce agreement; these descriptors are not certified pulse confidence. Explicit masks, flat-candidate and single-active-region cases have finite behavior.

Both gates are18→32→1 pointwise convolutions with GELU, followed by the existing17-frame logit smoothing and regional softmax. The final layer starts at zero, giving identical uniform initial fusion. Both have641 trainable parameters and102009 total stored parameters including the frozen source model. The moments control's six input channels are zero: tensor counts match, but this is not a claim of equal effective information capacity. The original source gate remains frozen inside the source model and is not used for the new regional weights.

Loss definitions/weights,40epochs,batch16,OneCycle peakLR1e-4,source split,checkpoint eligibility from epoch5 and source composite-loss selection match prior runs. The constant regional loss contributes no extractor update; fused correlation/MSE/spectrum and quality supervision train the new gate. Raw128×128 input, internal12×12 regions, runtime relative differences, standardized PPG labels and the primary evaluator remain fixed. Rate augmentation is disabled. This is fresh gate training, not optimizer-state resume or full-model fine-tuning.

## Server commands

Sync the changed repository files to the server with the usual Git workflow. Run from its `rPPG_VLM` root, in the existing working Python environment. First check source loading and finite gate gradients:

```bash
python tools/check_regional_dino.py --config configs/train_configs/ablations/regional_frozen_fusion_waveform.yaml --device cuda:0 --clips 4 --steps 20 --seed 100
```

This preflight saves no checkpoint. Expect `Model: RegionalFrozenFusion`, the correct `Frozen extractor source` hash/epoch and finite loss. Frozen pulse/backbone gradients should be zero; `quality_head` should receive gradients. Fitting a few clips is only a wiring check.

Train the matched control:

```bash
python main.py --config_file configs/train_configs/ablations/regional_frozen_fusion_moments.yaml --seed 100
```

Train the waveform-aware gate:

```bash
python main.py --config_file configs/train_configs/ablations/regional_frozen_fusion_waveform.yaml --seed 100
```

Keep both results. Expected model family `RegionalFrozenFusion`, gate mode `moments` or `waveform`,641trainableparameters, same source hash, seed100. The two runs have separate timestamped directories `runs/RegionalFrozenFusion_<timestamp>/`. Each retains only rotating best/last checkpoints. Best prefixes are `MMPD_PURE_FrozenFusionMoments_best.pth` and `MMPD_PURE_FrozenFusionWaveform_best.pth`.

## Full source audits after training

Use each run's **actual** timestamp:

```bash
python tools/audit_regional_dino.py \
  --config configs/train_configs/ablations/regional_frozen_fusion_moments.yaml \
  --checkpoint runs/RegionalFrozenFusion_<control_timestamp>/checkpoints/MMPD_PURE_FrozenFusionMoments_best.pth \
  --split valid --save-waveforms --output diagnostics/mmpd_frozen_fusion_moments_full
```

```bash
python tools/audit_regional_dino.py \
  --config configs/train_configs/ablations/regional_frozen_fusion_waveform.yaml \
  --checkpoint runs/RegionalFrozenFusion_<waveform_timestamp>/checkpoints/MMPD_PURE_FrozenFusionWaveform_best.pth \
  --split valid --save-waveforms --output diagnostics/mmpd_frozen_fusion_waveform_full
```

New checkpoints contain the frozen extractor and gate plus source path/hash/epoch/architecture provenance. Inference/audits reload the complete new checkpoint and do not need the original source file to remain present. Checkpoint model/gate-mode/architecture/source-hash validation is strict. Old model paths and constructor dictionaries remain unchanged.

Preserve both runs' `training.log`, `regional_history.jsonl`, `regional_waveforms.pt` and complete MMPD audit folders. The automatic PURE export suffices for matched target analysis; no extra PURE audit is needed.

## Decision criteria

Compare the waveform-aware gate against the **equally retrained moments gate**, uniform fusion of identical candidates, and the original Exp2 learned mixture. Candidate availability, individual regional spectra and regional-oracle HR should stay unchanged except numerical/DC conventions. If they change materially, inspect freezing/reconstruction before a performance claim. Source rate/subject variation, full-record HR and waveform measures all matter. Preserve all failures; an improvement toward the GT-selected oracle is not guaranteed or available as an inference selector.

A benefit over the retrained control would support the usefulness of candidate summaries for these fixed outputs. It would not establish a new pulse extractor, solve missing-candidate cases, certify regional confidence or by itself prove novelty. Compare the closest spectral/phase/regional-fusion prior art before a contribution claim. If no benefit appears, reassess extraction/phase/relative amplitudes rather than keep adding descriptors based on PURE. Repeated extractor seeds and fresh untouched data remain confirmation work. Current source validation has already guided development, and PURE remains exploratory.

## Local validation

Offline model construction confirms641trainableparameters in both configurations. All17tests passed: four new frozen-fusion tests and thirteen existing trainer/attention/seed checks. They cover matched uniform initialization, fixed candidate/dropout behavior, frozen gradients/weights, gate updates, mask isolation, summary flat/single-region behavior, source hash/architecture rejection, new checkpoint standalone reload/export and audit provenance. These are CPU wiring checks; no real training or accuracy improvement is claimed before the server results arrive.
