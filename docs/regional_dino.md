# Regional DINO prototype for rPPG waveform estimation

This is a research prototype alongside the existing DINOv2-TCN baseline. Its hypothesis is that a shallow path can preserve small local RGB changes while sampled DINOv2 features help estimate which regions to trust. It predicts a waveform for each region and combines those waveforms explicitly. Neither improved accuracy nor a novel contribution has been established; both require controlled experiments and a fuller related-work comparison.

Keep the existing baseline configuration and evaluator as experimental references. Any active run can finish before comparing the prototype.

## What changes relative to the baseline

The baseline processes each frame with DINOv2, averages its patch tokens, and passes the resulting frame sequence through a TCN. The prototype separates two jobs:

1. A shallow RGB path preserves local temporal variation. It uses raw RGB, temporal differences, and local RGB traces to estimate one waveform per region through a shared temporal network.
2. A sampled DINOv2 context path conditions regional reliability weights. DINO is a vision foundation model; this implementation has no language encoder or text prompts and is not a vision-language model.

The intended flow is:

```text
Raw RGB clip [B, T, 3, H, W]
    |
    +-- shallow RGB/difference features + local RGB traces
    |      -> region pooling -> shared temporal network
    |      -> regional waveforms s[r, t]
    |
    +-- sampled RGB frames -> DINOv2 context
           -> context aligned to the same region grid
           -> learned regional weights w[r, t]

Explicit waveform fusion: prediction[t] = sum_r w[r, t] * s[r, t]
```

`RegionalDINO(video)` returns `[B, T]`. With `return_aux=True`, it returns `waveform [B, T]`, `regional_waveforms [B, R, T]`, `region_weights [B, R, T]`, `quality_logits [B, R, T]`, and `valid_regions [B, R]`. The weights are nonnegative and sum to one over valid regions at every time step. `fusion="uniform"` and `use_dino=False` provide controls. The default DINO backbone is frozen; `finetune_blocks=2` enables only its last two blocks and final normalization. No backbone is downloaded when DINO is disabled.

Defaults use 9 grid regions, a 48-channel temporal network with four residual blocks (dilations 1, 2, 4, 8; receptive field 61 frames), and DINOv2 ViT-S/14 context sampled every 8 frames. The endpoint frame is included, context is linearly interpolated at the actual sampled frame positions, and gate logits receive a 17-frame moving average. Each local crop is pooled to 12 by 12 for the shallow spatial stem; its mean RGB and relative-difference traces bypass that image pooling. Relative differences use `(rgb[t] - rgb[t-1]) / (rgb[t] + rgb[t-1] + 1e-3)` on RGB divided by 255. The first difference is zero. Temporal convolution uses replicated padding and channel dropout; no BatchNorm or output standard-deviation division is used.

DINO context is sampled to reduce computation. It is not assumed to retain the pulse at the original frame rate. The shallow path preserves every input time step and produces the full output waveform. Frame rates, context sampling, interpolation, padding, and regional coordinate alignment must be recorded with the experiment.

## Regions and the meaning of a map

The initial regions are a 3 by 3 grid over the preprocessed face crop. They are **grid cells, not anatomical face masks**. A large face crop can place eyes, hair, mouth, or background inside a cell. Region indices describe coordinates in the resized crop; they do not identify stable pieces of facial tissue under head motion. A cell reconstructed from DINO patch coordinates is still a coarse cell, not a pixel-level measurement.

A caller may supply a boolean `region_mask [B, R]`, with at least one valid cell per example. Invalid cells are zeroed before full-frame DINO encoding, and their regional candidates are excluded from fusion and auxiliary supervision. Thus changing the original contents of an excluded cell cannot alter the remaining DINO context. This intervention changes the contextual image and recomputes weights; valid DINO tokens still have global receptive fields over the remaining image. A removal score is therefore not a direct anatomical or causal tissue attribution. Keep the same mask in inference, losses, and diagnostics, and distinguish expert removal from synthetic corruption of an active input cell. The default grid has no hidden skin-segmentation heuristic. Record the mask, crop parameters, grid dimensions, and region order for reproducibility. Landmark-aligned skin regions or a validated skin mask are future data-processing experiments, independent of changing the waveform network.

Saved regional waveforms show the experts' estimates. Fusion weights show their coefficients in the final prediction. Neither is a calibrated confidence probability: weights must sum to one even when every region is unreliable. Weight maps are not blood-perfusion maps, hemoglobin concentrations, or automatically faithful explanations. Such interpretations need independent physiological validation and perturbation tests. Compare region removal, controlled contamination, and weight changes before claiming that a displayed map explains a prediction.

## Training objective and supervision boundary

The proposed objective combines raw waveform MSE, a correlation term, a Jensen-Shannon divergence between normalized power spectra in the configured physiological band, regional waveform supervision, and quality supervision:

```text
L = lambda_mse L_mse + lambda_corr L_corr + lambda_psd L_js_psd
    + lambda_region L_region + lambda_quality L_quality
```

Default coefficients are MSE 0.1, correlation 1.0, PSD 1.0, regional 0.25, and quality 0.1. `L_corr` is `1 - Pearson` with epsilon-regularized variances. `L_js_psd` compares centered, Hann-windowed, normalized power in 0.75 to 2.5 Hz at 30 Hz sampling, using a power-of-two FFT length of at least 256. Epsilon is `1e-6`; flat signals receive a uniform spectral fallback, which is a numerical convention rather than evidence of pulse. The regional auxiliary term is the mean of regional correlation error plus regional JS spectral error; it does not match regional raw amplitudes to contact PPG.

Here raw means that the final predicted waveform is not z-scored before MSE. Regional predictions and the fused prediction are mean-centered, without scaling their standard deviation. A common DC correction is applied to all valid regional outputs so the returned waveform remains exactly their weighted sum. Targets retain recording-level standardization, so individual target clips can have a nonzero mean. Therefore a zero-mean prediction cannot drive raw MSE below the target clip mean-squared contribution; do not require an exact zero fixed-clip MSE. Correlation and PSD terms address shape and periodicity; a spectral term cannot alone establish correct waveform phase, morphology, or amplitude.

The quality teacher derives one relative ranking per clip from regional correlation error plus JS spectral error against training PPG. Targets are `softmax(-detached_error / 0.25)` over valid regions. A KL term compares that target with `softmax(mean_t quality_logits)`; it supervises a clip-level ranking, not the instantaneous quality of every frame. These targets are detached before the quality loss, so the weighting branch does not backpropagate through teacher construction. PPG is available for loss computation during training only. Model inference, regional weights, saved maps, and validation/test forward passes must depend solely on video and any explicitly supplied video-derived mask. Reporting validation quality against PPG is an evaluation, not an input to inference.

A detached teacher is still coupled to the current experts and is not independent quality ground truth. The teacher uses shape and spectral agreement, so it does not establish correct regional amplitude or absolute quality. A low-amplitude or biased waveform must still be inspected independently of the relative rank. Track raw variance, regional errors, weight entropy, expert similarity, and the all-regions-poor case. A uniform-weight control is essential.

The initial regional auxiliary coefficient is 0.25 and must be ablated. Supervising every cell with the same contact PPG is a modeling assumption: background and occluded cells may not contain a recoverable signal, and facial versus contact PPG may have timing differences. Region masks can exclude known invalid cells, but this prototype does not have anatomical or local perfusion ground truth. A low regional loss does not prove a valid local physiological measurement.

## Related work and what it supports

The local reading set provides mechanisms and evaluation questions, not proof that this combination will outperform the baseline:

- [Lighting effects on optimal facial regions for remote heart rate measurement](https://doi.org/10.1038/s44325-026-00140-7) motivates testing regional reliability under illumination changes. Its anatomical ROIs and conventional extraction algorithms do not validate a learned 3 by 3 grid or a DINO gate.
- [HemNet](https://doi.org/10.1109/TCSVT.2025.3650742) motivates preserving physiological color variation and distinguishing it from illumination and reflection. This prototype does not implement HemNet's hemoglobin disentanglement or claim to measure hemoglobin.
- [SaTPhys](https://doi.org/10.1109/TCSVT.2026.3686300) provides an example of explicitly modeling spatial and temporal physiological context. Its architecture and reported experimental protocols are separate from this prototype.
- The supplied MaskFusionNet and CodePhys papers are relevant comparisons for complementary streams, task-specific pretraining, and physiological representations. The prototype does not implement their masked-pretraining procedure or latent codebook and should not be described as a reproduction of either.

Vision-language work should be included in the scientific positioning: [VLphys](https://arxiv.org/html/2407.08507v2), [PhysLLM](https://arxiv.org/html/2505.03621v2), [PhysVLM](https://ieeexplore.ieee.org/document/11426886), [PhysVR](https://arxiv.org/html/2608.29663v1), and [VITAL](https://arxiv.org/html/2610.19052v1). PhysVR and VITAL are 2026 preprints in this reading set; the PhysVLM check was limited to its publisher abstract. These sources establish relevant prior directions, not a verified claim that the present model is the first to combine foundation features with rPPG. Compare actual backbone roles, language use, signal inputs, supervision, and evaluation protocols before making novelty or performance claims.

DINOv3 can be a later backbone ablation. The current model loads the DINOv2 hub and validates a 14-pixel patch-compatible encoder size; DINOv3 is not a supported drop-in backbone in this implementation. Better general visual features do not guarantee better sensitivity to weak pulse color changes. First determine whether DINO context adds value over an equally trained shallow model. A version change should keep the pulse path and evaluation protocol fixed and report patch-size, input-resolution, compute, and checkpoint differences.

## Ablation matrix

All comparisons use the same source split, cached input representation, evaluation windows, training budget, checkpoint eligibility, and recorded seeds. Tune peak learning rates on MMPD validation; the backbone rate must remain lower when partial fine-tuning is enabled. Start with a common raw-MSE objective where feasible, then add loss terms separately rather than attributing their benefit to the architecture.

| Experiment | Pulse estimator | Fusion | DINO context | Question |
| --- | --- | --- | --- | --- |
| Existing DINOv2-TCN | Mean DINO patch sequence plus TCN | Existing head | Per-frame baseline | Does the prototype improve on the original design? |
| Shallow uniform | Regional RGB/difference/traces | Uniform | Off | Is the shallow pulse path sufficient? |
| Shallow learned | Same regional path | Learned | Off | Do learned weights help without DINO? |
| Regional DINO frozen | Same regional path | Learned | Frozen sampled DINOv2 | Does pretrained context add measurable value? |
| Regional DINO partial | Same regional path | Learned | Last two DINOv2 blocks plus final norm trainable | Does context adaptation help? |
| DINOv3 context | Same regional path | Learned | Matched DINOv3 variant | Does backbone version improve the result at its compute cost? |

The supplied configurations are `regional_shallow_uniform.yaml`, `regional_shallow_learned.yaml`, `regional_partial_dino.yaml`, `regional_mse_only.yaml`, and `regional_no_quality.yaml` under `configs/train_configs/ablations/`. The MSE-only file sets MSE weight to 1 and every other coefficient to zero. The no-quality file removes only the teacher loss while retaining learned fusion. Shallow architecture controls otherwise inherit the combined objective; use a matched loss when attributing an effect to architecture.

For the strongest source-validated architecture, separately remove the correlation, PSD, regional auxiliary, and quality terms. Also compare context sampling intervals, grid sizes, and supplied region masks. These are future experiments unless a matching configuration and result are recorded. Use several seeds for the finalists and report parameter counts, trainable parameters, throughput, and peak GPU memory.

A useful scientific result includes a negative finding: if the shallow learned model matches the DINO-conditioned model, the evidence does not support an added foundation-model contribution.

## Validation and diagnostic checks

Select and tune using held-out MMPD subjects. Reserve PURE for final cross-dataset evaluation. A checkpoint evaluated on the same MMPD validation set used for selection provides a source-learning diagnostic, not an independent source test. An independent MMPD test split is needed for an unbiased source test estimate.

Before a full run:

- Fit a fixed subset of 2 to 4 aligned training clips with augmentation disabled. Check finite losses and gradients, waveform shape, raw output variance, and whether errors fall below the measured zero predictor. Verify that intended trainable DINO blocks receive gradients when enabled. The final quality layer is initialized to zero to start uniform fusion, so context/DINO gradients may be zero on the first backward pass; check them after the gate has begun updating.
- Run a static-frame clip and temporally shuffled video. A plausible periodic output on these controls is not evidence of recovered physiology; inspect whether an unsupported rhythm or dataset prior persists.
- Shuffle video-target pairings for a diagnostic run. Learning the normal supervised relationship should become harder; a convincing result with mismatched pairs demands a leakage or shortcut investigation.
- Contaminate one known region with controlled brightness variation, noise, occlusion, or motion-like disturbance while leaving the other regions unchanged. Compare uniform and learned fusion, region errors, and the direction of weight changes. Record contamination strength and its placement. Repeat cases in which all regions are degraded.
- Check that permutations of region order with corresponding coordinates and masks preserve the intended reconstruction. Plot region boundaries on the actual preprocessed frame. Confirm no frame order or time dimension changes across the two paths.

During training, save element-weighted raw MSE and the zero-prediction baseline across the complete validation set, waveform correlation, raw prediction and target standard deviations, HR MAE/correlation, and regional/fusion diagnostics. A reduction in raw MSE can come from predicting a mean or nearly flat waveform, so waveform and spectral diagnostics remain necessary. The loss module provides a per-clip Hann-windowed FFT HR helper, with NaN for unusable flat signals. This is a diagnostic and differs from the repository full-recording Welch evaluator; label the two separately and do not mix their scores in a results table.

For quality evaluation, compare predicted relative weights against held-out regional errors, report ranking agreement and corruption response, and examine cases where all experts fail. If reporting absolute confidence, fit and assess a separate calibration procedure on source validation data. Softmax weights alone cannot support that claim.

At inference, reconstruct recordings in numeric chunk order and keep raw predictions. Compare raw concatenation with the historical per-clip-normalized evaluation because independent clip normalization can amplify noise and create boundary artifacts. Any evaluation change must be applied to the baseline and prototype together. Keep HR frequency bands and window durations explicit; do not silently change the metric protocol.

## Running and experiment records

The main experiment configuration is `configs/train_configs/cross/MMPD_PURE_RegionalDINO.yaml`, inheriting existing dataset settings from `MMPD_PURE_DINOv2TCN.yaml`. It uses batch size 16 and 40 epochs, compared with batch size 32 in the inherited baseline file. Match the batch size when isolating architecture or report the difference. Architecture controls are under `configs/train_configs/ablations/regional_*.yaml`. Verify every dataset/cache path for the execution host. A configuration change does not regenerate old cached arrays; inspect a real batch for raw RGB scale, three channels, time order, and PPG alignment.

The preflight reads an existing training cache, chooses fixed clips, prints the actual input audit and losses, and saves no checkpoint. It uses constant LR, no augmentation, and eval-mode dropout to make fixed-batch learning easier to interpret; it is a diagnostic, not the full training recipe.

```bash
python tools/check_regional_dino.py --config configs/train_configs/cross/MMPD_PURE_RegionalDINO.yaml --clips 4 --steps 100 --device cuda:0
```

For an offline CPU wiring check using artificial color oscillations and no DINO download:

```bash
python tools/check_regional_dino.py --config configs/train_configs/cross/MMPD_PURE_RegionalDINO.yaml --synthetic --no-dino --clips 2 --steps 20 --device cpu
```

`--steps 0` performs forward diagnostics without fitting. `--seed` and `--threads` control preflight reproducibility and CPU threading. Synthetic fitting and static-input checks verify wiring; they do not validate rPPG accuracy or the pretrained DINO path. Run the real cached-subset check with DINO enabled on the training host before a full experiment.

After reviewing those diagnostics, launch the configured experiment from the repository root:

```bash
python main.py --config_file configs/train_configs/cross/MMPD_PURE_RegionalDINO.yaml
```

Use an ablation YAML in either command to select its experiment. In `only_test` mode, set `INFERENCE.MODEL_PATH` to a RegionalDINO checkpoint and keep architecture options equal to those saved in the checkpoint. A legacy DINOv2-TCN checkpoint is incompatible.

The RegionalDINO route respects an explicitly set `CUDA_VISIBLE_DEVICES`. Both `main.py` and the preflight default that variable to physical GPU `1` when it is absent; `DEVICE: cuda:0` refers to the first GPU visible to the process. For physical GPU `0`, set `$env:CUDA_VISIBLE_DEVICES = "0"` in PowerShell, or `export CUDA_VISIBLE_DEVICES=0` in bash, before launching Python. The preflight also accepts `--device cpu` or a logical CUDA device explicitly.
The implementation uses a standalone `RegionalDINOTrainer` so the existing baseline training code remains available for comparisons. It audits the configured input representations and an actual first batch, applies gradient clipping (default norm 1.0), and uses AdamW (default weight decay `1e-4`) with OneCycleLR. The scheduler receives one peak LR for the pulse/context group and a separate peak for trainable DINO parameters; the latter defaults to `1e-5`. The pulse/context peak comes from `TRAIN.LR`. The full backbone is absent from the optimizer when frozen. Record weight decay, both peak rates, and all loss weights because these differ from the legacy recipe and can confound a pure architecture comparison.

Each epoch appends `regional_history.jsonl` containing the total and component losses, full-set raw MSE and zero baseline, waveform correlation, raw prediction/target standard deviations, diagnostic clip HR MAE, undefined-HR fraction, weight entropy, learning rates, selection score, and checkpoint eligibility. `SELECTION_METRIC: loss` selects by total source-validation loss; `clip_hr_mae` is an explicit alternative, not an implicit change. Final testing saves `regional_waveforms.pt` with raw fused predictions, references, regional outputs, weights, validity masks, and grid/sample-rate metadata. The final evaluator uses raw clips joined by recording, so re-evaluate the baseline raw outputs as well for a fair reconstruction comparison. The historical RMSE uncertainty and SNR definitions remain in the legacy evaluator; do not present them as corrected statistics.

Checkpoint retention is limited to two files: the eligible best checkpoint and the latest checkpoint. Eligibility begins at zero-indexed epoch 5 to honor the experiment's storage policy. Early validation diagnostics are still recorded; the selected checkpoint is the best among eligible epochs, which can differ from the absolute minimum validation epoch. Both checkpoint payloads retain model arguments, model weights, optimizer/scheduler states, epoch, and metrics. A command to resume training is not implemented by this prototype; the saved states support a future resume implementation. Record which checkpoint supplied the final test result.

For every reported run, record the code revision, exact resolved configuration, data/cache/file-list identities, subject splits, random seeds, pretrained checkpoint identifier, trainable blocks, both parameter-group LR schedules, loss coefficients, region mask/grid, context sampling, checkpoint rule, and evaluator version. Report actual measured results only; successful shape checks or a synthetic smoke test do not establish rPPG accuracy.
