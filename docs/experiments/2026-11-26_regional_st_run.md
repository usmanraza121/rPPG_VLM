# Run the RegionalSTAttention architecture experiment

Prepared 2026-11-26. The design is now integrated into `main.py`, the shared regional trainer, preflight and audit/protocol tools. This is the first **architecture experiment**, with rate augmentation disabled. It trains from scratch on MMPD, selects its checkpoint using MMPD validation loss, then uses the existing full-PURE evaluation. Its physiological performance is not yet known.

## Transfer the update

Extract `regional_st_update.zip` **inside the server's existing rPPG_VLM repository**, allowing its listed source/config files to update. Do not extract it into Codes/, a dataset directory or the previous augmentation-update folder. The archive has repository-relative paths, with `main.py` and `config.py` at its root. It contains no datasets or trained checkpoints. Preserve any server-specific edits before overlaying shared files.

The existing PyTorch environment is sufficient; this model does not require DINO downloads, Mamba kernels, deformable-convolution extensions, a new preprocessing cache or raw data. Data paths are inherited from the previous working no-DINO configuration. If the server has different paths, keep the exact previously verified cache and file lists.

## First run: source-cache/GPU preflight

Run from the server repository root:

```bash
python tools/check_regional_dino.py \
  --config configs/train_configs/ablations/regional_st_attention.yaml \
  --device cuda:0 \
  --clips 4 \
  --steps 20
```

This reads four evenly spaced source-training clips, checks Raw RGB/labels and runs a short fixed-batch fit. It saves no checkpoint and does not alter the cache. Expect `Model: RegionalSTAttention`, `Use DINO: False`, finite losses/gradients, and a static-input output near zero. The backbone gradient should be zero because there is no DINO backbone. Correlation need not reach any particular threshold in20 steps on real data; this is a mechanical check, not a physiological validation score. Stop and inspect a traceback or nonfinite result before a full run.

The existing scripts default CUDA_VISIBLE_DEVICES to1 only if it is unset. `cuda:0` is the first GPU visible to that process. To choose physical GPU0 explicitly on Linux, prefix a command with `CUDA_VISIBLE_DEVICES=0`; otherwise preserve the GPU environment used for the previous runs. No global GPU setting was changed by this update.

## Start the architecture experiment

If the preflight completes, run:

```bash
python main.py --config_file configs/train_configs/ablations/regional_st_attention.yaml
```

Settings: local3D encoder, two local TCN blocks and three temporal-attention blocks;101,368 trainable parameters; nine regional heads and existing fusion;160frames,30Hz,128-square Raw RGB,12-square regional resampling and standardized PPG. Same loss weights,40epochs,batch16,peakLR1e-4,split,checkpoint eligibility from epoch5 and evaluator as the original no-DINO model. **Rate augmentation is disabled.** All160 timesteps remain present. No prior checkpoint is loaded for initialization.

The trainer is shared under `RegionalDINOTrainer`; the new architecture has its own `MODEL.NAME` and `MODEL.REGIONAL_ST` block. Shared loss/optimizer settings retain the existing `TRAIN.REGIONAL_DINO` name so they are identical. New checkpoints include `model_name` and exact constructor arguments; mismatched model/backend settings are rejected. Legacy unnamed RegionalDINO checkpoints remain readable only as RegionalDINO.

The new run directory is `runs/RegionalSTAttention_<timestamp>/`. Checkpoints:

- `checkpoints/MMPD_PURE_RegionalSTAttention_best.pth`
- `checkpoints/MMPD_PURE_RegionalSTAttention_last.pth`

Only those two rotating checkpoint files are retained; the update does not create a checkpoint for every epoch. The selected checkpoint is evaluated automatically at the end. `training.log`, `regional_history.jsonl` and `regional_waveforms.pt` are saved in the run directory. There should be no `rate_augmentation.jsonl` for this config. A different architecture can use more GPU memory despite its small parameter count; measure the actual run rather than infer VRAM from parameters. Keep batch16 for comparison when it fits.

## Matched control after the first run

The control retains exactly the same new3D encoder and swaps attention for six dilated TCN blocks. It has100,936 parameters, approximately.43% fewer:

```bash
python main.py --config_file configs/train_configs/ablations/regional_st_tcn.yaml
```

Its checkpoint prefix is `MMPD_PURE_RegionalSTTCN`; model family in the run directory remains RegionalSTAttention, with backend `tcn` stored in metadata. Do not load the attention checkpoint into it. Use separate runs from scratch with the same training settings. This comparison distinguishes the temporal mixing choice; it does not prove novelty from an attention component alone.

A context-only control using the original2D encoder with six TCN blocks is also prepared:

```bash
python main.py --config_file configs/train_configs/ablations/regional_shallow_long_tcn.yaml
```

Its91,024 parameters differ from the original62,800, so it is a longer-context/capacity control rather than a perfectly parameter-matched baseline. The original no-DINO run remains the reference; do not repeat it merely because the new configs exist.

## Audit the selected source checkpoint after training

Replace the timestamped run directory below with the one actually printed during training:

```bash
python tools/audit_regional_dino.py \
  --config configs/train_configs/ablations/regional_st_attention.yaml \
  --checkpoint runs/RegionalSTAttention_<timestamp>/checkpoints/MMPD_PURE_RegionalSTAttention_best.pth \
  --split valid \
  --save-waveforms \
  --output diagnostics/mmpd_regional_st_attention_full
```

Compare against `mmpd_no_dino_full/` using matched membership and recording-level metrics. Assess candidate retention, HR bias/ranges and subject differences, not only the PURE aggregate. The primary source baseline is MAE8.5660bpm; the original full-PURE score is18.2611. Do not compare full-record scores with the trainer's separate clip diagnostic. PURE has already influenced design, so its result remains exploratory; reserve fresh confirmatory data and repeated matched seeds for a paper claim.

## Local verification

The integrated model passed a small actual regional training/validation cycle, save/reload through only_test, raw waveform export, wrong-model/backend checkpoint rejection, and audit CLI reconstruction from synthetic cached files. Seven existing trainer tests passed, followed by19 tests covering the new integration and the audit/protocol/augmentation regressions, all with zero failures. Final regression-suite results are supplied with the update. A20-step synthetic preflight using the actual attention YAML changed correlation from-.968 to.997 with finite losses and static output about4.4e-16; this validates fitting/wiring only. CPU checks do not establish GPU speed, real-data accuracy, or performance improvement. Full server training has not been run here.

Detailed design: `docs/designs/2026-11-26_regional_st_attention.md`.
