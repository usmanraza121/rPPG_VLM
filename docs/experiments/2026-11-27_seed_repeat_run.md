# Next experiment: matched temporal models with seed101

Prepared 2026-11-27. All Exp2/Exp3 training and both requested source audits are complete. Exp2 has better aggregate HR on MMPD (6.8709 vs8.5147 bpm) and PURE (9.4464 vs16.7051), while Exp3 has better mean source waveform correlation/SNR. The next experiment checks whether this HR advantage reproduces under a second initialization and training order. It is a reproducibility experiment, not a new model contribution or a promise of a lower score.

## Implementation now ready

Updated directly in the GitHub-synced `rPPG_VLM` repository:

- `main.py --seed N`, overriding YAML `RANDOM_SEED`; existing configs default to100.
- `reproducibility.py` initializes Python, NumPy, torch/CUDA and separate training/general DataLoader generators before datasets/models. The old seed100 setup and cuDNN settings are preserved.
- Resolved configuration, training history and regional checkpoints record the seed. Test exports distinguish the loaded checkpoint's training seed from the inference process seed. Audits read the training seed from the checkpoint into the manifest.
- Older unnamed-seed checkpoints remain readable. Seed is provenance and is not part of architecture compatibility checks; auditing a101 checkpoint with the original YAML100 is valid.

No architecture, input/label representation, preprocessing, objective, learning-rate schedule, split, checkpoint-selection rule or primary metric was changed. No full server training was launched locally. Only the existing best/last rotating checkpoints are saved for each timestamped run.

## Run attention first

Sync these local repository changes to the server using the user's normal Git workflow. From the server `rPPG_VLM` root, in the same environment/GPU setup used previously:

```bash
python main.py --config_file configs/train_configs/ablations/regional_st_attention.yaml --seed 101
```

Train from scratch for the same40epochs, batch16, OneCycle peakLR1e-4, RawRGB/standardizedPPG,160frames, no rate augmentation. Expected log: `Random seed: 101`, resolved `RANDOM_SEED: 101`, model `RegionalSTAttention`, `temporal_backend: attention`,101368trainableparameters. The automatic PURE evaluation stays exploratory; use the unchanged source-validation loss to choose the checkpoint.

Do not restart seed100. Seed101 changes initialization, dropout and training order; matched seeds do not imply equivalent parameter draws or identical dropout masks across different architectures.

## Run the paired TCN control

After the attention run, or on a separate GPU if resources allow:

```bash
python main.py --config_file configs/train_configs/ablations/regional_st_tcn.yaml --seed 101
```

Expected backend`tcn`,100936parameters, seed101. Keep all common settings fixed. Separate timestamped directories protect previous runs. If only one new training run is feasible, prioritize attention101; that checks the current leading candidate, while the full causal comparison needs both variants.

## Source audit after each new run

Use the actual timestamp printed for **that new seed101 run**. The family directory remains `RegionalSTAttention_<timestamp>` for both backends; checkpoint prefixes distinguish them.

Attention:

```bash
python tools/audit_regional_dino.py \
  --config configs/train_configs/ablations/regional_st_attention.yaml \
  --checkpoint runs/RegionalSTAttention_<attention101_timestamp>/checkpoints/MMPD_PURE_RegionalSTAttention_best.pth \
  --split valid --save-waveforms \
  --output diagnostics/mmpd_attention_seed101_full
```

TCN:

```bash
python tools/audit_regional_dino.py \
  --config configs/train_configs/ablations/regional_st_tcn.yaml \
  --checkpoint runs/RegionalSTAttention_<tcn101_timestamp>/checkpoints/MMPD_PURE_RegionalSTTCN_best.pth \
  --split valid --save-waveforms \
  --output diagnostics/mmpd_tcn_seed101_full
```

Preserve each new run's `training.log`, `regional_history.jsonl`, `regional_waveforms.pt` and complete new MMPD audit directory. There is no need for another separate PURE audit: the automatic saved waveforms allow matched analysis. Audit manifests should report `checkpoint_training_seed: 101`; older checkpoints report null rather than guessing their training seed.

## Decide from the repeated comparison

Compare each seed pair using MMPD recording HR, high-rate error/candidate availability, subject variation, waveform correlation and SNR. Retain current full-PURE scores as secondary exploratory evidence. Report both runs/seed-level averages and variation rather than choosing whichever seed has the best PURE result.

If Exp2's advantage persists, it becomes a stronger baseline for a targeted pulse-specific contribution. The next design question is how to preserve correct fundamental frequency under nuisance while maintaining waveform fidelity, with source-validated controls. A third fixed pair(seed102) and untouched confirmatory data are still needed before strong model-comparison claims. If the advantage is unstable, resolve seed variability and source checkpoint/metric behavior before attributing the original gap to attention. No result here justifies changing the filter or selecting on PURE.

## Local checks and limitations

All13 focused tests passed: three seed/reproducibility tests, three spatiotemporal integration tests and seven existing regional trainer tests. They exercise default/YAML/CLI seed resolution, invalid values, exact legacy100 initialization, repeated/different seeds on both actual temporal models, independent training/general generators, tiny training/checkpoint histories, reload with a different inference seed and audit provenance. These CPU checks validate wiring/reproducibility on this environment, not physiological performance or cross-hardware bitwise equality. A direct local `main.py --help` attempt stops in the pre-existing dataset imports because this small test environment lacks OpenCV (`cv2`); full entry-point training must run in the previously working server environment. No dependency change is required for the new seed option.
