Extract inside the existing rPPG_VLM server repository, preserving folders.
Existing regional base YAML/model/audit helpers must remain present.
No cache preprocessing or dependency installation is needed.

First:
python tools/check_rate_augmentation.py --config configs/train_configs/ablations/regional_shallow_rate_aug.yaml --output diagnostics/rate_aug_preflight

If successful:
python main.py --config_file configs/train_configs/ablations/regional_shallow_rate_aug.yaml

See docs/experiments/2026-11-25_rate_augmentation_run.md for settings, limitations and verification.
