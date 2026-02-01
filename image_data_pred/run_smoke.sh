#!/usr/bin/env bash
set -e

export HYDRA_FULL_ERROR=1
export MPLBACKEND=Agg

python -u main.py experiment=task/Cls \
  data=Cls/kaggle_liver_fibrosis \
  model=Cls/vit \
  data.batch_size=16 data.num_workers=2 \
  train.epochs=2 train.warmup_epochs=1 \
  train.base_lr=2.5e-4 train.warmup_lr=5e-5 train.min_lr=1e-6 \
  train.val_freq=1 \
  train.accumulation_steps=1 \
  +model.model_cfg.pretrained=./assets/FMweight/USFM_latest.pth \
  tag=smoke
