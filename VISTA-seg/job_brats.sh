#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python train.py \
  --batch_size "${BATCH_SIZE:-2}" \
  --iter_per_epoch "${ITER_PER_EPOCH:--1}" \
  --num_epochs "${NUM_EPOCHS:-700}" \
  --shared_rep_method vista_seg \
  --fusion_type RFM \
  --use_reg_loss \
  --crop_size "${CROP_SIZE:-112}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --savepath "${SAVEPATH:-./runs/brats2020_rfm}" \
  --datapath "${DATAPATH:-/work/data1/BRATS2020_Training}" \
  --train_file "${TRAIN_FILE:-train.txt}" \
  --test_file "${TEST_FILE:-test.txt}"
