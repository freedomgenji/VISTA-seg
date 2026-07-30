#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash infer_brats.sh /path/to/model.pth"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_STEM="$(basename "$1" .pth)"
CASE_NAME_VALUE="${CASE_NAME:-BraTS-GLI-00043-000}"
PRED_SAVEPATH_VALUE="${PRED_SAVEPATH:-/root/qual_preds/vista_seg_full_meta_2023}"
SAVEPATH_VALUE="${SAVEPATH:-./runs/brats2023_qual_pred}"
METRICS_CSV_VALUE="${METRICS_CSV:-${SAVEPATH_VALUE}/metrics_${MODEL_STEM}.csv}"

EVAL_ARGS=()
EVAL_ARGS+=(--metrics_csv "${METRICS_CSV_VALUE}")
EVAL_ARGS+=(--case_name "${CASE_NAME_VALUE}")
EVAL_ARGS+=(--pred_savepath "${PRED_SAVEPATH_VALUE}")
if [[ "${SAVE_CROPPED_PRED:-1}" == "1" ]]; then
  EVAL_ARGS+=(--save_cropped_pred)
fi
if [[ -n "${SLICE_INDEX:-48}" ]]; then
  EVAL_ARGS+=(--slice_index "${SLICE_INDEX:-48}")
fi

python train.py \
  "${EVAL_ARGS[@]}" \
  --resume "$1" \
  --batch_size "${BATCH_SIZE:-1}" \
  --iter_per_epoch -1 \
  --dataname "${DATANAME:-BRATS2023}" \
  --shared_rep_method vista_seg \
  --fusion_type RFM \
  --use_reg_loss \
  --crop_size "${CROP_SIZE:-112}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --savepath "${SAVEPATH_VALUE}" \
  --datapath "${DATAPATH:-/root/shared-nvme/brats2023_npy}" \
  --train_file "${TRAIN_FILE:-train.txt}" \
  --test_file "${TEST_FILE:-test.txt}"
