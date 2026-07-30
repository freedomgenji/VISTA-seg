#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATAPATH="${DATAPATH:-/work/data1/brats2023_gli_npy}"
TRAIN_FILE="${TRAIN_FILE:-train.txt}"
TEST_FILE="${TEST_FILE:-test.txt}"
SAVEPATH="${SAVEPATH:-./runs/vista_seg_v1_crop085_et4_tc3_wt3_bs2_eval100_ep700}"

if [[ ! -f "${DATAPATH}/${TRAIN_FILE}" ]]; then
  echo "Missing train file: ${DATAPATH}/${TRAIN_FILE}" >&2
  echo "Set DATAPATH to the preprocessed NPY dataset root if your dataset is elsewhere." >&2
  echo "For raw BraTS2023 NIfTI data, run:" >&2
  echo "  python preprocess.py --src_root /work/data1/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData --dst_root ${DATAPATH} --train_ratio 0.8 --seed 42 --min_size 128" >&2
  exit 1
fi

if [[ ! -f "${DATAPATH}/${TEST_FILE}" ]]; then
  echo "Missing test file: ${DATAPATH}/${TEST_FILE}" >&2
  echo "Set TEST_FILE=... if your split file has a different name." >&2
  exit 1
fi

RESUME_ARGS=()
if [[ -n "${RESUME_TRAIN:-}" ]]; then
  if [[ ! -f "${RESUME_TRAIN}" ]]; then
    echo "Missing resume checkpoint: ${RESUME_TRAIN}" >&2
    echo "Set RESUME_TRAIN to the real checkpoint path, for example /root/2023/vista_seg/runs/xxx/model_last.pth" >&2
    exit 1
  fi
  RESUME_ARGS+=(--resume_train "${RESUME_TRAIN}")
fi
if [[ "${SKIP_RESUME_EVAL:-0}" == "1" ]]; then
  RESUME_ARGS+=(--skip_resume_eval)
fi

python train.py \
  "${RESUME_ARGS[@]}" \
  --shared_rep_method vista_seg \
  --batch_size "${BATCH_SIZE:-2}" \
  --datapath "${DATAPATH}" \
  --dataname "${DATANAME:-BRATS2023}" \
  --crop_size "${CROP_SIZE:-112}" \
  --meta_support_ratio "${META_SUPPORT_RATIO:-0.5}" \
  --meta_inner_steps "${META_INNER_STEPS:-1}" \
  --meta_inner_lr "${META_INNER_LR:-1e-3}" \
  --meta_feature_weight "${META_FEATURE_WEIGHT:-0.2}" \
  --meta_align_weight "${META_ALIGN_WEIGHT:-0.01}" \
  --meta_first_order \
  --use_tumor_aware_crop \
  --tumor_crop_prob "${TUMOR_CROP_PROB:-0.85}" \
  --tumor_crop_et_weight "${TUMOR_CROP_ET_WEIGHT:-4.0}" \
  --tumor_crop_tc_weight "${TUMOR_CROP_TC_WEIGHT:-3.0}" \
  --tumor_crop_wt_weight "${TUMOR_CROP_WT_WEIGHT:-3.0}" \
  --use_region_tversky_loss \
  --region_tversky_weight "${REGION_TVERSKY_WEIGHT:-0.5}" \
  --region_tversky_wt_weight "${REGION_TVERSKY_WT_WEIGHT:-0.5}" \
  --region_tversky_tc_weight "${REGION_TVERSKY_TC_WEIGHT:-1.5}" \
  --region_tversky_et_weight "${REGION_TVERSKY_ET_WEIGHT:-2.0}" \
  --tversky_alpha "${TVERSKY_ALPHA:-0.3}" \
  --tversky_beta "${TVERSKY_BETA:-0.7}" \
  --fusion_type RFM \
  --use_reg_loss \
  --amp \
  --grad_checkpoint \
  --num_epochs "${NUM_EPOCHS:-700}" \
  --iter_per_epoch "${ITER_PER_EPOCH:-150}" \
  --early_stop_patience "${EARLY_STOP_PATIENCE:-30}" \
  --early_stop_min_delta "${EARLY_STOP_MIN_DELTA:-0.0001}" \
  --early_stop_start_epoch "${EARLY_STOP_START_EPOCH:-300}" \
  --early_stop_eval_interval "${EARLY_STOP_EVAL_INTERVAL:-100}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --savepath "${SAVEPATH}" \
  --train_file "${TRAIN_FILE}" \
  --test_file "${TEST_FILE}"
