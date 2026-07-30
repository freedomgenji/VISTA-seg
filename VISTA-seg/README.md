# VISTA-Seg for Missing-Modality BraTS Segmentation

This repository contains the clean implementation of VISTA-Seg for
missing-modality BraTS segmentation.

Support samples perform an inner-loop update on the
`VISTASegSharedAdapter` fast parameters. Query samples and the decoder then use
the support-adapted adapter to produce `z_shared`. This differs from the original
cleaned model, where the support/query path mainly adapted the feature
reconstructor after `z_shared` had already been produced.

## Setup

```bash
conda create -n vista-seg python=3.10
conda activate vista-seg
pip install -r requirements.txt
```

## Data

Training expects the preprocessed BraTS NPY layout:

```text
DATA_ROOT/
  vol/*_vol.npy
  seg/*_seg.npy
  train.txt
  test.txt
```

`preprocess.py` can convert raw BraTS2020 NIfTI cases into this layout and write a
simple train/test split. Existing split files can also be supplied directly.

## Training

The fixed configuration used for the main VISTA-Seg experiment is wrapped by:

```bash
bash job_brats_vista_seg_fixed.sh
```

Useful overrides:

```bash
DATAPATH=/path/to/brats_npy \
TRAIN_FILE=train.txt \
TEST_FILE=test.txt \
SAVEPATH=./runs/vista_seg_main \
bash job_brats_vista_seg_fixed.sh
```

## Inference

```bash
bash infer_brats.sh /path/to/model_best_early_stop.pth
```

The evaluation runs all 15 available/missing modality combinations and reports
WT, TC, ET, and ET post-processed Dice scores.
By default this script is configured for the BraTS2023 qualitative case
`BraTS-GLI-00043-000` under `/root/shared-nvme/brats2023_npy`, and saves
predictions to `/root/qual_preds/vista_seg_full_meta_2023`.

To export qualitative prediction masks for a single BraTS case:

```bash
python train.py \
  --datapath /root/shared-nvme/brats2023_npy \
  --dataname BRATS2023 \
  --resume /path/to/vista_seg_full_meta_weight.pth \
  --case_name BraTS-GLI-00043-000 \
  --slice_index 48 \
  --pred_savepath /root/qual_preds/vista_seg_full_meta_2023 \
  --save_cropped_pred \
  --metrics_csv /root/qual_preds/vista_seg_full_meta_2023/metrics_BraTS-GLI-00043-000.csv \
  --batch_size 1 \
  --crop_size 112
```

Predictions are saved as `uint8` NPY volumes under
`{pred_savepath}/{mask_name}/{case_name}.npy` using the RFNet-style 15-mask
folder names. With `--save_cropped_pred`, the saved mask keeps the processed NPY
crop shape so it aligns with RFNet qualitative outputs. The optional
`--metrics_csv` file contains per-case per-mask Dice, IoU, and HD95 values for
WT, TC, ET, and ET_postpro.
