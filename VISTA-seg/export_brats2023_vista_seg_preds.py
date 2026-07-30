import argparse
import json
import os

import numpy as np
import torch

import models
from data.datasets_nii import normalize_modalities


CASE_ID_DEFAULT = "BraTS-GLI-00043-000"
DATA_ROOT_DEFAULT = "/root/shared-nvme/brats2023_npy"
CHECKPOINT_DEFAULT = "/root/2023/vista_seg/runs/full_meta_vista_seg/model_700.pth"
OUTPUT_ROOT_DEFAULT = "/root/shared-nvme/qual_preds_2023/vista_seg_full_meta"

MODEL_MODALITIES = ("t2f", "t1c", "t1n", "t2w")

MASKS = [
    ("flairt1cet1t2", ("t2f", "t1c", "t1n", "t2w")),
    ("flairt1cet1", ("t2f", "t1c", "t1n")),
    ("flairt1t2", ("t2f", "t1n", "t2w")),
    ("flairt1ce", ("t2f", "t1c")),
    ("t1c", ("t1c",)),
    ("flair", ("t2f",)),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export VISTA-Seg BraTS2023 single-case predictions for qualitative panels.")
    parser.add_argument("--case_name", default=CASE_ID_DEFAULT)
    parser.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    parser.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--crop_size", default=112, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--normalize_on_load", action="store_true", default=False)
    return parser.parse_args()


def mask_from_used(used_modalities):
    used = set(used_modalities)
    return np.asarray([name in used for name in MODEL_MODALITIES], dtype=np.bool_)


def load_case(data_root, case_name, normalize_on_load=False):
    vol_path = os.path.join(data_root, "vol", "{}_vol.npy".format(case_name))
    seg_path = os.path.join(data_root, "seg", "{}_seg.npy".format(case_name))
    if not os.path.exists(vol_path):
        raise FileNotFoundError("Missing volume: {}".format(vol_path))
    if not os.path.exists(seg_path):
        raise FileNotFoundError("Missing segmentation: {}".format(seg_path))

    volume = np.load(vol_path).astype(np.float32)
    if normalize_on_load:
        volume = normalize_modalities(volume)
    seg = np.load(seg_path).astype(np.uint8)
    seg[seg == 4] = 3

    if volume.ndim != 4 or volume.shape[-1] != 4:
        raise ValueError("Expected volume shape [H, W, D, 4], got {}".format(volume.shape))
    if seg.shape != volume.shape[:3]:
        raise ValueError("Seg shape {} does not match volume spatial shape {}".format(
            seg.shape, volume.shape[:3]))
    return volume, seg


def build_model(checkpoint_path, device):
    model = models.VISTASeg(
        num_cls=4,
        fusion_type="RFM",
        shared_rep_method="vista_seg",
        meta_support_ratio=0.5,
        meta_inner_steps=1,
        meta_inner_lr=1.0e-3,
        meta_first_order=True,
    )

    use_cuda = device.type == "cuda"
    if use_cuda:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    keys = list(state_dict.keys())
    has_module_prefix = bool(keys) and keys[0].startswith("module.")

    if use_cuda and has_module_prefix:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    elif use_cuda:
        missing, unexpected = model.module.load_state_dict(state_dict, strict=False)
    elif has_module_prefix:
        stripped = {key[len("module."):]: value for key, value in state_dict.items()}
        missing, unexpected = model.load_state_dict(stripped, strict=False)
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        print("checkpoint load with strict=False")
        print("missing:", missing)
        print("unexpected:", unexpected)

    target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    target_model.is_training = False
    target_model.use_checkpoint = False
    model.eval()
    return model


def scan_indices(size, crop_size):
    if size <= crop_size:
        return [0]
    stride = int(crop_size * 0.5)
    count = int(np.ceil((size - crop_size) / float(stride)))
    indices = [idx * stride for idx in range(count)]
    indices.append(size - crop_size)
    return sorted(set(indices))


def sliding_window_predict(model, x_tensor, mask_tensor, crop_size, device):
    _, _, height, width, depth = x_tensor.shape
    h_indices = scan_indices(height, crop_size)
    w_indices = scan_indices(width, crop_size)
    d_indices = scan_indices(depth, crop_size)

    weight_tensor = torch.ones(1, crop_size, crop_size, crop_size, device=device)
    weight = torch.zeros(1, 1, height, width, depth, device=device)
    pred = torch.zeros(1, 4, height, width, depth, device=device)

    with torch.no_grad():
        for h in h_indices:
            for w in w_indices:
                for d in d_indices:
                    x_part = x_tensor[:, :, h:h + crop_size, w:w + crop_size, d:d + crop_size]
                    pred_part = model(x_part, mask_tensor)
                    pred[:, :, h:h + crop_size, w:w + crop_size, d:d + crop_size] += pred_part * weight_tensor
                    weight[:, :, h:h + crop_size, w:w + crop_size, d:d + crop_size] += weight_tensor

    pred = pred / weight.clamp_min(1.0e-6)
    pred_label = torch.argmax(pred, dim=1)
    foreground = torch.sum(torch.abs(x_tensor), dim=1) > 0
    pred_label[~foreground] = 0
    return pred_label[0].detach().cpu().numpy().astype(np.uint8)


def region_counts(label):
    label = np.asarray(label)
    et = label == 3
    tc = np.logical_or(label == 1, et)
    wt = label > 0
    return int(et.sum()), int(tc.sum()), int(wt.sum())


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    os.makedirs(args.output_root, exist_ok=True)

    volume, seg = load_case(args.data_root, args.case_name, normalize_on_load=args.normalize_on_load)
    foreground = np.any(volume != 0, axis=-1)
    seg = seg.copy()
    seg[~foreground] = 0

    print("Case:", args.case_name)
    print("Volume shape:", volume.shape, "dtype:", volume.dtype)
    print("GT seg shape:", seg.shape, "dtype:", seg.dtype, "unique:", [int(x) for x in np.unique(seg)])
    print("Checkpoint:", args.checkpoint)
    print("Output root:", args.output_root)

    model = build_model(args.checkpoint, device)
    x_np = np.ascontiguousarray(volume.transpose(3, 0, 1, 2)[None])
    x_tensor = torch.from_numpy(x_np).float().to(device)

    saved = []
    for mask_name, used_modalities in MASKS:
        mask_np = mask_from_used(used_modalities)
        mask_tensor = torch.from_numpy(mask_np[None]).bool().to(device)
        pred = sliding_window_predict(model, x_tensor, mask_tensor, args.crop_size, device)
        pred[~foreground] = 0
        pred[pred == 4] = 3

        if pred.shape != seg.shape:
            raise ValueError("Prediction shape {} does not match GT shape {} for mask {}".format(
                pred.shape, seg.shape, mask_name))

        save_dir = os.path.join(args.output_root, mask_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "{}.npy".format(args.case_name))
        np.save(save_path, pred.astype(np.uint8))
        saved.append(save_path)

        et_count, tc_count, wt_count = region_counts(pred)
        print("{}: shape={}, dtype=uint8, unique={}, pred_ET_voxels={}, pred_TC_voxels={}, pred_WT_voxels={}".format(
            mask_name,
            pred.shape,
            [int(x) for x in np.unique(pred)],
            et_count,
            tc_count,
            wt_count,
        ))

    config_path = os.path.join(args.output_root, "{}_export_config.json".format(args.case_name))
    config = {
        "case_name": args.case_name,
        "data_root": args.data_root,
        "checkpoint": args.checkpoint,
        "output_root": args.output_root,
        "crop_size": args.crop_size,
        "model_modalities": list(MODEL_MODALITIES),
        "masks": [{"name": name, "used_modalities": list(used)} for name, used in MASKS],
        "saved": saved,
    }
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print("Saved config:", config_path)
    print("Prediction export finished.")


if __name__ == "__main__":
    main()
