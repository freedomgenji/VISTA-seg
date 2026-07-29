import argparse
import csv
import os

import numpy as np
import torch

import models
from data.datasets_nii import normalize_modalities


CASE_ID_DEFAULT = "BraTS-GLI-00043-000"
DATA_ROOT_DEFAULT = "/root/shared-nvme/brats2023_npy"
CHECKPOINT_DEFAULT = "/root/2023/metaviewer_full_meta/runs/full_meta_metaviewer/model_700.pth"
OUTPUT_DIR_DEFAULT = "qualitative_BraTS-GLI-00043-000_ours_all_slices"

MODEL_MODALITIES = ("t2f", "t1c", "t1n", "t2w")
DISPLAY_MODALITIES = ("t1n", "t1c", "t2w", "t2f")
ALL_MODALITIES = ("t1n", "t1c", "t2w", "t2f")

COMBOS = [
    ("full_t1n_t1c_t2w_t2f", ("t1n", "t1c", "t2w", "t2f")),
    ("t1c_t2w_t2f", ("t1c", "t2w", "t2f")),
    ("t1n_t2w_t2f", ("t1n", "t2w", "t2f")),
    ("t1n_t1c_t2f", ("t1n", "t1c", "t2f")),
    ("t1n_t1c_t2w", ("t1n", "t1c", "t2w")),
    ("t2w_t2f", ("t2w", "t2f")),
    ("t1c_t2f", ("t1c", "t2f")),
    ("t1c_t2w", ("t1c", "t2w")),
    ("t1n_t2f", ("t1n", "t2f")),
    ("t1n_t2w", ("t1n", "t2w")),
    ("t1n_t1c", ("t1n", "t1c")),
    ("t2f", ("t2f",)),
    ("t2w", ("t2w",)),
    ("t1c", ("t1c",)),
    ("t1n", ("t1n",)),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-case qualitative visualization for MetaViewer on BraTS2023.")
    parser.add_argument("--case_id", default=CASE_ID_DEFAULT)
    parser.add_argument("--datapath", default=DATA_ROOT_DEFAULT)
    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    parser.add_argument("--output_dir", default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--crop_size", default=112, type=int)
    parser.add_argument("--normalize_on_load", action="store_true", default=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="Reserved for compatibility; this script loads one case directly.")
    return parser.parse_args()


def modality_index(name):
    return MODEL_MODALITIES.index(name)


def mask_from_used(used_modalities):
    used = set(used_modalities)
    return np.asarray([name in used for name in MODEL_MODALITIES], dtype=np.bool_)


def load_case(datapath, case_id, normalize_on_load=False):
    vol_path = os.path.join(datapath, "vol", "{}_vol.npy".format(case_id))
    seg_path = os.path.join(datapath, "seg", "{}_seg.npy".format(case_id))
    if not os.path.exists(vol_path):
        raise FileNotFoundError("Missing volume: {}".format(vol_path))
    if not os.path.exists(seg_path):
        raise FileNotFoundError("Missing segmentation: {}".format(seg_path))

    volume = np.load(vol_path).astype(np.float32)
    if normalize_on_load:
        volume = normalize_modalities(volume)
    seg = np.load(seg_path).astype(np.uint8)
    return volume, seg


def build_model(checkpoint_path, device):
    model = models.MetaViewerSeg(
        num_cls=4,
        fusion_type="RFM",
        shared_rep_method="metaviewer",
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
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

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
                    weight[:, :, h:h + crop_size, w:w + crop_size, d:d + crop_size] += weight_tensor
                    x_part = x_tensor[:, :, h:h + crop_size, w:w + crop_size, d:d + crop_size]
                    pred_part = model(x_part, mask_tensor)
                    pred[:, :, h:h + crop_size, w:w + crop_size, d:d + crop_size] += pred_part * weight_tensor

    pred = pred / weight
    pred_label = torch.argmax(pred, dim=1)
    foreground = torch.sum(torch.abs(x_tensor), dim=1) > 0
    pred_label[~foreground] = 0
    return pred_label[0].detach().cpu().numpy().astype(np.uint8)


def region_masks(label_slice):
    label_slice = np.asarray(label_slice)
    et = np.logical_or(label_slice == 3, label_slice == 4)
    tc = np.logical_or(label_slice == 1, et)
    wt = label_slice > 0
    return et, tc, wt


def dice_iou(pred_mask, gt_mask):
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    pred_sum = int(pred_mask.sum())
    gt_sum = int(gt_mask.sum())
    intersection = int(np.logical_and(pred_mask, gt_mask).sum())
    if pred_sum + gt_sum == 0:
        dice = 1.0
    else:
        dice = 2.0 * intersection / float(pred_sum + gt_sum)
    union = int(np.logical_or(pred_mask, gt_mask).sum())
    iou = 1.0 if union == 0 else intersection / float(union)
    return dice, iou


def robust_normalize(image):
    image = np.asarray(image, dtype=np.float32)
    mask = image != 0
    if mask.any():
        low, high = np.percentile(image[mask], [1, 99])
    else:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def overlay_label(base_image, label_slice, foreground_slice=None, alpha=0.55):
    base = robust_normalize(base_image)
    rgb = np.repeat(base[..., None], 3, axis=2)
    label = np.asarray(label_slice).copy()
    if foreground_slice is not None:
        label[~foreground_slice] = 0

    color_map = {
        1: np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        2: np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        3: np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        4: np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    }
    for label_value, color in color_map.items():
        mask = label == label_value
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color
    return rgb


def add_panel(ax, image, title):
    ax.imshow(np.rot90(image), cmap="gray" if image.ndim == 2 else None, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_slice_png(output_path, volume, seg, predictions, slice_idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 5, figsize=(22, 16), dpi=120)
    axes = axes.reshape(-1)

    for panel_idx, modality_name in enumerate(DISPLAY_MODALITIES):
        channel = modality_index(modality_name)
        add_panel(
            axes[panel_idx],
            robust_normalize(volume[:, :, slice_idx, channel]),
            modality_name,
        )

    t1c_slice = volume[:, :, slice_idx, modality_index("t1c")]
    foreground_slice = np.any(volume[:, :, slice_idx, :] != 0, axis=-1)
    add_panel(axes[4], overlay_label(t1c_slice, seg[:, :, slice_idx], foreground_slice), "GT")

    for panel_idx, (combo_name, _) in enumerate(COMBOS, start=5):
        add_panel(
            axes[panel_idx],
            overlay_label(t1c_slice, predictions[combo_name][:, :, slice_idx], foreground_slice),
            combo_name,
        )

    fig.suptitle("BraTS2023 qualitative axial slice {}".format(slice_idx), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def collect_slice_metrics(case_id, seg, predictions):
    rows = []
    depth = seg.shape[2]
    for slice_idx in range(depth):
        gt_et, gt_tc, gt_wt = region_masks(seg[:, :, slice_idx])
        for combo_name, used_modalities in COMBOS:
            pred_slice = predictions[combo_name][:, :, slice_idx]
            pred_et, pred_tc, pred_wt = region_masks(pred_slice)
            dice_et, iou_et = dice_iou(pred_et, gt_et)
            dice_tc, iou_tc = dice_iou(pred_tc, gt_tc)
            dice_wt, iou_wt = dice_iou(pred_wt, gt_wt)
            missing = [name for name in ALL_MODALITIES if name not in used_modalities]
            rows.append({
                "case_id": case_id,
                "slice": slice_idx,
                "available_modalities": combo_name,
                "used_modalities": ",".join(used_modalities),
                "missing_modalities": ",".join(missing),
                "gt_ET_voxels": int(gt_et.sum()),
                "gt_TC_voxels": int(gt_tc.sum()),
                "gt_WT_voxels": int(gt_wt.sum()),
                "pred_ET_voxels": int(pred_et.sum()),
                "pred_TC_voxels": int(pred_tc.sum()),
                "pred_WT_voxels": int(pred_wt.sum()),
                "dice_ET": dice_et,
                "dice_TC": dice_tc,
                "dice_WT": dice_wt,
                "iou_ET": iou_et,
                "iou_TC": iou_tc,
                "iou_WT": iou_wt,
                "dice_mean": float(np.mean([dice_et, dice_tc, dice_wt])),
                "iou_mean": float(np.mean([iou_et, iou_tc, iou_wt])),
            })
    return rows


def write_slice_metrics(csv_path, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_volume_debug(combo_name, pred_volume):
    et, tc, wt = region_masks(pred_volume)
    print(
        "{}: pred_ET_voxels={}, pred_TC_voxels={}, pred_WT_voxels={}".format(
            combo_name, int(et.sum()), int(tc.sum()), int(wt.sum())))


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    volume, seg = load_case(args.datapath, args.case_id, normalize_on_load=args.normalize_on_load)
    if volume.ndim != 4 or volume.shape[-1] != 4:
        raise ValueError("Expected volume shape [H, W, D, 4], got {}".format(volume.shape))
    if seg.shape != volume.shape[:3]:
        raise ValueError("Segmentation shape {} does not match volume shape {}".format(seg.shape, volume.shape[:3]))

    foreground = np.any(volume != 0, axis=-1)
    seg = seg.copy()
    seg[~foreground] = 0

    model = build_model(args.checkpoint, device)
    x_np = np.ascontiguousarray(volume.transpose(3, 0, 1, 2)[None])
    x_tensor = torch.from_numpy(x_np).float().to(device)

    predictions = {}
    for combo_name, used_modalities in COMBOS:
        mask_np = mask_from_used(used_modalities)
        mask_tensor = torch.from_numpy(mask_np[None]).bool().to(device)
        pred = sliding_window_predict(model, x_tensor, mask_tensor, args.crop_size, device)
        pred[~foreground] = 0
        predictions[combo_name] = pred
        print_volume_debug(combo_name, pred)

    metrics_path = os.path.join(args.output_dir, "slice_metrics.csv")
    rows = collect_slice_metrics(args.case_id, seg, predictions)
    write_slice_metrics(metrics_path, rows)
    print("Saved slice metrics:", metrics_path)

    depth = seg.shape[2]
    for slice_idx in range(depth):
        output_path = os.path.join(args.output_dir, "slice_{:03d}.png".format(slice_idx))
        save_slice_png(output_path, volume, seg, predictions, slice_idx)
        if (slice_idx + 1) % 20 == 0 or slice_idx + 1 == depth:
            print("Saved {}/{} slice PNGs".format(slice_idx + 1, depth))

    print("Output directory:", args.output_dir)


if __name__ == "__main__":
    main()
