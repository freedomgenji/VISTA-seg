import argparse
import json
import os
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LABEL_COLORS = {
    1: np.array([230, 57, 70], dtype=np.float32),
    2: np.array([46, 204, 113], dtype=np.float32),
    3: np.array([255, 205, 0], dtype=np.float32),
    4: np.array([255, 205, 0], dtype=np.float32),
}

ALPHA = 0.45
TILE_SIZE = 320
MARGIN = 35
BASE_MODALITY = "flair"
AUTO_SLICE_TARGET = "enhancing"

MODALITY_CHANNELS = {
    "flair": 0,
    "t1ce": 1,
    "t1": 2,
    "t2": 3,
}

MASKS = [
    ("flairt1cet1t2", "All"),
    ("flairt1cet1", "Missing T2"),
    ("flairt1t2", "Missing T1ce"),
    ("flairt1ce", "FLAIR+T1ce"),
    ("t1c", "Only T1ce"),
    ("flair", "Only FLAIR"),
]

METHODS = [
    ("GT", None),
    ("RFNet", "RFNet"),
    ("mmFormer", "mmFormer"),
    ("M3AE", "M3AE"),
    ("AdaMM", "AdaMM"),
    ("DC-seg", "DC-seg"),
    ("VISTA-Seg", "vista_seg_full_meta"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create BraTS2023 qualitative panels with the BraTS2020 style.")
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--data_root", default="/root/shared-nvme/DC-Seg/brats2023_npy")
    parser.add_argument("--pred_root", default="/root/qual_preds_2023")
    parser.add_argument("--out_root", default="/root/shared-nvme/qual_panels_2023")
    parser.add_argument("--slice", default="auto",
                        help='Use "auto" or an integer axial slice index.')
    parser.add_argument("--base_modality", default=BASE_MODALITY,
                        choices=sorted(MODALITY_CHANNELS.keys()))
    parser.add_argument("--margin", default=MARGIN, type=int)
    parser.add_argument("--tile", default=TILE_SIZE, type=int)
    parser.add_argument("--alpha", default=ALPHA, type=float)
    parser.add_argument("--crop_mode", default="foreground",
                        choices=["foreground", "tumor", "full"],
                        help="foreground matches the BraTS2020 full-brain qualitative view; tumor is a tight lesion crop.")
    parser.add_argument("--auto_slice_target", default=AUTO_SLICE_TARGET,
                        choices=["enhancing", "tc", "wt"])
    parser.add_argument(
        "--methods",
        default="all",
        help='Use "all" or a comma-separated list of display names, for example "Ours" or "RFNet,Ours".',
    )
    return parser.parse_args()


def font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def normalize_labels(label):
    label = np.asarray(label).copy()
    label[label == 4] = 3
    return label.astype(np.uint8)


def load_volume_and_seg(data_root, case_name):
    vol_path = os.path.join(data_root, "vol", "{}_vol.npy".format(case_name))
    seg_path = os.path.join(data_root, "seg", "{}_seg.npy".format(case_name))
    if not os.path.exists(vol_path):
        raise FileNotFoundError("Missing volume: {}".format(vol_path))
    if not os.path.exists(seg_path):
        raise FileNotFoundError("Missing segmentation: {}".format(seg_path))

    volume = np.load(vol_path)
    seg_raw = np.load(seg_path)
    if volume.ndim != 4 or volume.shape[-1] != 4:
        raise ValueError("Expected volume shape [H, W, D, 4], got {}".format(volume.shape))
    if seg_raw.shape != volume.shape[:3]:
        raise ValueError("Seg shape {} does not match volume spatial shape {}".format(
            seg_raw.shape, volume.shape[:3]))
    return volume.astype(np.float32), normalize_labels(seg_raw), seg_raw


def select_methods(method_arg):
    if method_arg.strip().lower() == "all":
        return METHODS

    requested = [item.strip() for item in method_arg.split(",") if item.strip()]
    method_map = {display_name.lower(): (display_name, method_dir)
                  for display_name, method_dir in METHODS[1:]}
    selected = [METHODS[0]]
    missing = []
    for item in requested:
        key = item.lower()
        if key not in method_map:
            missing.append(item)
        else:
            selected.append(method_map[key])
    if missing:
        raise ValueError("Unknown method(s): {}. Valid methods: {}".format(
            ", ".join(missing),
            ", ".join([display_name for display_name, _ in METHODS[1:]])))
    if len(selected) == 1:
        raise ValueError("No prediction method selected.")
    return selected


def region_channels_to_label(region_pred):
    region_pred = np.asarray(region_pred)
    if region_pred.shape[0] != 3:
        raise ValueError("Expected WT/TC/ET channel-first prediction, got {}".format(region_pred.shape))
    threshold = 0.5 if np.issubdtype(region_pred.dtype, np.floating) else 0
    wt = region_pred[0] > threshold
    tc = region_pred[1] > threshold
    et = region_pred[2] > threshold
    label = np.zeros(region_pred.shape[1:], dtype=np.uint8)
    label[wt] = 2
    label[tc] = 1
    label[et] = 3
    return label


def prediction_to_label(pred, seg_shape):
    pred = np.asarray(pred)
    raw_shape = pred.shape
    pred = np.squeeze(pred)

    if pred.shape == seg_shape:
        return normalize_labels(pred), raw_shape, pred.shape

    if pred.ndim == 4:
        if pred.shape[0] == 3 and pred.shape[1:] == seg_shape:
            return region_channels_to_label(pred), raw_shape, pred.shape[1:]
        if pred.shape[-1] == 3 and pred.shape[:-1] == seg_shape:
            return region_channels_to_label(np.moveaxis(pred, -1, 0)), raw_shape, pred.shape[:-1]
        if pred.shape[0] == 4 and pred.shape[1:] == seg_shape:
            return normalize_labels(np.argmax(pred, axis=0)), raw_shape, pred.shape[1:]
        if pred.shape[-1] == 4 and pred.shape[:-1] == seg_shape:
            return normalize_labels(np.argmax(pred, axis=-1)), raw_shape, pred.shape[:-1]

    raise ValueError(
        "Prediction spatial shape mismatch. raw_shape={}, squeezed_shape={}, expected={}".format(
            raw_shape, pred.shape, seg_shape))


def load_predictions(pred_root, case_name, seg_shape, foreground, methods):
    predictions = OrderedDict()
    checks = []
    for mask_name, _ in MASKS:
        predictions[mask_name] = OrderedDict()
        for display_name, method_dir in methods[1:]:
            pred_path = os.path.join(pred_root, method_dir, mask_name, "{}.npy".format(case_name))
            if not os.path.exists(pred_path):
                raise FileNotFoundError(
                    "Missing prediction for model={}, mask={}: {}".format(
                        display_name, mask_name, pred_path))
            pred_raw = np.load(pred_path)
            pred_label, raw_shape, spatial_shape = prediction_to_label(pred_raw, seg_shape)
            if tuple(spatial_shape) != tuple(seg_shape):
                raise ValueError(
                    "Shape mismatch for model={}, mask={}: prediction spatial shape {}, GT shape {}".format(
                        display_name, mask_name, spatial_shape, seg_shape))
            pred_label[~foreground] = 0
            predictions[mask_name][display_name] = pred_label
            checks.append({
                "model": display_name,
                "mask": mask_name,
                "path": pred_path,
                "raw_shape": list(raw_shape),
                "spatial_shape": list(spatial_shape),
                "dtype": str(pred_raw.dtype),
                "unique_labels": [int(x) for x in np.unique(pred_label)],
            })
    return predictions, checks


def select_slice(seg, slice_arg, target):
    if slice_arg != "auto":
        idx = int(slice_arg)
        if idx < 0 or idx >= seg.shape[2]:
            raise ValueError("Slice {} outside depth {}".format(idx, seg.shape[2]))
        return idx

    if target == "enhancing":
        mask = seg == 3
    elif target == "tc":
        mask = np.logical_or(seg == 1, seg == 3)
    else:
        mask = seg > 0
    counts = mask.sum(axis=(0, 1))
    if counts.max() == 0 and target != "wt":
        counts = (seg > 0).sum(axis=(0, 1))
    return int(np.argmax(counts))


def robust_base_u8(volume, modality):
    base = volume[..., MODALITY_CHANNELS[modality]].astype(np.float32)
    foreground = np.any(volume != 0, axis=-1)
    values = base[foreground]
    if values.size == 0:
        values = base.reshape(-1)
    lo, hi = np.percentile(values, [1, 99])
    if hi <= lo:
        return np.zeros(base.shape, dtype=np.uint8)
    out = np.clip((base - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def square_crop_box(mask_2d, margin, shape):
    rows, cols = np.where(mask_2d)
    height, width = shape
    if rows.size == 0:
        center_r = height / 2.0
        center_c = width / 2.0
        size = max(height, width)
    else:
        r0 = int(rows.min()) - margin
        r1 = int(rows.max()) + margin + 1
        c0 = int(cols.min()) - margin
        c1 = int(cols.max()) + margin + 1
        center_r = (r0 + r1) / 2.0
        center_c = (c0 + c1) / 2.0
        size = max(r1 - r0, c1 - c0)
    size = max(1, int(np.ceil(size)))
    r0 = int(round(center_r - size / 2.0))
    c0 = int(round(center_c - size / 2.0))
    return r0, r0 + size, c0, c0 + size


def crop_with_padding(array, box, fill=0):
    r0, r1, c0, c1 = box
    out_shape = (r1 - r0, c1 - c0)
    out = np.full(out_shape, fill, dtype=array.dtype)
    src_r0 = max(0, r0)
    src_r1 = min(array.shape[0], r1)
    src_c0 = max(0, c0)
    src_c1 = min(array.shape[1], c1)
    if src_r1 <= src_r0 or src_c1 <= src_c0:
        return out
    dst_r0 = src_r0 - r0
    dst_c0 = src_c0 - c0
    out[dst_r0:dst_r0 + (src_r1 - src_r0),
        dst_c0:dst_c0 + (src_c1 - src_c0)] = array[src_r0:src_r1, src_c0:src_c1]
    return out


def resize_array(array, size, nearest=False):
    resample = Image.NEAREST if nearest else Image.BILINEAR
    return np.asarray(Image.fromarray(array).resize((size, size), resample=resample))


def overlay_tile(base_slice_u8, label_slice, crop_box, tile_size, alpha):
    base_crop = crop_with_padding(base_slice_u8, crop_box, fill=0)
    label_crop = crop_with_padding(label_slice.astype(np.uint8), crop_box, fill=0)
    base_tile = resize_array(base_crop, tile_size, nearest=False)
    label_tile = resize_array(label_crop, tile_size, nearest=True)

    rgb = np.repeat(base_tile[..., None], 3, axis=2).astype(np.float32)
    for label_value in (1, 2, 3, 4):
        region = label_tile == label_value
        if np.any(region):
            rgb[region] = (1.0 - alpha) * rgb[region] + alpha * LABEL_COLORS[label_value]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb).transpose(Image.ROTATE_90)


def draw_centered_text(draw, box, text, text_font, fill=(0, 0, 0)):
    x0, y0, x1, y1 = box
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=text_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    else:
        tw, th = draw.textsize(text, font=text_font)
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2),
              text, font=text_font, fill=fill)


def make_grid(rows, row_titles, col_titles, tile_size, path_png, path_pdf=None):
    header_h = 46
    row_label_w = 145
    width = row_label_w + len(col_titles) * tile_size
    height = header_h + len(rows) * tile_size
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = font(22, bold=True)
    row_font = font(22, bold=True)

    for col_idx, title in enumerate(col_titles):
        x0 = row_label_w + col_idx * tile_size
        draw_centered_text(draw, (x0, 0, x0 + tile_size, header_h), title, header_font)
    for row_idx, (row_tiles, row_title) in enumerate(zip(rows, row_titles)):
        y0 = header_h + row_idx * tile_size
        draw_centered_text(draw, (0, y0, row_label_w, y0 + tile_size), row_title, row_font)
        for col_idx, tile in enumerate(row_tiles):
            x0 = row_label_w + col_idx * tile_size
            canvas.paste(tile, (x0, y0))

    canvas.save(path_png)
    if path_pdf is not None:
        canvas.save(path_pdf, "PDF", resolution=300.0)


def print_checks(seg_raw, seg, prediction_checks):
    print("GT seg shape:", seg.shape)
    print("GT seg dtype:", seg_raw.dtype)
    print("GT unique labels raw:", [int(x) for x in np.unique(seg_raw)])
    print("GT unique labels mapped:", [int(x) for x in np.unique(seg)])
    print("Prediction checks:")
    for check in prediction_checks:
        print(
            "  model={model}, mask={mask}, raw_shape={raw_shape}, spatial_shape={spatial_shape}, "
            "dtype={dtype}, unique_labels={unique_labels}".format(**check))


def write_config(path, args, selected_slice, output_dir, methods):
    config = {
        "case_name": args.case_name,
        "data_root": args.data_root,
        "pred_root": args.pred_root,
        "output_dir": output_dir,
        "selected_slice": selected_slice,
        "slice_arg": args.slice,
        "auto_slice_target": args.auto_slice_target,
        "base_modality": args.base_modality,
        "crop": args.crop_mode,
        "margin": args.margin,
        "tile": args.tile,
        "alpha": args.alpha,
        "label_colors": {str(k): v.astype(int).tolist() for k, v in LABEL_COLORS.items()},
        "label_mapping": {
            "0": "background",
            "1": "NCR/NET",
            "2": "edema",
            "3": "enhancing tumor",
            "4": "mapped to 3 / enhancing tumor",
        },
        "masks": [{"name": name, "row_title": title} for name, title in MASKS],
        "methods": [{"display": name, "directory": directory} for name, directory in methods],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def main():
    args = parse_args()
    methods = select_methods(args.methods)
    output_dir = os.path.join(args.out_root, "{}_six_modalities".format(args.case_name))
    os.makedirs(output_dir, exist_ok=True)

    volume, seg, seg_raw = load_volume_and_seg(args.data_root, args.case_name)
    foreground = np.any(volume != 0, axis=-1)
    seg = seg.copy()
    seg[~foreground] = 0
    predictions, prediction_checks = load_predictions(
        args.pred_root, args.case_name, seg.shape, foreground, methods)
    print_checks(seg_raw, seg, prediction_checks)

    selected_slice = select_slice(seg, args.slice, args.auto_slice_target)
    slice_token = "sliceAUTO" if args.slice == "auto" else "slice{:03d}".format(selected_slice)
    print("Selected axial slice:", selected_slice)

    base_u8 = robust_base_u8(volume, args.base_modality)
    if args.crop_mode == "tumor":
        crop_source = seg[:, :, selected_slice] > 0
    elif args.crop_mode == "foreground":
        crop_source = foreground[:, :, selected_slice]
    else:
        crop_source = np.ones(seg.shape[:2], dtype=bool)
    crop_box = square_crop_box(crop_source, args.margin, seg.shape[:2])
    print("Crop box r0,r1,c0,c1:", crop_box)

    config_path = os.path.join(output_dir, "config.json")
    write_config(config_path, args, selected_slice, output_dir, methods)

    col_titles = [name for name, _ in methods]
    combined_rows = []
    panel_paths = []
    for mask_name, row_title in MASKS:
        row_tiles = []
        base_slice = base_u8[:, :, selected_slice]
        gt_tile = overlay_tile(base_slice, seg[:, :, selected_slice], crop_box, args.tile, args.alpha)
        row_tiles.append(gt_tile)
        for display_name, _ in methods[1:]:
            pred_tile = overlay_tile(
                base_slice,
                predictions[mask_name][display_name][:, :, selected_slice],
                crop_box,
                args.tile,
                args.alpha)
            row_tiles.append(pred_tile)
        combined_rows.append(row_tiles)

        panel_png = os.path.join(
            output_dir, "{}_{}_{}.png".format(args.case_name, mask_name, slice_token))
        make_grid([row_tiles], [row_title], col_titles, args.tile, panel_png)
        panel_paths.append(panel_png)
        print("Saved panel:", panel_png)

    fig_png = os.path.join(output_dir, "Fig_{}_six_modalities.png".format(args.case_name))
    fig_pdf = os.path.join(output_dir, "Fig_{}_six_modalities.pdf".format(args.case_name))
    make_grid(combined_rows, [title for _, title in MASKS], col_titles, args.tile, fig_png, fig_pdf)
    print("Saved combined PNG:", fig_png)
    print("Saved combined PDF:", fig_pdf)
    print("Saved config:", config_path)
    print("Shape and label checks passed.")


if __name__ == "__main__":
    main()
