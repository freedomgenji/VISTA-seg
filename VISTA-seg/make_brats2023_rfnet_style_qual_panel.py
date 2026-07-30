import argparse
import base64
import io
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

MODALITIES = OrderedDict([
    ("FLAIR", 0),
    ("T1ce", 1),
    ("T1", 2),
    ("T2", 3),
])

METHODS = OrderedDict([
    ("RFNet", "RFNet_2023"),
    ("mmFormer", "mmFormer_2023"),
    ("DC-seg", "DC-seg_2023"),
    ("VISTA-Seg", "vista_seg_full_meta_2023"),
])

MASK_LABELS = {
    "flairt1cet1t2": "FLAIR+T1ce+T1+T2",
    "flairt1cet1": "FLAIR+T1ce+T1",
    "flairt1t2": "FLAIR+T1+T2",
    "flairt1ce": "FLAIR+T1ce",
    "t1c": "T1ce only",
    "flair": "FLAIR only",
}

AXIS_ALIASES = {
    "0": 0,
    "x": 0,
    "sagittal": 0,
    "1": 1,
    "y": 1,
    "coronal": 1,
    "2": 2,
    "z": 2,
    "axial": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="BraTS2023 qualitative panel in the BraTS2020 RFNet-style cropped canvas.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--mask_name", required=True)
    parser.add_argument("--axis", default="axial",
                        help="axial/z/2, coronal/y/1, or sagittal/x/0.")
    parser.add_argument("--slice_index", required=True, type=int)
    parser.add_argument("--session_name", required=True)
    parser.add_argument("--pred_root", default="/root/qual_preds")
    parser.add_argument("--out_root", default="/root/qual_vis")
    parser.add_argument("--overlay_modality", default="FLAIR",
                        choices=list(MODALITIES.keys()) + [name.lower() for name in MODALITIES.keys()])
    parser.add_argument(
        "--methods",
        default="all",
        help='Use "all" or comma-separated method display names, e.g. "VISTA-Seg" or "RFNet,mmFormer,DC-seg,VISTA-Seg".',
    )
    parser.add_argument("--alpha", default=0.45, type=float)
    parser.add_argument("--rot90", default=1, type=int,
                        help="Apply np.rot90 k times to match the BraTS2020 display orientation.")
    parser.add_argument("--save_pdf", action="store_true", default=True)
    parser.add_argument("--save_svg", action="store_true", default=True)
    return parser.parse_args()


def select_methods(method_arg):
    if method_arg.strip().lower() == "all":
        return METHODS
    requested = [item.strip() for item in method_arg.split(",") if item.strip()]
    method_map = {name.lower(): (name, directory) for name, directory in METHODS.items()}
    aliases = {
        "ours": "vista-seg",
        "vista_seg": "vista-seg",
        "vista_seg_full_meta": "vista-seg",
    }
    selected = OrderedDict()
    missing = []
    for item in requested:
        key = aliases.get(item.lower(), item.lower())
        if key not in method_map:
            missing.append(item)
        else:
            name, directory = method_map[key]
            selected[name] = directory
    if missing:
        raise ValueError("Unknown method(s): {}. Valid methods: {}".format(
            ", ".join(missing), ", ".join(METHODS.keys())))
    if not selected:
        raise ValueError("No methods selected.")
    return selected


def parse_axis(axis_arg):
    key = str(axis_arg).strip().lower()
    if key not in AXIS_ALIASES:
        raise ValueError("Unsupported axis: {}. Use axial/z/2, coronal/y/1, or sagittal/x/0.".format(axis_arg))
    return AXIS_ALIASES[key]


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


def draw_centered_text(draw, box, text, text_font, fill=(30, 30, 30)):
    x0, y0, x1, y1 = box
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=text_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    else:
        tw, th = draw.textsize(text, font=text_font)
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2),
              text, font=text_font, fill=fill)


def normalize_labels(label):
    label = np.asarray(label).copy()
    label[label == 4] = 3
    return label.astype(np.uint8)


def load_case(data_root, case_name):
    vol_path = os.path.join(data_root, "vol", "{}_vol.npy".format(case_name))
    seg_path = os.path.join(data_root, "seg", "{}_seg.npy".format(case_name))
    if not os.path.exists(vol_path):
        raise FileNotFoundError("Missing volume: {}".format(vol_path))
    if not os.path.exists(seg_path):
        raise FileNotFoundError("Missing segmentation: {}".format(seg_path))

    volume = np.load(vol_path).astype(np.float32)
    seg_raw = np.load(seg_path)
    seg = normalize_labels(seg_raw)
    if volume.ndim != 4 or volume.shape[-1] != 4:
        raise ValueError("Expected vol shape [H, W, D, 4], got {}".format(volume.shape))
    if seg.shape != volume.shape[:3]:
        raise ValueError("Seg shape {} does not match vol spatial shape {}".format(seg.shape, volume.shape[:3]))
    return volume, seg, seg_raw


def region_channels_to_label(region_pred):
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


def prediction_to_label(pred, expected_shape):
    pred = np.asarray(pred)
    raw_shape = pred.shape
    pred = np.squeeze(pred)
    if pred.shape == expected_shape:
        return normalize_labels(pred), raw_shape

    if pred.ndim == 4:
        if pred.shape[0] == 3 and pred.shape[1:] == expected_shape:
            return region_channels_to_label(pred), raw_shape
        if pred.shape[-1] == 3 and pred.shape[:-1] == expected_shape:
            return region_channels_to_label(np.moveaxis(pred, -1, 0)), raw_shape
        if pred.shape[0] == 4 and pred.shape[1:] == expected_shape:
            return normalize_labels(np.argmax(pred, axis=0)), raw_shape
        if pred.shape[-1] == 4 and pred.shape[:-1] == expected_shape:
            return normalize_labels(np.argmax(pred, axis=-1)), raw_shape

    raise ValueError("Prediction shape mismatch. raw_shape={}, squeezed_shape={}, expected={}".format(
        raw_shape, pred.shape, expected_shape))


def load_predictions(pred_root, mask_name, case_name, seg_shape, methods):
    predictions = OrderedDict()
    checks = []
    reference_shape = None
    for display_name, method_dir in methods.items():
        path = os.path.join(pred_root, method_dir, mask_name, "{}.npy".format(case_name))
        if not os.path.exists(path):
            raise FileNotFoundError("Missing prediction for {}: {}".format(display_name, path))
        raw = np.load(path)
        pred, raw_shape = prediction_to_label(raw, seg_shape)
        if reference_shape is None:
            reference_shape = pred.shape
        if pred.shape != reference_shape:
            raise ValueError("{} pred shape {} does not match RFNet/reference shape {}".format(
                display_name, pred.shape, reference_shape))
        if pred.shape != seg_shape:
            raise ValueError("{} pred shape {} does not match seg shape {}".format(
                display_name, pred.shape, seg_shape))
        predictions[display_name] = pred
        checks.append({
            "method": display_name,
            "path": path,
            "raw_shape": list(raw_shape),
            "shape": list(pred.shape),
            "dtype": str(raw.dtype),
            "unique_labels": [int(x) for x in np.unique(pred)],
        })
    return predictions, checks


def take_slice(array, axis, index):
    if index < 0 or index >= array.shape[axis]:
        raise ValueError("slice_index {} outside axis {} size {}".format(index, axis, array.shape[axis]))
    return np.take(array, index, axis=axis)


def orient_slice(image, rot90):
    out = np.asarray(image)
    if rot90 % 4:
        out = np.rot90(out, k=rot90)
    return np.ascontiguousarray(out)


def normalize_to_u8(image):
    image = np.asarray(image, dtype=np.float32)
    foreground = image != 0
    if np.any(foreground):
        values = image[foreground]
    else:
        values = image.reshape(-1)
    lo, hi = np.percentile(values, [1, 99])
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    out = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    out[~foreground] = 0.0
    return (out * 255.0).astype(np.uint8)


def grayscale_tile(volume, modality_index, axis, slice_index, rot90):
    image = take_slice(volume[..., modality_index], axis, slice_index)
    image = orient_slice(normalize_to_u8(image), rot90)
    return np.repeat(image[..., None], 3, axis=2).astype(np.uint8)


def overlay_tile(base_u8, label_slice, alpha, rot90):
    base = orient_slice(base_u8, rot90)
    label = orient_slice(normalize_labels(label_slice), rot90)
    rgb = np.repeat(base[..., None], 3, axis=2).astype(np.float32)
    for label_value in (1, 2, 3, 4):
        mask = label == label_value
        if np.any(mask):
            rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * LABEL_COLORS[label_value]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def make_panel(tiles, titles, mask_label):
    gap = 8
    combo_h = 30
    header_h = 30
    combo_font = font(18, bold=False)
    title_font = font(18, bold=False)
    tile_h, tile_w = tiles[0].shape[:2]
    width = len(tiles) * tile_w + (len(tiles) - 1) * gap
    height = combo_h + header_h + tile_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw_centered_text(draw, (0, 0, width, combo_h), mask_label, combo_font)
    for idx, (tile, title) in enumerate(zip(tiles, titles)):
        if tile.shape[:2] != (tile_h, tile_w):
            raise ValueError("Panel tile {} shape {} differs from first tile {}".format(
                title, tile.shape[:2], (tile_h, tile_w)))
        x0 = idx * (tile_w + gap)
        draw_centered_text(draw, (x0, combo_h, x0 + tile_w, combo_h + header_h), title, title_font)
        canvas.paste(Image.fromarray(tile), (x0, combo_h + header_h))
    return canvas


def save_svg_with_embedded_png(image, path):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    width, height = image.size
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        'viewBox="0 0 {w} {h}">\n'
        '<image width="{w}" height="{h}" href="data:image/png;base64,{data}"/>\n'
        '</svg>\n'
    ).format(w=width, h=height, data=encoded)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(svg)


def main():
    args = parse_args()
    axis = parse_axis(args.axis)
    methods = select_methods(args.methods)
    overlay_key = args.overlay_modality
    overlay_key = next(name for name in MODALITIES if name.lower() == overlay_key.lower())
    output_dir = os.path.join(args.out_root, args.session_name)
    os.makedirs(output_dir, exist_ok=True)

    volume, seg, seg_raw = load_case(args.data_root, args.case_name)
    predictions, pred_checks = load_predictions(args.pred_root, args.mask_name, args.case_name, seg.shape, methods)

    print("vol shape:", volume.shape, "dtype:", volume.dtype)
    print("seg shape:", seg.shape, "raw dtype:", seg_raw.dtype,
          "raw unique:", [int(x) for x in np.unique(seg_raw)],
          "mapped unique:", [int(x) for x in np.unique(seg)])
    for check in pred_checks:
        print("{method} pred shape={shape}, raw_shape={raw_shape}, dtype={dtype}, unique={unique_labels}, path={path}".format(**check))
    print("axis:", axis, "slice_index:", args.slice_index, "mask_name:", args.mask_name)

    base_u8 = normalize_to_u8(take_slice(volume[..., MODALITIES[overlay_key]], axis, args.slice_index))
    tiles = []
    titles = []
    for title, channel_idx in MODALITIES.items():
        tiles.append(grayscale_tile(volume, channel_idx, axis, args.slice_index, args.rot90))
        titles.append(title)

    seg_slice = take_slice(seg, axis, args.slice_index)
    tiles.append(overlay_tile(base_u8, seg_slice, args.alpha, args.rot90))
    titles.append("GT")

    for display_name, pred in predictions.items():
        pred_slice = take_slice(pred, axis, args.slice_index)
        tiles.append(overlay_tile(base_u8, pred_slice, args.alpha, args.rot90))
        titles.append(display_name)

    mask_label = MASK_LABELS.get(args.mask_name, args.mask_name)
    panel = make_panel(tiles, titles, mask_label)
    stem = "BraTS2023_{}_{}_slice{:03d}".format(args.case_name, args.mask_name, args.slice_index)
    png_path = os.path.join(output_dir, "{}.png".format(stem))
    pdf_path = os.path.join(output_dir, "{}.pdf".format(stem))
    svg_path = os.path.join(output_dir, "{}.svg".format(stem))
    config_path = os.path.join(output_dir, "{}_config.json".format(stem))

    panel.save(png_path)
    if args.save_pdf:
        panel.save(pdf_path, "PDF", resolution=300.0)
    if args.save_svg:
        save_svg_with_embedded_png(panel, svg_path)

    config = {
        "data_root": args.data_root,
        "case_name": args.case_name,
        "mask_name": args.mask_name,
        "axis": axis,
        "axis_arg": args.axis,
        "slice_index": args.slice_index,
        "session_name": args.session_name,
        "pred_root": args.pred_root,
        "out_root": args.out_root,
        "overlay_modality": overlay_key,
        "alpha": args.alpha,
        "rot90": args.rot90,
        "label_colors": {str(k): v.astype(int).tolist() for k, v in LABEL_COLORS.items()},
        "modalities": list(MODALITIES.keys()),
        "methods": list(methods.keys()),
        "mask_label": mask_label,
        "outputs": {
            "png": png_path,
            "pdf": pdf_path if args.save_pdf else None,
            "svg": svg_path if args.save_svg else None,
        },
    }
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print("Saved PNG:", png_path)
    if args.save_pdf:
        print("Saved PDF:", pdf_path)
    if args.save_svg:
        print("Saved SVG:", svg_path)
    print("Saved config:", config_path)
    print("All shape checks passed; no resize/crop/padding was applied.")


if __name__ == "__main__":
    main()
