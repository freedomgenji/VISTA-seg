import argparse
import os
import random
from glob import glob

import nibabel as nib
import numpy as np


MODALITIES = ("flair", "t1ce", "t1", "t2")
MODALITY_SUFFIX_ALIASES = {
    "flair": ("flair", "t2f"),
    "t1ce": ("t1ce", "t1c"),
    "t1": ("t1", "t1n"),
    "t2": ("t2", "t2w"),
    "seg": ("seg",),
}
NIFTI_EXTENSIONS = (".nii.gz", ".nii")


def ensure_min_size(lower, upper, limit, min_size):
    if upper - lower >= min_size:
        return lower, upper

    pad = min_size - (upper - lower)
    lower -= pad // 2
    upper += pad - pad // 2

    if lower < 0:
        upper -= lower
        lower = 0
    if upper > limit:
        shift = upper - limit
        lower = max(0, lower - shift)
        upper = limit
    return lower, upper


def crop_bounds(volume, min_size):
    if volume.ndim == 4:
        nonzero_source = np.amax(volume, axis=0)
    else:
        nonzero_source = volume

    nonzero = np.where(nonzero_source != 0)
    if len(nonzero[0]) == 0:
        raise ValueError("the case is entirely zero after loading")

    x_min, x_max = int(np.amin(nonzero[0])), int(np.amax(nonzero[0])) + 1
    y_min, y_max = int(np.amin(nonzero[1])), int(np.amax(nonzero[1])) + 1
    z_min, z_max = int(np.amin(nonzero[2])), int(np.amax(nonzero[2])) + 1

    x_min, x_max = ensure_min_size(x_min, x_max, volume.shape[-3], min_size)
    y_min, y_max = ensure_min_size(y_min, y_max, volume.shape[-2], min_size)
    z_min, z_max = ensure_min_size(z_min, z_max, volume.shape[-1], min_size)

    return x_min, x_max, y_min, y_max, z_min, z_max


def normalize(volume):
    mask = volume.sum(0) > 0
    for modality_idx in range(volume.shape[0]):
        data = volume[modality_idx]
        foreground = data[mask]
        if foreground.size == 0:
            continue
        std = foreground.std()
        if std < 1e-8:
            std = 1.0
        volume[modality_idx] = (data - foreground.mean()) / std
    return volume


def load_nii(path, dtype):
    return np.asarray(nib.load(path).dataobj, dtype=dtype)


def resolve_nifti_file(case_dir, case_name, suffix):
    suffixes = MODALITY_SUFFIX_ALIASES.get(suffix, (suffix,))
    separators = ("_", "-")
    for alias in suffixes:
        for separator in separators:
            for extension in NIFTI_EXTENSIONS:
                candidate = os.path.join(case_dir, f"{case_name}{separator}{alias}{extension}")
                if os.path.exists(candidate):
                    return candidate

    matches = []
    for alias in suffixes:
        for separator in separators:
            for extension in NIFTI_EXTENSIONS:
                matches.extend(glob(os.path.join(case_dir, f"*{separator}{alias}{extension}")))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Found multiple candidates for {suffix} in {case_dir}: {matches}")
    return None


def find_case_dirs(src_root):
    case_dirs = []
    for current_root, _, _ in os.walk(src_root):
        case_name = os.path.basename(current_root)
        if not case_name:
            continue

        required = [resolve_nifti_file(current_root, case_name, modality) for modality in MODALITIES]
        required.append(resolve_nifti_file(current_root, case_name, "seg"))
        if all(path is not None for path in required):
            case_dirs.append((case_name, current_root))
    return sorted(case_dirs)


def save_split_files(case_names, dst_root, train_ratio, seed):
    shuffled = list(case_names)
    random.Random(seed).shuffle(shuffled)

    if len(shuffled) <= 1:
        train_names = shuffled
        test_names = shuffled
    else:
        split_index = int(len(shuffled) * train_ratio)
        split_index = min(max(split_index, 1), len(shuffled) - 1)
        train_names = shuffled[:split_index]
        test_names = shuffled[split_index:]

    split_map = {
        "train.txt": train_names,
        "test.txt": test_names,
        "val.txt": test_names,
    }
    for filename, names in split_map.items():
        with open(os.path.join(dst_root, filename), "w", encoding="utf-8") as handle:
            handle.write("\n".join(names))
            handle.write("\n")

    print(f"Prepared {len(train_names)} train cases and {len(test_names)} eval cases.")


def preprocess_case(case_name, case_dir, dst_root, min_size):
    modality_volumes = []
    for modality in MODALITIES:
        modality_path = resolve_nifti_file(case_dir, case_name, modality)
        modality_volumes.append(load_nii(modality_path, np.float32))

    volume = np.stack(modality_volumes, axis=0).astype(np.float32)
    bounds = crop_bounds(volume, min_size=min_size)
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    volume = normalize(volume[:, x_min:x_max, y_min:y_max, z_min:z_max])
    volume = volume.transpose(1, 2, 3, 0)

    seg_path = resolve_nifti_file(case_dir, case_name, "seg")
    seg = load_nii(seg_path, np.uint8)
    seg = seg[x_min:x_max, y_min:y_max, z_min:z_max]
    seg[seg == 4] = 3

    np.save(os.path.join(dst_root, "vol", f"{case_name}_vol.npy"), volume)
    np.save(os.path.join(dst_root, "seg", f"{case_name}_seg.npy"), seg)
    print(f"Processed {case_name}: {volume.shape}")


def main():
    parser = argparse.ArgumentParser(description="Convert raw BraTS NIfTI data into the project NPY layout.")
    parser.add_argument("--src_root", required=True, help="Raw BraTS training root, e.g. ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
    parser.add_argument("--dst_root", required=True, help="Output directory for vol/ seg/ and split txt files")
    parser.add_argument("--train_ratio", default=0.8, type=float, help="Fraction of cases written into train.txt")
    parser.add_argument("--seed", default=42, type=int, help="Random seed used for the train/test split")
    parser.add_argument("--min_size", default=128, type=int, help="Minimum cropped size on each axis")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.dst_root, "vol"), exist_ok=True)
    os.makedirs(os.path.join(args.dst_root, "seg"), exist_ok=True)

    case_dirs = find_case_dirs(args.src_root)
    if not case_dirs:
        raise FileNotFoundError(
            f"No BraTS cases were found under {args.src_root}. "
            "Expected folders with BraTS modality files such as *_flair/_t1ce/_t1/_t2 or *-t2f/-t1c/-t1n/-t2w and *seg .nii.gz files."
        )

    case_names = []
    for case_name, case_dir in case_dirs:
        preprocess_case(case_name, case_dir, args.dst_root, min_size=args.min_size)
        case_names.append(case_name)

    save_split_files(case_names, args.dst_root, train_ratio=args.train_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
