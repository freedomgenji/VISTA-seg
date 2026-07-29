import os
import torch
from torch.utils.data import Dataset

from .rand import Uniform
from .transforms import Rot90, Flip, Identity, Compose
from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
from .transforms import NumpyType
from .data_utils import pkload

import numpy as np
import nibabel as nib

HGG = []
LGG = []
for i in range(0, 260):
    HGG.append(str(i).zfill(3))
for i in range(336, 370):
    HGG.append(str(i).zfill(3))
for i in range(260, 336):
    LGG.append(str(i).zfill(3))

mask_array = np.array([[True, False, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True],
                      [True, True, False, False], [True, False, True, False], [True, False, False, True], [False, True, True, False], [False, True, False, True], [False, False, True, True], [True, True, True, False], [True, True, False, True], [True, False, True, True], [False, True, True, True],
                      [True, True, True, True]])

def normalize_case_name(case_name):
    name = str(case_name).strip().replace('\\', '/').split('/')[-1]
    if name.endswith('_vol.npy'):
        name = name[:-8]
    elif name.endswith('_seg.npy'):
        name = name[:-8]
    elif name.endswith('.npy'):
        name = name[:-4]
    return name

def load_case_list(root, split_file=None, case_name=None):
    if case_name is not None:
        if isinstance(case_name, (list, tuple)):
            datalist = [normalize_case_name(item) for item in case_name]
        else:
            datalist = [normalize_case_name(item) for item in str(case_name).split(',')]
        return sorted([item for item in datalist if item])

    data_file_path = os.path.join(root, split_file)
    with open(data_file_path, 'r') as f:
        datalist = [i.strip() for i in f.readlines()]
    datalist.sort()
    return datalist

def normalize_modalities(x):
    """Foreground z-score for volumes stored as [H, W, D, C]."""
    x = x.astype(np.float32, copy=True)
    foreground_mask = np.any(x != 0, axis=-1)
    if not np.any(foreground_mask):
        return x
    for modality_idx in range(x.shape[-1]):
        modality = x[..., modality_idx]
        foreground = modality[foreground_mask]
        std = foreground.std()
        if std < 1e-8:
            std = 1.0
        modality[foreground_mask] = (foreground - foreground.mean()) / std
        modality[~foreground_mask] = 0.0
        x[..., modality_idx] = modality
    return x

class Brats_loadall_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, train_file='train.txt',
                 tumor_aware_crop=False, crop_size=None, tumor_crop_prob=0.0,
                 tumor_crop_et_weight=4.0, tumor_crop_tc_weight=3.0, tumor_crop_wt_weight=2.0,
                 normalize_on_load=False):
        data_file_path = os.path.join(root, train_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls
        self.tumor_aware_crop = tumor_aware_crop
        self.crop_size = tuple(crop_size) if crop_size is not None else None
        self.tumor_crop_prob = tumor_crop_prob
        self.tumor_crop_weights = {
            'et': float(tumor_crop_et_weight),
            'tc': float(tumor_crop_tc_weight),
            'wt': float(tumor_crop_wt_weight),
        }
        self.normalize_on_load = normalize_on_load
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def _random_crop_start(self, shape, size):
        start = []
        for dim, crop in zip(shape, size):
            if dim <= crop:
                start.append(0)
            else:
                start.append(np.random.randint(0, dim - crop + 1))
        return start

    def _centered_crop_start(self, center, shape, size):
        start = []
        for coord, dim, crop in zip(center, shape, size):
            if dim <= crop:
                start.append(0)
                continue
            offset = np.random.randint(0, crop)
            start.append(int(np.clip(coord - offset, 0, dim - crop)))
        return start

    def _sample_tumor_center(self, y):
        et_label = 3 if self.num_cls <= 4 else self.num_cls - 1
        candidates = {
            'et': np.argwhere(y == et_label),
            'tc': np.argwhere((y == 1) | (y == et_label)),
            'wt': np.argwhere(y > 0),
        }

        names = []
        weights = []
        for name, coords in candidates.items():
            weight = self.tumor_crop_weights.get(name, 0.0)
            if weight > 0.0 and len(coords) > 0:
                names.append(name)
                weights.append(weight)
        if not names:
            return None

        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
        name = np.random.choice(names, p=weights)
        coords = candidates[name]
        return coords[np.random.randint(0, len(coords))]

    def _crop_pair(self, x, y, start, size):
        slices = tuple(slice(s, s + c) for s, c in zip(start, size))
        return x[slices + (slice(None),)], y[slices]

    def _tumor_aware_crop(self, x, y):
        if not self.tumor_aware_crop or self.crop_size is None:
            return x, y

        shape = y.shape
        size = tuple(min(crop, dim) for crop, dim in zip(self.crop_size, shape))
        use_tumor = np.random.rand() < self.tumor_crop_prob
        center = self._sample_tumor_center(y) if use_tumor else None
        if center is None:
            start = self._random_crop_start(shape, size)
        else:
            start = self._centered_crop_start(center, shape, size)
        return self._crop_pair(x, y, start, size)

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        
        x = np.load(volpath)
        if self.normalize_on_load:
            x = normalize_modalities(x)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)
        x, y = self._tumor_aware_crop(x, y)
        x, y = x[None, ...], y[None, ...]

        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        _, H, W, Z = np.shape(y)
        y = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))

        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)

        mask_idx = np.random.choice(15, 1)
        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)
        return x, yo, mask, name

    def __len__(self):
        return len(self.volpaths)

class Brats_loadall_test_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', test_file='test.txt',
                 normalize_on_load=False, case_name=None):
        datalist = load_case_list(root, split_file=test_file, case_name=case_name)
        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.normalize_on_load = normalize_on_load
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        if self.normalize_on_load:
            x = normalize_modalities(x)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)

        x = x[:, self.modal_ind, :, :, :]
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        return x, y, name

    def __len__(self):
        return len(self.volpaths)

class Brats_loadall_val_nii(Dataset):
    def __init__(self, transforms='', root=None, settype='train', modal='all',
                 normalize_on_load=False):
        data_file_path = os.path.join(root, 'val.txt')
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()
        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.normalize_on_load = normalize_on_load
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        if self.normalize_on_load:
            x = normalize_modalities(x)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)
        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        mask = mask_array[index%15]
        mask = torch.squeeze(torch.from_numpy(mask), dim=0)
        return x, y, mask, name

    def __len__(self):
        return len(self.volpaths)
