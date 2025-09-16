
import os
import csv
import glob
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import pydicom
except Exception as e:
    pydicom = None
    # We keep import optional so the file can be imported even if pydicom not installed.
    # The dataset will raise a clear error at runtime if missing.


def _load_dicom_series(series_dir: str) -> np.ndarray:
    '''
    Load a DICOM series (a folder of DICOM files) into a 3D numpy volume [Z, H, W],
    sorted by InstanceNumber (or by filename as fallback). Applies rescale slope/intercept.
    '''
    if pydicom is None:
        raise ImportError('pydicom is required. Please `pip install pydicom`.')
    files = sorted(glob.glob(os.path.join(series_dir, '*')))
    if not files:
        raise FileNotFoundError(f'No files found under {series_dir}')

    # Read all slices
    slices = []
    meta = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, force=True)
            if hasattr(ds, 'InstanceNumber'):
                meta.append((int(ds.InstanceNumber), f, ds))
            else:
                meta.append((len(meta), f, ds))
        except Exception:
            # skip non-dicom or corrupt
            continue

    if not meta:
        raise RuntimeError(f'No readable DICOM slices in {series_dir}')

    # sort by InstanceNumber fallback to order
    meta.sort(key=lambda x: x[0])

    for _, _, ds in meta:
        arr = ds.pixel_array.astype(np.float32)
        # Apply rescale (if present)
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        arr = arr * slope + intercept
        slices.append(arr)

    vol = np.stack(slices, axis=0)  # [Z, H, W]
    return vol


def _window_level(vol: np.ndarray, center: Optional[float] = None, width: Optional[float] = None) -> np.ndarray:
    '''
    Apply window/level. If center/width are None, uses robust min/max (2/98 percentiles).
    '''
    v = vol.copy()
    if center is None or width is None:
        lo, hi = np.percentile(v, [2.0, 98.0])
        center = (lo + hi) / 2.0
        width = (hi - lo)
        if width <= 0:
            width = max(1.0, hi - lo)
    low = center - width / 2.0
    high = center + width / 2.0
    v = np.clip(v, low, high)
    v = (v - low) / max(1e-6, (high - low))  # [0,1]
    return v


def _resize_slice(img: np.ndarray, size: int = 224) -> np.ndarray:
    '''
    Resize a 2D slice to square [size, size] using OpenCV if available, else numpy fallback.
    '''
    try:
        import cv2
        out = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    except Exception:
        # naive nearest fallback
        zoom_h = size / img.shape[0]
        zoom_w = size / img.shape[1]
        out = np.zeros((size, size), dtype=img.dtype)
        ys = (np.arange(size) / zoom_h).astype(int).clip(0, img.shape[0]-1)
        xs = (np.arange(size) / zoom_w).astype(int).clip(0, img.shape[1]-1)
        out = img[ys[:, None], xs[None, :]]
    return out


def _select_channels_from_volume(vol: np.ndarray, strategy: str = 'middle3', max_channels: int = 6) -> np.ndarray:
    '''
    Convert [Z, H, W] volume into [C, H, W] channels by selecting slices.
    strategy:
      - 'middle3': take 3 middle slices (pad if less than 3)
      - 'firstlast': take first, middle, last (C=3)
      - 'uniform': sample up to max_channels uniformly across Z
      - 'all_clip': take all slices and clip/truncate to max_channels from center
    '''
    Z = vol.shape[0]
    if Z == 0:
        raise ValueError('Empty volume')

    def take_indices(idxs: List[int]) -> np.ndarray:
        idxs = [int(np.clip(i, 0, Z-1)) for i in idxs]
        return vol[idxs]

    if strategy == 'middle3':
        if Z >= 3:
            mid = Z // 2
            idxs = [max(0, mid-1), mid, min(Z-1, mid+1)]
        else:
            idxs = list(range(Z))
            while len(idxs) < 3:
                idxs.append(idxs[-1])
        out = take_indices(idxs)
    elif strategy == 'firstlast':
        mid = Z // 2
        idxs = [0, mid, Z-1]
        out = take_indices(idxs)
    elif strategy == 'uniform':
        C = min(max_channels, Z)
        idxs = np.linspace(0, Z-1, C).round().astype(int).tolist()
        out = take_indices(idxs)
    elif strategy == 'all_clip':
        if Z <= max_channels:
            out = vol
            while out.shape[0] < max_channels:
                out = np.concatenate([out, out[-1:]], axis=0)
        else:
            mid = Z // 2
            half = max_channels // 2
            start = max(0, mid - half)
            end = start + max_channels
            out = vol[start:end]
    else:
        raise ValueError(f'Unknown strategy: {strategy}')

    return out  # [C, H, W]


class DicomPairDataset(Dataset):
    '''
    Dataset for before/after DICOM series pairs.
    Expect a CSV manifest with columns: before_dir, after_dir, label
      - before_dir: path to the BEFORE series folder
      - after_dir:  path to the AFTER  series folder
      - label: 0 or 1
    Options:
      - size: resize H,W to this square (e.g., 224)
      - slice_strategy: how to select channels from the 3D volume (see _select_channels_from_volume)
      - max_channels: cap channels (C) after selection
      - normalize: per-sample z-score normalization
      - window: whether to apply window/level to rescale intensities to [0,1]
    The output per sample:
      before: FloatTensor[C, size, size]
      after:  FloatTensor[C, size, size]
      label:  int (0 or 1)
    '''
    def __init__(
        self,
        manifest_csv: str,
        size: int = 224,
        slice_strategy: str = 'uniform',
        max_channels: int = 6,
        normalize: bool = True,
        window: bool = True,
    ):
        super().__init__()
        self.size = size
        self.slice_strategy = slice_strategy
        self.max_channels = max_channels
        self.normalize = normalize
        self.window = window

        self.rows = []
        with open(manifest_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                bd = row.get('before_dir')
                ad = row.get('after_dir')
                lb = row.get('label')
                if bd is None or ad is None or lb is None:
                    raise ValueError('CSV must contain columns: before_dir, after_dir, label')
                self.rows.append((bd, ad, int(lb)))

        if len(self.rows) == 0:
            raise ValueError('Empty manifest')

    def __len__(self):
        return len(self.rows)

    def _prep_series(self, series_dir: str) -> np.ndarray:
        vol = _load_dicom_series(series_dir)  # [Z,H,W] float32
        if self.window:
            vol = _window_level(vol)  # to [0,1]
        vol = np.stack([_resize_slice(z, size=self.size) for z in vol], axis=0)  # [Z,S,S]
        ch = _select_channels_from_volume(vol, strategy=self.slice_strategy, max_channels=self.max_channels)  # [C,S,S]
        return ch

    def __getitem__(self, idx: int):
        bdir, adir, lab = self.rows[idx]
        b = self._prep_series(bdir)
        a = self._prep_series(adir)

        if self.normalize:
            def zscore(x):
                x = x.astype(np.float32)
                mean = x.mean(axis=(1,2), keepdims=True)
                std  = x.std(axis=(1,2), keepdims=True) + 1e-6
                return (x - mean) / std
            b = zscore(b)
            a = zscore(a)

        before = torch.from_numpy(b).float()
        after  = torch.from_numpy(a).float()
        label  = int(lab)
        return before, after, label
