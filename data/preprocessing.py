"""
Volume preprocessing and augmentation for BraTS 3D MRI data.
All functions operate on numpy arrays in (C, H, W, D) convention.
"""

import random
import numpy as np
import scipy.ndimage as ndimage
import nibabel as nib


def normalize_volume(vol, mask=None, clip_percentiles=(0.5, 99.5), eps=1e-8):
    """
    Z-score normalize a 3D MRI volume within the brain mask.
    Steps:
      1. Clip to [0.5th, 99.5th] percentile — removes scanner noise outliers
      2. Subtract mean, divide by std — computed inside brain mask only
      3. Zero out background (outside mask)

    Args:
        vol:              3D numpy array (H, W, D)
        mask:             boolean brain mask — if None uses vol > 0
        clip_percentiles: (low, high) percentile for clipping
        eps:              small value to prevent division by zero

    Returns:
        normalized float32 array of same shape
    """
    vol = vol.astype(np.float32)
    if mask is None:
        mask = vol > 0
    if np.sum(mask) == 0:
        return vol
    masked_vals = vol[mask]
    low, high = np.percentile(masked_vals, clip_percentiles)
    vol = np.clip(vol, low, high)
    masked_vals = vol[mask]
    mean = masked_vals.mean()
    std  = masked_vals.std()
    vol  = (vol - mean) / (std + eps)
    vol[~mask] = 0.0
    return vol.astype(np.float32)


def load_volume(case_dict: dict):
    """
    Load a BraTS case from NIfTI files.
    Normalizes each modality independently, remaps label 4→3 (BraTS2020 ET label).

    Args:
        case_dict: dict with keys 'flair', 't1', 't1ce', 't2', 'seg'

    Returns:
        image: (4, H, W, D) float32 array  — [FLAIR, T1, T1ce, T2]
        seg:   (H, W, D)    uint8  array   — labels 0,1,2,3
    """
    try:
        flair = normalize_volume(np.asarray(nib.load(case_dict['flair']).dataobj))
        t1    = normalize_volume(np.asarray(nib.load(case_dict['t1']).dataobj))
        t1ce  = normalize_volume(np.asarray(nib.load(case_dict['t1ce']).dataobj))
        t2    = normalize_volume(np.asarray(nib.load(case_dict['t2']).dataobj))
        seg   = np.asarray(nib.load(case_dict['seg']).dataobj).astype(np.uint8)
    except Exception as e:
        raise RuntimeError(f'Failed to load NIfTI: {e}') from e

    ref = flair.shape
    for name, arr in [('t1', t1), ('t1ce', t1ce), ('t2', t2), ('seg', seg)]:
        if arr.shape != ref:
            raise RuntimeError(f'Shape mismatch: flair={ref}, {name}={arr.shape}')

    # BraTS2020 uses label 4 for ET; BraTS2023 uses label 3. Unify to 3.
    seg[seg == 4] = 3
    return np.stack([flair, t1, t1ce, t2], axis=0), seg


def _fit_last3_dims(vol: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Crop or pad the last 3 dims to exactly target_shape."""
    lead_ndim = vol.ndim - 3
    cur_shape  = vol.shape[-3:]
    slices     = [slice(None)] * lead_ndim
    slices    += [slice(0, min(c, t)) for c, t in zip(cur_shape, target_shape)]
    out        = vol[tuple(slices)]
    pad_widths  = [(0, 0)] * lead_ndim
    pad_widths += [(0, max(t - c, 0)) for c, t in zip(out.shape[-3:], target_shape)]
    if any(p[1] > 0 for p in pad_widths):
        out = np.pad(out, pad_widths, mode='constant')
    return out


def resize_3d(image: np.ndarray, seg: np.ndarray, roi_size: tuple):
    """
    Resize the whole volume to a fixed ROI using zoom.
    Trilinear interpolation for image, nearest-neighbor for segmentation.

    Args:
        image:    (C, H, W, D) float32
        seg:      (H, W, D)    uint8
        roi_size: (H_out, W_out, D_out)

    Returns:
        resized image (C, H_out, W_out, D_out) and seg (H_out, W_out, D_out)
    """
    C, H, W, D = image.shape
    zoom_factors = (roi_size[0] / H, roi_size[1] / W, roi_size[2] / D)
    img_r = np.stack(
        [ndimage.zoom(image[c], zoom_factors, order=1, mode='nearest') for c in range(C)],
        axis=0
    ).astype(np.float32)
    seg_r = ndimage.zoom(seg.astype(np.float32), zoom_factors, order=0, mode='nearest')
    seg_r = np.round(seg_r).astype(np.uint8)
    img_r = _fit_last3_dims(img_r, roi_size)
    seg_r = _fit_last3_dims(seg_r, roi_size)
    return img_r, seg_r


def augment_3d_intensity_only(
    image: np.ndarray,
    seg:   np.ndarray,
    scale_factor: float = 0.1,
    shift_offset: float = 0.1,
    prob: float = 1.0
):
    """
    Per-channel random intensity scale + shift (no spatial augmentation).

    Why intensity-only?  Text prompts and case reports are computed from the
    original volume geometry. Spatial augmentation (flip, rotate) would change
    anatomical landmarks described in the report (e.g. "left hemisphere"), making
    the text conditioning inconsistent with the image. Intensity jitter is safe
    because it doesn't change spatial relationships.

    Args:
        image:        (C, D, H, W) float32  [PyTorch convention, after transpose]
        seg:          (D, H, W)    long
        scale_factor: max relative intensity scale jitter
        shift_offset: max absolute intensity shift
        prob:         probability of applying each jitter per channel

    Returns:
        augmented image and unchanged seg
    """
    for c in range(image.shape[0]):
        if random.random() < prob:
            factor = 1.0 + random.uniform(-scale_factor, scale_factor)
            image[c] = image[c] * factor
        if random.random() < prob:
            offset = random.uniform(-shift_offset, shift_offset)
            image[c] = image[c] + offset
    return image, seg
