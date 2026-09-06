"""
Evaluation metrics for BraTS brain tumor segmentation.

Per-class metrics:  Dice, IoU, Accuracy, Precision, Sensitivity, HD95
BraTS composite:    WT (Whole Tumor), TC (Tumor Core), ET (Enhancing Tumor)

BraTS region definitions:
  WT = labels 1 + 2 + 3  (everything that is tumor)
  TC = labels 1 + 3       (necrotic core + enhancing tumor)
  ET = label  3            (enhancing tumor only)
"""

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F


def _hd95_binary(pred_3d: np.ndarray, gt_3d: np.ndarray,
                 voxel_spacing=(1.0, 1.0, 1.0)) -> float:
    """
    95th-percentile Hausdorff distance between two binary 3D masks.
    Returns nan if either surface is empty.
    """
    from scipy.spatial import cKDTree
    struct    = ndimage.generate_binary_structure(3, 1)
    pred_surf = pred_3d.astype(bool) ^ ndimage.binary_erosion(pred_3d.astype(bool), struct)
    gt_surf   = gt_3d.astype(bool)   ^ ndimage.binary_erosion(gt_3d.astype(bool),   struct)
    if not pred_surf.any() or not gt_surf.any():
        return float('nan')
    sz, sy, sx  = voxel_spacing
    pred_pts    = np.argwhere(pred_surf) * np.array([sz, sy, sx])
    gt_pts      = np.argwhere(gt_surf)   * np.array([sz, sy, sx])
    tree_gt     = cKDTree(gt_pts)
    tree_pred   = cKDTree(pred_pts)
    d1, _       = tree_gt.query(pred_pts, workers=-1)
    d2, _       = tree_pred.query(gt_pts,  workers=-1)
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def compute_metrics(
    logits,
    target,
    num_classes: int = 4,
    smooth: float = 1e-6,
    compute_hd95: bool = False,
    voxel_spacing=(1.0, 1.0, 1.0)
) -> dict:
    """
    Compute all metrics for a batch.

    Args:
        logits:       model output (B, C, D, H, W) or tuple (training with aux heads)
        target:       ground truth labels (B, D, H, W), dtype long
        num_classes:  number of segmentation classes
        smooth:       Laplace smoothing to avoid division by zero
        compute_hd95: whether to compute HD95 (slow — use for final test eval)

    Returns:
        dict with keys:
          'mean'      → mean foreground metrics
          'per_class' → per-class metrics dict
          'brats'     → BraTS WT/TC/ET metrics dict
    """
    if isinstance(logits, tuple):
        logits = logits[0]

    pred   = logits.argmax(dim=1)
    p_flat = pred.reshape(-1)
    t_flat = target.reshape(-1)

    # ── Per-class metrics ─────────────────────────────────────────────────────
    per_class = {}
    for c in range(num_classes):
        p_c = (p_flat == c).float()
        t_c = (t_flat == c).float()
        TP  = (p_c * t_c).sum().item()
        FP  = (p_c * (1 - t_c)).sum().item()
        FN  = ((1 - p_c) * t_c).sum().item()
        TN  = ((1 - p_c) * (1 - t_c)).sum().item()
        per_class[c] = {
            'dice':        (2*TP + smooth) / (2*TP + FP + FN + smooth),
            'iou':         (TP + smooth)   / (TP + FP + FN + smooth),
            'accuracy':    (TP + TN + smooth) / (TP + TN + FP + FN + smooth),
            'precision':   (TP + smooth)   / (TP + FP + smooth),
            'sensitivity': (TP + smooth)   / (TP + FN + smooth),
            'hd95':        float('nan'),
        }

    if compute_hd95:
        pred_np   = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        for c in range(num_classes):
            vals = []
            for b in range(pred_np.shape[0]):
                h = _hd95_binary(pred_np[b] == c, target_np[b] == c, voxel_spacing)
                if not np.isnan(h):
                    vals.append(h)
            per_class[c]['hd95'] = float(np.mean(vals)) if vals else float('nan')

    # ── Mean foreground metrics ───────────────────────────────────────────────
    fg = list(range(1, num_classes))
    mean_m = {k: sum(per_class[c][k] for c in fg) / len(fg)
              for k in per_class[0] if k != 'hd95'}
    hd95_fg = [per_class[c]['hd95'] for c in fg if not np.isnan(per_class[c]['hd95'])]
    mean_m['hd95'] = float(np.mean(hd95_fg)) if hd95_fg else float('nan')

    # ── BraTS composite region metrics ───────────────────────────────────────
    def _binary_stats(p_mask, t_mask, p_3d=None, t_3d=None):
        p_mask = p_mask.float()
        t_mask = t_mask.float()
        TP = (p_mask * t_mask).sum().item()
        FP = (p_mask * (1 - t_mask)).sum().item()
        FN = ((1 - p_mask) * t_mask).sum().item()
        stats = {
            'dice':        (2*TP + smooth) / (2*TP + FP + FN + smooth),
            'iou':         (TP + smooth)   / (TP + FP + FN + smooth),
            'sensitivity': (TP + smooth)   / (TP + FN + smooth),
            'precision':   (TP + smooth)   / (TP + FP + smooth),
            'gt_present':  t_mask.sum().item() > 0,
            'hd95':        float('nan'),
        }
        if compute_hd95 and p_3d is not None:
            vals = []
            for b in range(p_3d.shape[0]):
                h = _hd95_binary(p_3d[b], t_3d[b], voxel_spacing)
                if not np.isnan(h):
                    vals.append(h)
            stats['hd95'] = float(np.mean(vals)) if vals else float('nan')
        return stats

    pred_np   = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    brats = {
        'WT': _binary_stats(
            p_flat > 0, t_flat > 0,
            pred_np > 0, target_np > 0
        ),
        'TC': _binary_stats(
            (p_flat == 1) | (p_flat == 3),
            (t_flat == 1) | (t_flat == 3),
            (pred_np == 1) | (pred_np == 3),
            (target_np == 1) | (target_np == 3),
        ),
        'ET': _binary_stats(
            p_flat == 3, t_flat == 3,
            pred_np == 3, target_np == 3,
        ),
    }

    return {'mean': mean_m, 'per_class': per_class, 'brats': brats}
