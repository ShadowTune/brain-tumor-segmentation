"""
Comprehensive visualization plots for BilRAUNet training analysis.
Reproduces all 14 plots (A→N) from the original notebook.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

LABEL_COLORS = {'BG': '#AAAAAA', 'NCR': '#FF4444', 'Edema': '#44BB44', 'ET': '#4444FF'}
REGION_COLORS = {'WT': '#FF6B6B', 'TC': '#4ECDC4', 'ET': '#45B7D1'}
FIG_DPI = 150


def _load_history(history_path: str) -> dict:
    if isinstance(history_path, dict):
        return history_path
    with open(history_path) as f:
        return json.load(f)


def _safe_get(history, split, key, sub=None):
    """Safely extract a metric series from history dict."""
    metrics = history.get(f'{split}_metrics', [])
    series  = []
    for m in metrics:
        try:
            val = m[key][sub] if sub else m[key]
            if isinstance(val, dict):
                val = val.get('dice', float('nan'))
            series.append(val if val == val else None)
        except (KeyError, TypeError):
            series.append(None)
    epochs = list(range(1, len(series) + 1))
    return epochs, series


def _clean_series(epochs, series):
    clean_e, clean_s = [], []
    for e, s in zip(epochs, series):
        if s is not None and s == s:
            clean_e.append(e)
            clean_s.append(s)
    return clean_e, clean_s


def plot_A_training_loss(history, save_dir):
    """Plot A: Training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ep = list(range(1, len(history['train_loss']) + 1))
    ax.plot(ep, history['train_loss'], 'b-o', ms=3, label='Train Loss')
    ax.plot(ep, history['val_loss'],   'r-o', ms=3, label='Val Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('A — Training & Validation Loss'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'A_loss.png'), dpi=FIG_DPI)
    plt.close(fig)
    print('Saved: A_loss.png')


def plot_B_brats_dice(history, save_dir):
    """Plot B: BraTS WT/TC/ET Dice per epoch (train + val)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, split in zip(axes, ['train', 'val']):
        for region, color in REGION_COLORS.items():
            ep, s = _safe_get(history, split, 'brats', region)
            ep_c, s_c = _clean_series(ep, s)
            if not ep_c: continue
            dice_vals = [v.get('dice', 0) if isinstance(v, dict) else v for v in s_c]
            ax.plot(ep_c, dice_vals, color=color, marker='o', ms=2, label=f'{region} Dice')
        ax.set_title(f'B — BraTS Dice ({split})')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Dice'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'B_brats_dice.png'), dpi=FIG_DPI)
    plt.close(fig)
    print('Saved: B_brats_dice.png')


def plot_C_per_class_dice(history, save_dir):
    """Plot C: Per-class Dice curves (val only)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for cls_id, (name, color) in enumerate(LABEL_COLORS.items()):
        ep, s = _safe_get(history, 'val', 'per_class', cls_id)
        ep_c, s_c = _clean_series(ep, s)
        if not ep_c: continue
        dice_vals = [v.get('dice', 0) if isinstance(v, dict) else v for v in s_c]
        ax.plot(ep_c, dice_vals, color=color, marker='o', ms=2, label=f'{name}')
    ax.set_title('C — Per-Class Dice (Val)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Dice'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'C_per_class_dice.png'), dpi=FIG_DPI)
    plt.close(fig)
    print('Saved: C_per_class_dice.png')


def plot_D_sensitivity(history, save_dir):
    """Plot D: Sensitivity per class (val)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for cls_id, (name, color) in enumerate(LABEL_COLORS.items()):
        ep, s = _safe_get(history, 'val', 'per_class', cls_id)
        ep_c, s_c = _clean_series(ep, s)
        if not ep_c: continue
        sens_vals = [v.get('sensitivity', 0) if isinstance(v, dict) else 0 for v in s_c]
        ax.plot(ep_c, sens_vals, color=color, marker='o', ms=2, label=f'{name}')
    ax.set_title('D — Sensitivity per Class (Val)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Sensitivity')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'D_sensitivity.png'), dpi=FIG_DPI)
    plt.close(fig)
    print('Saved: D_sensitivity.png')


def plot_E_precision(history, save_dir):
    """Plot E: Precision per class (val)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for cls_id, (name, color) in enumerate(LABEL_COLORS.items()):
        ep, s = _safe_get(history, 'val', 'per_class', cls_id)
        ep_c, s_c = _clean_series(ep, s)
        if not ep_c: continue
        prec_vals = [v.get('precision', 0) if isinstance(v, dict) else 0 for v in s_c]
        ax.plot(ep_c, prec_vals, color=color, marker='o', ms=2, label=f'{name}')
    ax.set_title('E — Precision per Class (Val)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Precision')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'E_precision.png'), dpi=FIG_DPI)
    plt.close(fig)
    print('Saved: E_precision.png')


def plot_F_iou(history, save_dir):
    """Plot F: IoU per class (val)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for cls_id, (name, color) in enumerate(LABEL_COLORS.items()):
        ep, s = _safe_get(history, 'val', 'per_class', cls_id)
        ep_c, s_c = _clean_series(ep, s)
        if not ep_c: continue
        iou_vals = [v.get('iou', 0) if isinstance(v, dict) else 0 for v in s_c]
        ax.plot(ep_c, iou_vals, color=color, marker='o', ms=2, label=f'{name}')
    ax.set_title('F — IoU per Class (Val)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('IoU')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'F_iou.png'), dpi=FIG_DPI)
    plt.close(fig)
    print('Saved: F_iou.png')


def plot_G_train_val_dice(history, save_dir):
    """Plot G: Train vs Val mean foreground Dice (overfitting check)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for split, color in [('train', 'blue'), ('val', 'red')]:
        ep, s = _safe_get(history, split, 'mean')
        ep_c, s_c = _clean_series(ep, s)
        if not ep_c: continue
        dice_vals = [v.get('dice', v) if isinstance(v, dict) else v for v in s_c]
        ax.plot(ep_c, dice_vals, color=color, marker='o', ms=2, label=f'{split} mean Dice')
    ax.set_title('G — Train vs Val Mean Foreground Dice')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Mean Dice')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'G_train_val_dice.png'), dpi=FIG_DPI)
    plt.close(fig)
    print('Saved: G_train_val_dice.png')


def plot_H_brats_sensitivity(history, save_dir):
    """Plot H: BraTS WT/TC/ET sensitivity."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for region, color in REGION_COLORS.items():
        ep, s = _safe_get(history, 'val', 'brats', region)
        ep_c, s_c = _clean_series(ep, s)
        if not ep_c: continue
        sens_vals = [v.get('sensitivity', 0) if isinstance(v, dict) else 0 for v in s_c]
        ax.plot(ep_c, sens_vals, color=color, marker='o', ms=2, label=region)
    ax.set_title('H — BraTS Sensitivity (WT/TC/ET) Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Sensitivity')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'H_brats_sensitivity.png'), dpi=FIG_DPI)
    plt.close(fig)
    print('Saved: H_brats_sensitivity.png')


def plot_I_brats_precision(history, save_dir):
    """Plot I: BraTS WT/TC/ET precision."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for region, color in REGION_COLORS.items():
        ep, s = _safe_get(history, 'val', 'brats', region)
        ep_c, s_c = _clean_series(ep, s)
        if not ep_c: continue
        prec_vals = [v.get('precision', 0) if isinstance(v, dict) else 0 for v in s_c]
        ax.plot(ep_c, prec_vals, color=color, marker='o', ms=2, label=region)
    ax.set_title('I — BraTS Precision (WT/TC/ET) Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Precision')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'I_brats_precision.png'), dpi=FIG_DPI)
    plt.close(fig)
    print('Saved: I_brats_precision.png')


def plot_J_dice_bar(test_results, save_dir):
    """Plot J: Final test Dice bar chart (BraTS + per-class)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    brats = test_results.get('brats', {})
    regions = ['WT', 'TC', 'ET']
    vals    = [brats.get(r, {}).get('dice', 0) for r in regions]
    colors  = [REGION_COLORS[r] for r in regions]
    axes[0].bar(regions, vals, color=colors)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=9)
    axes[0].set_ylim(0, 1.05); axes[0].set_title('J — Test BraTS Dice')
    axes[0].set_ylabel('Dice')

    pc = test_results.get('per_class', {})
    label_names = {0: 'BG', 1: 'NCR', 2: 'Edema', 3: 'ET'}
    cls_vals   = [pc.get(c, {}).get('dice', 0) for c in range(4)]
    cls_colors = list(LABEL_COLORS.values())
    axes[1].bar(list(label_names.values()), cls_vals, color=cls_colors)
    for i, v in enumerate(cls_vals):
        axes[1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=9)
    axes[1].set_ylim(0, 1.05); axes[1].set_title('J — Test Per-Class Dice')
    axes[1].set_ylabel('Dice')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'J_test_dice_bar.png'), dpi=FIG_DPI)
    plt.close(fig)
    print('Saved: J_test_dice_bar.png')


def plot_K_all_metrics_bar(test_results, save_dir):
    """Plot K: All metrics (Dice, IoU, Sens, Prec) for BraTS regions."""
    fig, ax = plt.subplots(figsize=(12, 5))
    brats   = test_results.get('brats', {})
    regions = ['WT', 'TC', 'ET']
    metrics = ['dice', 'iou', 'sensitivity', 'precision']
    x = np.arange(len(regions)); width = 0.2
    colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12']
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [brats.get(r, {}).get(metric, 0) for r in regions]
        ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=color)
    ax.set_xticks(x + width * 1.5); ax.set_xticklabels(regions)
    ax.set_ylim(0, 1.1); ax.set_title('K — All Metrics (BraTS Regions, Test)')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'K_all_metrics_bar.png'), dpi=FIG_DPI)
    plt.close(fig)
    print('Saved: K_all_metrics_bar.png')


def plot_L_loss_smooth(history, save_dir, window=5):
    """Plot L: Smoothed loss curve (moving average)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for losses, label, color in [
        (history['train_loss'], 'Train', 'blue'),
        (history['val_loss'],   'Val',   'red'),
    ]:
        ep = list(range(1, len(losses) + 1))
        smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
        ep_s   = ep[window-1:]
        ax.plot(ep, losses, color=color, alpha=0.3)
        ax.plot(ep_s, smooth, color=color, linewidth=2, label=f'{label} (smooth)')
    ax.set_title(f'L — Loss (smoothed, window={window})')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'L_loss_smooth.png'), dpi=FIG_DPI)
    plt.close(fig)
    print('Saved: L_loss_smooth.png')


def plot_M_summary_grid(history, test_results, save_dir):
    """Plot M: 2×3 summary grid — loss + 5 key metrics."""
    fig = plt.figure(figsize=(18, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    brats = test_results.get('brats', {}) if test_results else {}

    # Loss
    ax = fig.add_subplot(gs[0, 0])
    ep = list(range(1, len(history['train_loss']) + 1))
    ax.plot(ep, history['train_loss'], 'b-', label='Train'); ax.plot(ep, history['val_loss'], 'r-', label='Val')
    ax.set_title('Loss'); ax.legend(); ax.grid(True, alpha=0.3)

    # BraTS WT Dice
    ax = fig.add_subplot(gs[0, 1])
    for split, color in [('train','blue'), ('val','red')]:
        ep_s, s = _safe_get(history, split, 'brats', 'WT')
        ep_c, s_c = _clean_series(ep_s, s)
        dice_vals = [v.get('dice',0) if isinstance(v,dict) else v for v in s_c]
        if ep_c: ax.plot(ep_c, dice_vals, color=color, label=split)
    ax.set_title('WT Dice'); ax.legend(); ax.grid(True, alpha=0.3)

    # BraTS ET Dice
    ax = fig.add_subplot(gs[0, 2])
    for split, color in [('train','blue'), ('val','red')]:
        ep_s, s = _safe_get(history, split, 'brats', 'ET')
        ep_c, s_c = _clean_series(ep_s, s)
        dice_vals = [v.get('dice',0) if isinstance(v,dict) else v for v in s_c]
        if ep_c: ax.plot(ep_c, dice_vals, color=color, label=split)
    ax.set_title('ET Dice'); ax.legend(); ax.grid(True, alpha=0.3)

    # Test bar
    ax = fig.add_subplot(gs[1, 0])
    if brats:
        r_names = ['WT', 'TC', 'ET']
        vals    = [brats.get(r, {}).get('dice', 0) for r in r_names]
        ax.bar(r_names, vals, color=[REGION_COLORS[r] for r in r_names])
        ax.set_ylim(0, 1.05); ax.set_title('Test BraTS Dice')
        for i, v in enumerate(vals): ax.text(i, v+0.01, f'{v:.3f}', ha='center')

    # Per-class dice val (last epoch)
    ax = fig.add_subplot(gs[1, 1])
    if history.get('val_metrics'):
        last_m = history['val_metrics'][-1].get('per_class', {})
        cls_names = {0:'BG', 1:'NCR', 2:'ED', 3:'ET'}
        vals = [last_m.get(c, {}).get('dice', 0) for c in range(4)]
        ax.bar(list(cls_names.values()), vals, color=list(LABEL_COLORS.values()))
        ax.set_ylim(0, 1.05); ax.set_title('Per-Class Dice (Last Val)')

    # ET sensitivity val
    ax = fig.add_subplot(gs[1, 2])
    ep_s, s = _safe_get(history, 'val', 'brats', 'ET')
    ep_c, s_c = _clean_series(ep_s, s)
    sens_vals = [v.get('sensitivity',0) if isinstance(v,dict) else 0 for v in s_c]
    if ep_c: ax.plot(ep_c, sens_vals, color=REGION_COLORS['ET'])
    ax.set_title('ET Sensitivity (Val)'); ax.grid(True, alpha=0.3)

    fig.suptitle('M — Training Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'M_summary_grid.png'), dpi=FIG_DPI)
    plt.close(fig)
    print('Saved: M_summary_grid.png')


def plot_N_slice_overlay(image_np, pred_seg, gt_seg, save_dir, case_id='sample'):
    """
    Plot N: Mid-axial slice overlay — FLAIR + predicted + ground truth.
    image_np: (4, H, W, D) original MRI — uses FLAIR (index 0).
    pred_seg, gt_seg: (H, W, D) uint8 segmentation masks.
    """
    slice_idx = image_np.shape[-1] // 2
    flair     = image_np[0, :, :, slice_idx]
    pred_sl   = pred_seg[:, :, slice_idx]
    gt_sl     = gt_seg[:, :, slice_idx]

    cmap_seg = matplotlib.colors.ListedColormap(['black','red','green','blue'])
    norm_seg  = matplotlib.colors.BoundaryNorm([0,1,2,3,4], cmap_seg.N)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(flair.T, cmap='gray', origin='lower')
    axes[0].set_title('FLAIR'); axes[0].axis('off')

    axes[1].imshow(flair.T, cmap='gray', origin='lower')
    axes[1].imshow(np.ma.masked_where(pred_sl.T == 0, pred_sl.T),
                   cmap=cmap_seg, norm=norm_seg, alpha=0.5, origin='lower')
    axes[1].set_title('Predicted'); axes[1].axis('off')

    axes[2].imshow(flair.T, cmap='gray', origin='lower')
    axes[2].imshow(np.ma.masked_where(gt_sl.T == 0, gt_sl.T),
                   cmap=cmap_seg, norm=norm_seg, alpha=0.5, origin='lower')
    axes[2].set_title('Ground Truth'); axes[2].axis('off')

    patches = [mpatches.Patch(color=c, label=n)
               for n, c in [('NCR','red'),('Edema','green'),('ET','blue')]]
    fig.legend(handles=patches, loc='lower center', ncol=3)
    fig.suptitle(f'N — Slice Overlay: {case_id}')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'N_slice_overlay_{case_id}.png'), dpi=FIG_DPI)
    plt.close(fig)
    print(f'Saved: N_slice_overlay_{case_id}.png')


def generate_all_plots(history_or_path, test_results_or_path=None, save_dir='outputs/plots'):
    """
    Generate all 14 plots in one call.

    Args:
        history_or_path:      dict or path to history.json
        test_results_or_path: dict or path to test_results.json (optional)
        save_dir:             output directory for PNG files
    """
    os.makedirs(save_dir, exist_ok=True)
    history = _load_history(history_or_path)

    test_results = None
    if test_results_or_path:
        test_results = _load_history(test_results_or_path)

    print(f'Generating all plots → {save_dir}/')
    plot_A_training_loss(history, save_dir)
    plot_B_brats_dice(history, save_dir)
    plot_C_per_class_dice(history, save_dir)
    plot_D_sensitivity(history, save_dir)
    plot_E_precision(history, save_dir)
    plot_F_iou(history, save_dir)
    plot_G_train_val_dice(history, save_dir)
    plot_H_brats_sensitivity(history, save_dir)
    plot_I_brats_precision(history, save_dir)
    plot_L_loss_smooth(history, save_dir)

    if test_results:
        plot_J_dice_bar(test_results, save_dir)
        plot_K_all_metrics_bar(test_results, save_dir)
        plot_M_summary_grid(history, test_results, save_dir)

    print(f'\nAll plots saved to {save_dir}/')
