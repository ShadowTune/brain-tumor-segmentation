"""
Test set evaluation with 8-fold TTA (flip ensemble).

TTA strategy: flip all combinations of axes (D, H, W) → 8 predictions per volume.
Average softmax probabilities → argmax → final prediction.

Why TTA improves results:
  Brain tumors are roughly spatially symmetric, but the model was trained on
  the original orientation. Flipping + averaging captures the model's uncertainty
  about orientation and produces sharper, more calibrated predictions on the boundary.
"""

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.metrics import compute_metrics


def tta_predict(model, images, text_emb):
    """
    8-fold test-time augmentation using all axis flip combinations.

    Spatial dims are (D=2, H=3, W=4) in (B, C, D, H, W) convention.
    8 combinations = 2^3 (each of the 3 spatial axes either flipped or not).
    """
    model.eval()
    all_probs = []

    for flip_axes in [
        [],       # original
        [2],      # flip D
        [3],      # flip H
        [4],      # flip W
        [2, 3],   # flip D+H
        [2, 4],   # flip D+W
        [3, 4],   # flip H+W
        [2, 3, 4] # flip all
    ]:
        aug_img = images.clone()
        if flip_axes:
            aug_img = torch.flip(aug_img, dims=flip_axes)

        with torch.no_grad():
            logits = model(aug_img, text_emb)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)

        if flip_axes:
            probs = torch.flip(probs, dims=flip_axes)

        all_probs.append(probs)

    return torch.stack(all_probs).mean(dim=0)


def evaluate(
    model,
    text_encoder_base,
    test_loader,
    device,
    cfg,
    use_tta: bool = True,
    compute_hd95: bool = True,
    output_dir: str = 'outputs',
    source_breakdown: bool = True,
):
    """
    Full test set evaluation.

    Args:
        model:            trained BilRAUNet
        text_encoder_base: multilingual-e5-large base
        test_loader:      DataLoader for test split
        device:           torch device
        cfg:              config dict
        use_tta:          whether to use 8-fold flip TTA
        compute_hd95:     whether to compute 95th percentile Hausdorff distance
        output_dir:       where to save results JSON
        source_breakdown: if True, also break results down by brats2020 vs brats2023

    Returns:
        results dict
    """
    model.eval()
    text_encoder_base.eval()
    nc = cfg['model']['num_classes']

    all_metrics   = []
    src_metrics   = {'brats2020': [], 'brats2023': []}

    pbar = tqdm(test_loader, desc='[test eval]', dynamic_ncols=True)
    for images, text_emb, targets, label_ids in pbar:
        images   = images.to(device)
        text_emb = text_emb.to(device)
        targets  = targets.to(device)

        if use_tta:
            probs  = tta_predict(model, images, text_emb)
            logits = torch.log(probs + 1e-8)
        else:
            with torch.no_grad():
                logits = model(images, text_emb)

        m = compute_metrics(logits, targets, num_classes=nc,
                             compute_hd95=compute_hd95)
        all_metrics.append(m)
        pbar.set_postfix({
            'WT': f"{m['brats']['WT']['dice']:.3f}",
            'TC': f"{m['brats']['TC']['dice']:.3f}",
            'ET': f"{m['brats']['ET']['dice']:.3f}",
        })

    # Aggregate
    from training.train import _aggregate_metrics
    results = _aggregate_metrics(all_metrics)

    print('\n' + '='*65)
    print('  TEST RESULTS' + (' + TTA' if use_tta else ''))
    print('='*65)
    brats = results.get('brats', {})
    for region in ['WT', 'TC', 'ET']:
        r = brats.get(region, {})
        hd = r.get('hd95', float('nan'))
        hd_str = f'{hd:.2f}mm' if hd == hd else 'N/A'
        print(f'  {region}:  Dice={r.get("dice",0):.4f}  '
              f'IoU={r.get("iou",0):.4f}  '
              f'Sens={r.get("sensitivity",0):.4f}  '
              f'HD95={hd_str}')
    print('='*65)
    pc = results.get('per_class', {})
    label_names = {0: 'BG', 1: 'NCR', 2: 'Edema', 3: 'ET'}
    for c in range(nc):
        r = pc.get(c, {})
        print(f'  {label_names.get(c,"?"):8}:  '
              f'Dice={r.get("dice",0):.4f}  '
              f'Sens={r.get("sensitivity",0):.4f}  '
              f'Prec={r.get("precision",0):.4f}')
    print('='*65)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'test_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'Results saved → {out_path}')
    return results
