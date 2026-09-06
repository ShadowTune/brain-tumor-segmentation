"""
Full training loop for BilRAUNet.

Features:
  • AMP (automatic mixed precision) — fp16 forward, fp32 optimizer
  • Gradient accumulation — effective batch = batch_size × grad_accum
  • Cosine LR scheduler with linear warmup
  • Checkpoint save/resume (model, optimizer, scheduler, history)
  • Per-epoch metric logging for all classes + BraTS regions
  • Stage-aware text encoder freezing
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.loss import ImprovedLoss
from training.metrics import compute_metrics
from utils.checkpoint import save_checkpoint, load_checkpoint


def build_optimizer(model, text_encoder_base, cfg):
    """
    Separate parameter groups with different learning rates:
      - Text encoder unfrozen layers: LR × 0.1  (pre-trained, needs gentle update)
      - Segmentation model:           LR × 1.0
    """
    base_lr = cfg['training']['learning_rate']
    wd      = cfg['training']['weight_decay']

    text_params = [p for p in text_encoder_base.parameters() if p.requires_grad]
    model_params = list(model.parameters())

    param_groups = [
        {'params': model_params, 'lr': base_lr, 'weight_decay': wd},
    ]
    if text_params:
        param_groups.append(
            {'params': text_params, 'lr': base_lr * 0.1, 'weight_decay': wd}
        )

    return torch.optim.AdamW(param_groups)


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    """Cosine annealing with linear warmup."""
    from torch.optim.lr_scheduler import OneCycleLR
    tc     = cfg['training']
    epochs = tc['num_epochs']
    return OneCycleLR(
        optimizer,
        max_lr=[g['lr'] for g in optimizer.param_groups],
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=tc['warmup_epochs'] / epochs,
        anneal_strategy='cos',
    )


def run_epoch(
    model,
    text_encoder_base,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    cfg,
    phase='train',
    grad_accum=4,
):
    """
    Run one epoch (train or val).

    Returns:
        epoch_loss: float
        epoch_metrics: dict (mean + per_class + brats)
    """
    is_train = phase == 'train'
    model.train(is_train)
    text_encoder_base.train(is_train and cfg['stage'] == 2)

    total_loss = 0.0
    all_metrics = []
    n_batches   = 0

    if is_train:
        optimizer.zero_grad()

    pbar = tqdm(loader, desc=f'[{phase}]', leave=False, dynamic_ncols=True)

    for step, (images, text_emb, targets, _) in enumerate(pbar):
        images   = images.to(device, non_blocking=True)
        text_emb = text_emb.to(device, non_blocking=True)
        targets  = targets.to(device, non_blocking=True)

        ctx = autocast() if scaler is not None else torch.no_grad.__class__()
        ctx = autocast() if is_train else torch.no_grad()

        with ctx:
            outputs = model(images, text_emb)
            loss    = criterion(outputs, targets)
            if is_train:
                loss = loss / grad_accum

        if is_train:
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        list(model.parameters()) +
                        [p for p in text_encoder_base.parameters() if p.requires_grad],
                        max_norm=1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(
                        list(model.parameters()) +
                        [p for p in text_encoder_base.parameters() if p.requires_grad],
                        max_norm=1.0
                    )
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        batch_loss = loss.item() * (grad_accum if is_train else 1)
        total_loss += batch_loss
        n_batches  += 1

        with torch.no_grad():
            m = compute_metrics(outputs, targets,
                                num_classes=cfg['model']['num_classes'],
                                compute_hd95=False)
            all_metrics.append(m)

        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'wt_dice': f"{m['brats']['WT']['dice']:.3f}",
            'et_dice': f"{m['brats']['ET']['dice']:.3f}",
        })

    epoch_loss = total_loss / max(n_batches, 1)

    # Aggregate metrics across batches
    epoch_metrics = _aggregate_metrics(all_metrics)
    return epoch_loss, epoch_metrics


def _aggregate_metrics(metric_list: list) -> dict:
    """Average metric dicts across all batches in an epoch."""
    if not metric_list:
        return {}
    agg = {'mean': {}, 'per_class': {}, 'brats': {}}
    keys_mean = list(metric_list[0]['mean'].keys())
    for k in keys_mean:
        vals = [m['mean'][k] for m in metric_list
                if m['mean'].get(k) is not None and
                not (isinstance(m['mean'].get(k), float) and
                     m['mean'].get(k) != m['mean'].get(k))]
        agg['mean'][k] = sum(vals) / max(len(vals), 1)

    nc = list(metric_list[0]['per_class'].keys())
    for c in nc:
        agg['per_class'][c] = {}
        for k in metric_list[0]['per_class'][c]:
            vals = [m['per_class'][c][k] for m in metric_list
                    if not (isinstance(m['per_class'][c].get(k), float) and
                            m['per_class'][c].get(k) != m['per_class'][c].get(k))]
            agg['per_class'][c][k] = sum(vals) / max(len(vals), 1)

    for region in ['WT', 'TC', 'ET']:
        agg['brats'][region] = {}
        for k in metric_list[0]['brats'][region]:
            if k == 'gt_present':
                continue
            vals = [m['brats'][region][k] for m in metric_list
                    if not (isinstance(m['brats'][region].get(k), float) and
                            m['brats'][region].get(k) != m['brats'][region].get(k))]
            agg['brats'][region][k] = sum(vals) / max(len(vals), 1)
    return agg


def log_epoch(epoch, n_epochs, phase, loss, metrics):
    """Pretty-print one epoch summary to stdout."""
    brats = metrics.get('brats', {})
    wt    = brats.get('WT', {})
    tc    = brats.get('TC', {})
    et    = brats.get('ET', {})
    pc    = metrics.get('per_class', {})
    print(
        f'  Ep {epoch+1:03}/{n_epochs}  [{phase}]  '
        f'loss={loss:.4f}  '
        f'WT_dice={wt.get("dice",0):.4f}  '
        f'TC_dice={tc.get("dice",0):.4f}  '
        f'ET_dice={et.get("dice",0):.4f}  '
        f'| NCR={pc.get(1,{}).get("dice",0):.3f}  '
        f'ED={pc.get(2,{}).get("dice",0):.3f}  '
        f'ET={pc.get(3,{}).get("dice",0):.3f}'
    )


def train(
    model,
    text_encoder_base,
    train_loader,
    val_loader,
    cfg,
    device,
    output_dir: str = 'outputs',
    resume: bool = True,
):
    """
    Full training loop.

    Args:
        model:             BilRAUNet instance
        text_encoder_base: multilingual-e5-large base model
        train_loader:      DataLoader for training set
        val_loader:        DataLoader for validation set
        cfg:               config dict
        device:            torch device
        output_dir:        directory to save checkpoints and history
        resume:            if True, load from checkpoint_load path in cfg

    Returns:
        history: dict with train/val loss and metrics per epoch
    """
    os.makedirs(output_dir, exist_ok=True)
    tc      = cfg['training']
    nc      = cfg['model']['num_classes']
    n_ep    = tc['num_epochs']
    g_accum = tc['grad_accum']

    criterion = ImprovedLoss(
        num_classes=nc,
        class_weights=cfg['loss']['class_weights'],
    ).to(device)

    optimizer = build_optimizer(model, text_encoder_base, cfg)
    steps_per_ep = len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, steps_per_ep)
    scaler    = GradScaler() if tc.get('use_amp', True) else None

    history = {'train_loss': [], 'val_loss': [],
               'train_metrics': [], 'val_metrics': []}
    start_epoch     = 0
    best_wt_dice    = 0.0
    best_mean_dice  = 0.0

    # ── Resume from checkpoint ────────────────────────────────────────────────
    ckpt_load = cfg['paths'].get('checkpoint_load')
    hist_load = cfg['paths'].get('history_load')

    if resume and ckpt_load and os.path.isfile(ckpt_load):
        start_epoch = load_checkpoint(
            ckpt_load, model, optimizer, scheduler, scaler, device
        )
        print(f'Resumed from epoch {start_epoch}')
    else:
        print('Starting from scratch')

    if resume and hist_load and os.path.isfile(hist_load):
        with open(hist_load) as f:
            history = json.load(f)
        print(f'Loaded history ({len(history["train_loss"])} epochs)')

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, n_ep):
        t0 = time.time()

        tr_loss, tr_m = run_epoch(
            model, text_encoder_base, train_loader,
            criterion, optimizer, scheduler, scaler,
            device, cfg, phase='train', grad_accum=g_accum
        )
        vl_loss, vl_m = run_epoch(
            model, text_encoder_base, val_loader,
            criterion, None, None, None,
            device, cfg, phase='val', grad_accum=1
        )

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_metrics'].append(tr_m)
        history['val_metrics'].append(vl_m)

        log_epoch(epoch, n_ep, 'train', tr_loss, tr_m)
        log_epoch(epoch, n_ep, 'val',   vl_loss, vl_m)
        print(f'  Epoch time: {time.time()-t0:.1f}s')

        # ── Save last checkpoint ──────────────────────────────────────────────
        save_checkpoint(
            os.path.join(output_dir, 'last_model.pth'),
            model, optimizer, scheduler, scaler, epoch + 1
        )

        # ── Save best checkpoint ──────────────────────────────────────────────
        wt_dice   = vl_m.get('brats', {}).get('WT', {}).get('dice', 0.0)
        mean_dice = vl_m.get('mean', {}).get('dice', 0.0)

        if wt_dice > best_wt_dice:
            best_wt_dice = wt_dice
            save_checkpoint(
                os.path.join(output_dir, 'best_wt_model.pth'),
                model, optimizer, scheduler, scaler, epoch + 1
            )
            print(f'  ✓ New best WT Dice: {best_wt_dice:.4f}')

        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            save_checkpoint(
                os.path.join(output_dir, 'best_mean_model.pth'),
                model, optimizer, scheduler, scaler, epoch + 1
            )
            print(f'  ✓ New best mean Dice: {best_mean_dice:.4f}')

        # ── Save history ──────────────────────────────────────────────────────
        with open(os.path.join(output_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2, default=str)

    print(f'\nTraining complete.')
    print(f'Best WT Dice  : {best_wt_dice:.4f}')
    print(f'Best mean Dice: {best_mean_dice:.4f}')
    return history
