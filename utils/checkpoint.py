"""Save and load training checkpoints."""

import torch
import os


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'epoch':           epoch,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler else None,
        'scaler_state':    scaler.state_dict()    if scaler    else None,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device='cpu'):
    """Load checkpoint. Returns start epoch."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    if optimizer and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    if scheduler and ckpt.get('scheduler_state'):
        scheduler.load_state_dict(ckpt['scheduler_state'])
    if scaler and ckpt.get('scaler_state'):
        scaler.load_state_dict(ckpt['scaler_state'])
    return ckpt.get('epoch', 0)
