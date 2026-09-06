"""
Neural network building blocks for BilRAUNet.

Blocks:
  DoubleConv3D     — two conv-norm-gelu layers with residual shortcut + gradient checkpointing
  SE3D             — squeeze-excitation channel attention
  AttentionGate3D  — spatial attention gate (Oktay et al.)
  FiLM3D           — feature-wise linear modulation by text embedding
  BidirectionalCrossAttention — text ↔ image cross-attention (bottleneck only)
  TextProjection   — MLP projecting text from 1024-dim to text_proj_dim
"""

import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt


def _cap(base: int, mult: int, max_ch: int = 256) -> int:
    """Cap channel count at max_ch to keep model under 50M params."""
    return min(base * mult, max_ch)


class DoubleConv3D(nn.Module):
    """
    Two consecutive Conv3d → InstanceNorm3d → GELU blocks with a residual shortcut.
    Gradient checkpointing is applied in training mode to save ~50% VRAM.

    Why InstanceNorm instead of BatchNorm?
    3D medical imaging uses small batch sizes (2-4). BatchNorm statistics are
    unreliable with batch_size=2. InstanceNorm normalizes per-sample per-channel,
    giving stable gradients regardless of batch size.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True), nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True), nn.GELU(),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True)
            ) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class SE3D(nn.Module):
    """
    3D Squeeze-and-Excitation block.
    Globally pools spatial dims → MLP → per-channel scalar gates.
    Lets the network suppress uninformative channels.
    Reduction ratio r=8 keeps parameter cost low.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(channels, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).view(-1, x.shape[1], 1, 1, 1)


class AttentionGate3D(nn.Module):
    """
    Attention gate from Oktay et al. "Attention U-Net" (2018).
    Suppresses irrelevant feature regions at skip connections.

    g = gating signal (from decoder, coarser resolution)
    x = skip connection (from encoder, same resolution as output)

    Output: x weighted by spatial attention map ψ ∈ [0,1]
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, 1, bias=False),
            nn.InstanceNorm3d(F_int, affine=True)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, 1, bias=False),
            nn.InstanceNorm3d(F_int, affine=True)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, 1, bias=False),
            nn.InstanceNorm3d(1, affine=True),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        gp = self.W_g(g)
        xp = self.W_x(x)
        if gp.shape[2:] != xp.shape[2:]:
            gp = F.interpolate(gp, size=xp.shape[2:], mode='trilinear', align_corners=False)
        return x * self.psi(F.gelu(gp + xp))


class FiLM3D(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on text embedding.
    Applied at every decoder level (O(C) cost — no spatial matrix).

    γ and β are predicted from the text vector via a 2-layer MLP.
    The output is: (γ + 1) * x + β   (residual formulation for stable init)

    Why +1?  At init, γ ≈ 0 so the gate starts as identity (x → x).
    Training gradually learns to modulate based on the text.
    """

    def __init__(self, text_dim: int, num_channels: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_dim, num_channels * 2), nn.GELU(),
            nn.Linear(num_channels * 2, num_channels * 2),
        )

    def forward(self, x, t):
        params = self.fc(t)
        gamma, beta = params.chunk(2, dim=-1)
        return (gamma + 1.0).view(-1, gamma.shape[1], 1, 1, 1) * x \
             + beta.view(-1, beta.shape[1], 1, 1, 1)


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention between image features and text (BCA).
    Placed at the BOTTLENECK ONLY (spatial size 4³ = 64 tokens).

    Why bottleneck only?
    At decoder level 3 the spatial size is 16³ = 4096 tokens.
    A dense attention matrix (B × 4096 × 4096) in fp32 = ~4GB per item.
    At the bottleneck, 4³ = 64 tokens → 64×64 matrix = negligible.

    Two-step bidirectional attention:
      Step 1 T2I: text queries image  → refined image features
      Step 2 I2T: image queries refined → final image features
    """

    def __init__(self, feat_dim: int, text_dim: int, num_heads: int = 4):
        super().__init__()
        assert feat_dim % num_heads == 0
        self.proj_t = nn.Linear(text_dim, feat_dim)
        # Step 1: text → image
        self.attn1  = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm1a = nn.LayerNorm(feat_dim)
        self.ff1    = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2), nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim)
        )
        self.norm1b = nn.LayerNorm(feat_dim)
        # Step 2: image → refined
        self.attn2  = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm2a = nn.LayerNorm(feat_dim)
        self.ff2    = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2), nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim)
        )
        self.norm2b = nn.LayerNorm(feat_dim)

    def forward(self, feat, text_emb):
        B, C, D, H, W = feat.shape
        fi = feat.view(B, C, -1).permute(0, 2, 1)     # (B, N, C)
        t  = self.proj_t(text_emb).unsqueeze(1)        # (B, 1, C)
        # T2I
        fi_r, _ = self.attn1(t, fi, fi)
        fi_r = self.norm1a(fi + fi_r.expand_as(fi))
        fi_r = self.norm1b(fi_r + self.ff1(fi_r))
        # I2T
        fj, _ = self.attn2(fi, fi_r, fi_r)
        fj = self.norm2a(fi + fj)
        fj = self.norm2b(fj + self.ff2(fj))
        return fj.permute(0, 2, 1).view(B, C, D, H, W)


class TextProjection(nn.Module):
    """
    Project text from encoder dim (1024) to model text_proj_dim (256).
    2-layer MLP with LayerNorm + Dropout for regularization.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2), nn.LayerNorm(out_dim * 2), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim * 2, out_dim), nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.proj(x)


def wrap_grad_checkpoint(model: nn.Module):
    """
    Wrap the forward() of every DoubleConv3D block with gradient checkpointing.
    During training, checkpointing discards intermediate activations and recomputes
    them during backward — trading compute for ~50% VRAM saving.
    """
    for module in model.modules():
        if isinstance(module, DoubleConv3D):
            orig = module.forward.__func__

            def _make(fn):
                def _fwd(self_inner, x):
                    if self_inner.training:
                        return grad_ckpt(fn, self_inner, x, use_reentrant=False)
                    return fn(self_inner, x)
                return _fwd

            module.forward = types.MethodType(_make(orig), module)
