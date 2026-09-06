"""
BilRAUNet — Bilingual 3D Text-Guided Attention U-Net.
5-level encoder-decoder with:
  • DoubleConv3D + SE3D at every encoder level
  • BCA (bidirectional cross-attention) at bottleneck only
  • FiLM3D + AttentionGate3D at every decoder level
  • Deep supervision (3 auxiliary heads)
  • Gradient checkpointing on all DoubleConv3D blocks

Parameter count: ~28M (base_ch=32, max_ch=256) — comfortably under 50M.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import (
    _cap, DoubleConv3D, SE3D, AttentionGate3D,
    FiLM3D, BidirectionalCrossAttention, TextProjection,
    wrap_grad_checkpoint,
)


class BilRAUNet(nn.Module):
    """
    Args:
        in_channels:   number of MRI modalities (4 for BraTS)
        base_ch:       base channel count (doubles at each level up to max_ch)
        max_ch:        maximum channel count (caps exponential growth)
        num_classes:   segmentation classes (4: BG, NCR, Edema, ET)
        text_in_dim:   text encoder output dim (1024 for multilingual-e5-large)
        text_proj_dim: projected text dim fed into FiLM and BCA
    """

    def __init__(
        self,
        in_channels:   int = 4,
        base_ch:       int = 32,
        max_ch:        int = 256,
        num_classes:   int = 4,
        text_in_dim:   int = 1024,
        text_proj_dim: int = 256,
    ):
        super().__init__()
        C  = lambda m: _cap(base_ch, m, max_ch)
        tp = text_proj_dim
        C1, C2, C3, C4, C5, CB = C(1), C(2), C(4), C(8), C(16), C(32)

        self.text_proj = TextProjection(text_in_dim, tp)
        self.pool      = nn.MaxPool3d(2)

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc1 = DoubleConv3D(in_channels, C1, dropout=0.00)
        self.se1  = SE3D(C1)
        self.enc2 = DoubleConv3D(C1, C2, dropout=0.05)
        self.se2  = SE3D(C2)
        self.enc3 = DoubleConv3D(C2, C3, dropout=0.05)
        self.se3  = SE3D(C3)
        self.enc4 = DoubleConv3D(C3, C4, dropout=0.10)
        self.se4  = SE3D(C4)
        self.enc5 = DoubleConv3D(C4, C5, dropout=0.10)
        self.se5  = SE3D(C5)

        # ── Bottleneck: FiLM + BCA ────────────────────────────────────────────
        # patch=128 → bottleneck 4³=64 tokens → 64×64 attention = negligible VRAM
        self.bot      = DoubleConv3D(C5, CB, dropout=0.20)
        self.film_bot = FiLM3D(tp, CB)
        self.bca_bot  = BidirectionalCrossAttention(CB, text_in_dim, num_heads=4)

        # ── Decoder: AttentionGate + FiLM (NO cross-attention below bottleneck)
        self.up5  = nn.ConvTranspose3d(CB, C5, 2, 2)
        self.ag5  = AttentionGate3D(C5, C5, max(C5 // 2, 1))
        self.dec5 = DoubleConv3D(C5 * 2, C5)
        self.film5 = FiLM3D(tp, C5)

        self.up4  = nn.ConvTranspose3d(C5, C4, 2, 2)
        self.ag4  = AttentionGate3D(C4, C4, max(C4 // 2, 1))
        self.dec4 = DoubleConv3D(C4 * 2, C4)
        self.film4 = FiLM3D(tp, C4)

        self.up3  = nn.ConvTranspose3d(C4, C3, 2, 2)
        self.ag3  = AttentionGate3D(C3, C3, max(C3 // 2, 1))
        self.dec3 = DoubleConv3D(C3 * 2, C3)
        self.film3 = FiLM3D(tp, C3)

        self.up2  = nn.ConvTranspose3d(C3, C2, 2, 2)
        self.ag2  = AttentionGate3D(C2, C2, max(C2 // 2, 1))
        self.dec2 = DoubleConv3D(C2 * 2, C2)
        self.film2 = FiLM3D(tp, C2)

        self.up1  = nn.ConvTranspose3d(C2, C1, 2, 2)
        self.ag1  = AttentionGate3D(C1, C1, max(C1 // 2, 1))
        self.dec1 = DoubleConv3D(C1 * 2, C1)
        self.film1 = FiLM3D(tp, C1)

        # ── Output + auxiliary heads ──────────────────────────────────────────
        self.out_conv = nn.Conv3d(C1, num_classes, 1)
        self.aux5     = nn.Conv3d(C5, num_classes, 1)
        self.aux4     = nn.Conv3d(C4, num_classes, 1)
        self.aux3     = nn.Conv3d(C3, num_classes, 1)

        # Apply gradient checkpointing to all DoubleConv3D blocks
        wrap_grad_checkpoint(self)

        # Verify param count
        total_p = sum(p.numel() for p in self.parameters())
        assert total_p < 50_000_000, \
            f"Model too large: {total_p:,} params (max 50M). Reduce base_ch or max_ch."
        print(f'BilRAUNet: {total_p:,} total params  '
              f'| {sum(p.numel() for p in self.parameters() if p.requires_grad):,} trainable')

    def forward(self, x, text_emb):
        """
        Args:
            x:        (B, 4, D, H, W) MRI volume
            text_emb: (B, text_in_dim) text embedding from encoder

        Returns (training):   (main, aux5, aux4, aux3) — 4-tuple of logits
        Returns (inference):  main logits (B, num_classes, D, H, W)
        """
        t = self.text_proj(text_emb)    # (B, text_proj_dim)

        # Encoder
        e1 = self.se1(self.enc1(x))
        e2 = self.se2(self.enc2(self.pool(e1)))
        e3 = self.se3(self.enc3(self.pool(e2)))
        e4 = self.se4(self.enc4(self.pool(e3)))
        e5 = self.se5(self.enc5(self.pool(e4)))

        # Bottleneck (4×4×4 spatial → 64 tokens, BCA safe)
        b = self.bot(self.pool(e5))
        b = self.film_bot(b, t)
        b = self.bca_bot(b, text_emb)

        # Decoder with attention gates + FiLM
        d5 = self.up5(b)
        d5 = self.film5(self.dec5(torch.cat([d5, self.ag5(d5, e5)], 1)), t)

        d4 = self.up4(d5)
        d4 = self.film4(self.dec4(torch.cat([d4, self.ag4(d4, e4)], 1)), t)

        d3 = self.up3(d4)
        d3 = self.film3(self.dec3(torch.cat([d3, self.ag3(d3, e3)], 1)), t)

        d2 = self.up2(d3)
        d2 = self.film2(self.dec2(torch.cat([d2, self.ag2(d2, e2)], 1)), t)

        d1 = self.up1(d2)
        d1 = self.film1(self.dec1(torch.cat([d1, self.ag1(d1, e1)], 1)), t)

        main = self.out_conv(d1)

        if self.training:
            sz = x.shape[2:]    # original spatial size for upsampling aux heads
            return (
                main,
                F.interpolate(self.aux5(d5), size=sz, mode='trilinear', align_corners=False),
                F.interpolate(self.aux4(d4), size=sz, mode='trilinear', align_corners=False),
                F.interpolate(self.aux3(d3), size=sz, mode='trilinear', align_corners=False),
            )
        return main


def build_model(cfg: dict, device) -> BilRAUNet:
    """Build model from config dict and move to device."""
    mc = cfg['model']
    model = BilRAUNet(
        in_channels=mc['in_channels'],
        base_ch=mc['base_ch'],
        max_ch=mc['max_ch'],
        num_classes=mc['num_classes'],
        text_in_dim=mc['text_in_dim'],
        text_proj_dim=mc['text_proj_dim'],
    ).to(device)
    return model
