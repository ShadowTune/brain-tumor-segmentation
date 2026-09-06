# 🧠 BilRAUNet — Bilingual 3D Brain Tumor Segmentation

**Bilingual Prompt-Driven 3D Brain Tumor Segmentation** using a Text-Guided Attention U-Net (BilRAUNet) on BraTS 2020 + BraTS 2023. Supports English and Bengali clinical prompts via multilingual text encoders.

---

## 🏗️ Architecture

```
Input: 4-channel MRI (FLAIR, T1, T1ce, T2) → (B, 4, 128, 128, 128)
Text:  Bilingual prompt OR case report → multilingual-e5-large → (B, 1024)

                    Text Encoder (intfloat/multilingual-e5-large)
                            ↓  TextProjection  ↓
                        text embedding (B, 256)

    ┌──────────────────────────────────────────────────────────┐
    │              5-Level 3D Attention U-Net                  │
    │                                                          │
    │  Encoder: DoubleConv3D + SE3D (squeeze-excitation)      │
    │  Bottleneck: FiLM + BidirectionalCrossAttention (BCA)   │
    │             (64 tokens only — no OOM)                    │
    │  Decoder:  AttentionGate3D + FiLM at every level        │
    │  Aux heads at decoder levels 3, 4, 5 (deep supervision) │
    └──────────────────────────────────────────────────────────┘
                            ↓
         Segmentation map: (B, 4, 128, 128, 128)
         Classes: 0=BG  1=NCR  2=Edema  3=Enhancing
```

---

## 📁 Project Structure

```
brain-tumor-segmentation/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml                  # All hyperparameters
├── data/
│   ├── prompts.py                   # Bilingual prompt dictionary (15 per class)
│   ├── preprocessing.py             # Volume normalization, resize, augmentation
│   ├── case_reports.py              # Templated case report generation
│   ├── dataset_builder.py           # BraTS2020 + BraTS2023 collection & dedup
│   └── dataset.py                   # BraTS3DBilingualDataset (PyTorch Dataset)
├── models/
│   ├── text_encoder.py              # Multilingual text encoder + encode_text()
│   ├── blocks.py                    # DoubleConv3D, SE3D, AttentionGate3D, FiLM3D, BCA
│   └── bilraunet.py                 # FixedTextGuided3DAttentionUNet (full model)
├── training/
│   ├── loss.py                      # TverskyFocalLoss + ImprovedLoss (Dice+CE+Tversky)
│   ├── metrics.py                   # compute_metrics() — Dice, IoU, HD95, BraTS WT/TC/ET
│   └── train.py                     # Full training loop with TTA, checkpointing, history
├── evaluation/
│   └── evaluate.py                  # Test set evaluation, per-source breakdown, TTA
├── inference/
│   └── predict.py                   # Single-volume inference pipeline
├── visualization/
│   └── plots.py                     # 14 comprehensive metric plots (A→N)
├── utils/
│   └── checkpoint.py                # Save/load checkpoint helpers
└── kaggle_notebook.py               # Complete Kaggle notebook (all cells)
```

---

## 🚀 Quick Start (Kaggle)

```python
import os
os.chdir('/kaggle/working/brain-tumor-segmentation')
!pip install nibabel transformers sentencepiece monai scipy -q
!python kaggle_notebook.py
```

Or cell by cell — see `kaggle_notebook.py`.

---

## 📊 Key Results (BraTS 2020 + 2023, Stage 2)

| Region | Dice | IoU | Sensitivity | HD95 |
|--------|------|-----|-------------|------|
| WT (Whole Tumor) | — | — | — | — |
| TC (Tumor Core)  | — | — | — | — |
| ET (Enhancing)   | — | — | — | — |

*(Fill in after your training run)*

---

## ⚙️ Key Design Decisions

- **BCA at bottleneck only** (4³=64 tokens) → no OOM. Decoder uses FiLM (O(C), no matrix).
- **Gradient checkpointing** on all DoubleConv3D blocks → ~50% VRAM reduction.
- **Deep supervision** with 3 auxiliary heads → faster convergence.
- **Bilingual case reports** at train time; class prompts at val/test → richer training signal.
- **8-fold TTA** at test time → flip ensemble over all axis combinations.

---

## License
MIT
