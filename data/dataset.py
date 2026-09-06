"""
BraTS3DBilingualDataset — PyTorch Dataset for bilingual text-guided segmentation.

Text conditioning sources:
  'prompt' — short class-level prompt embedding (15 options per class per language)
  'report' — per-patient case report embedding (richer, used during training)
             Falls back to prompt if patient has no report.
"""

import random
import torch
from torch.utils.data import Dataset
from data.preprocessing import load_volume, resize_3d, augment_3d_intensity_only


class BraTS3DBilingualDataset(Dataset):
    """
    Args:
        cases:                   list of all case dicts (from dataset_builder)
        indices:                 which indices to use (train / val / test split)
        prompt_embeddings:       dict {label_id: {'en': Tensor(15,D), 'bn': Tensor(15,D)}}
        case_report_embeddings:  dict {patient_id: {'en': Tensor(D), 'bn': Tensor(D)}}
        patch_size:              (H, W, D) — all volumes resized to this
        augment:                 whether to apply intensity augmentation
        text_source:             'prompt' or 'report'
    """

    def __init__(
        self,
        cases,
        indices,
        prompt_embeddings,
        case_report_embeddings=None,
        patch_size=(128, 128, 128),
        augment=False,
        text_source='prompt',
    ):
        assert text_source in ('prompt', 'report'), \
            "text_source must be 'prompt' or 'report'"
        if text_source == 'report' and case_report_embeddings is None:
            raise ValueError("text_source='report' requires case_report_embeddings")

        self.cases                  = [cases[i] for i in indices]
        self.prompt_embeddings      = prompt_embeddings
        self.case_report_embeddings = case_report_embeddings or {}
        self.patch_size             = patch_size
        self.augment                = augment
        self.text_source            = text_source
        self.label_ids              = [1, 2, 3]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        # Robust loading: try up to len(cases) fallbacks on corrupt files
        for attempt in range(len(self.cases)):
            try:
                case   = self.cases[(idx + attempt) % len(self.cases)]
                image, seg = load_volume(case)
                break
            except Exception as e:
                if attempt == 0:
                    print(f'  [WARN] corrupt/skipped idx={idx}: {e}')
        else:
            raise RuntimeError('All fallback volumes failed.')

        # Resize whole volume → patch_size  (H, W, D)
        image, seg = resize_3d(image, seg, self.patch_size)

        # Transpose from (C, H, W, D) → PyTorch (C, D, H, W)
        image = image.transpose(0, 3, 1, 2)
        seg   = seg.transpose(2, 0, 1)

        if self.augment:
            image, seg = augment_3d_intensity_only(image, seg)

        # Choose which label to condition on (prefer present labels)
        present  = [l for l in self.label_ids if (seg == l).sum() > 0]
        if not present:
            present = self.label_ids
        label_id = random.choice(present)

        # Choose language randomly
        lang = random.choice(['en', 'bn'])
        pid  = case['patient_id']

        # Get text embedding
        if self.text_source == 'report' and pid in self.case_report_embeddings:
            text_emb = self.case_report_embeddings[pid][lang]
        else:
            if self.text_source == 'report':
                print(f'  [WARN] no report for {pid}, using fallback prompt')
            embs     = self.prompt_embeddings[label_id][lang]
            text_emb = embs[random.randint(0, len(embs) - 1)]

        return (
            torch.tensor(image,   dtype=torch.float32),
            text_emb.float(),
            torch.tensor(seg.copy(), dtype=torch.long),
            label_id,
        )
