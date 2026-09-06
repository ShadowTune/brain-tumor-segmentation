"""
Single-volume inference pipeline.
Given a new patient's NIfTI files + a bilingual text prompt, produces a 3D segmentation mask.
"""

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib

from data.preprocessing import load_volume, resize_3d
from evaluation.evaluate import tta_predict
from models.text_encoder import encode_text


LABEL_NAMES = {0: 'Background', 1: 'NCR/NET', 2: 'Peritumoral Edema', 3: 'Enhancing Tumor'}
LABEL_COLORS = {0: (0,0,0), 1: (255,0,0), 2: (0,255,0), 3: (0,0,255)}


def predict_volume(
    case_dict:         dict,
    model,
    text_encoder_base,
    tokenizer,
    device,
    patch_size:        tuple = (128, 128, 128),
    prompt:            str   = 'Segment all tumor regions in this brain MRI',
    use_tta:           bool  = True,
    save_path:         str   = None,
):
    """
    Run full segmentation pipeline on one patient.

    Args:
        case_dict:  dict with 'flair','t1','t1ce','t2' paths
        model:      trained BilRAUNet (eval mode)
        ...
        prompt:     text prompt to condition the model
        use_tta:    whether to apply 8-fold flip TTA
        save_path:  if given, save predicted NIfTI to this path

    Returns:
        pred_seg: (H, W, D) uint8 numpy array with class labels 0-3
        probs:    (4, H, W, D) float32 class probabilities
    """
    model.eval()
    text_encoder_base.eval()

    # Load and preprocess
    image, _ = load_volume({**case_dict, 'seg': case_dict.get('seg', case_dict['flair'])})
    orig_shape = image.shape[1:]    # (H, W, D)
    image_r, _ = resize_3d(image, np.zeros(orig_shape, dtype=np.uint8), patch_size)

    # (C, H, W, D) → (C, D, H, W)
    image_t = torch.tensor(image_r.transpose(0, 3, 1, 2), dtype=torch.float32).unsqueeze(0).to(device)

    # Encode text
    text_emb = encode_text([prompt], tokenizer, text_encoder_base, device)
    text_emb = text_emb.to(device)

    # Inference
    if use_tta:
        probs = tta_predict(model, image_t, text_emb)[0]   # (C, D, H, W)
    else:
        with torch.no_grad():
            logits = model(image_t, text_emb)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)[0]

    pred_seg = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)   # (D, H, W)
    probs_np = probs.cpu().numpy()

    # Resize prediction back to original volume size (D, H, W) → (H, W, D)
    pred_seg = pred_seg.transpose(1, 2, 0)   # → (H, W, D)

    import scipy.ndimage as ndimage
    zf = tuple(o / p for o, p in zip(orig_shape, patch_size))
    pred_seg = np.round(
        ndimage.zoom(pred_seg.astype(np.float32), zf, order=0, mode='nearest')
    ).astype(np.uint8)

    # Print statistics
    label_names = {0: 'BG', 1: 'NCR', 2: 'Edema', 3: 'ET'}
    print('\nPrediction statistics:')
    for cls, name in label_names.items():
        cnt = (pred_seg == cls).sum()
        vol_cm3 = cnt * 1.0 / 1000
        print(f'  {name:8}: {cnt:8,} voxels  ({vol_cm3:.1f} cm³)')

    # Save NIfTI if requested
    if save_path:
        ref_nii = nib.load(case_dict['flair'])
        out_nii = nib.Nifti1Image(pred_seg, affine=ref_nii.affine, header=ref_nii.header)
        nib.save(out_nii, save_path)
        print(f'Saved → {save_path}')

    return pred_seg, probs_np
