"""
Templated bilingual case report generation for BraTS volumes.
Extracts anatomical facts from segmentation masks and fills English + Bengali templates.
These reports are used as richer text conditioning during training (vs short prompts).

Based on the four-part structure from the BilRAUNet paper (Fig. 3):
  1. Location (hemisphere, AP region, SI region, total volume, shape)
  2. Peritumoral edema (presence, fraction, FLAIR signal)
  3. Necrotic core (presence, fraction, T1ce signal)
  4. Enhancing tumor (presence, fraction, T1ce signal)
"""

import json
import os
import numpy as np
import scipy.ndimage as ndimage
import nibabel as nib
from typing import Dict, List
from data.preprocessing import normalize_volume


# ── Anatomical fact extraction ────────────────────────────────────────────────

def compute_hemisphere(seg: np.ndarray) -> str:
    """Left / right / bilateral based on centroid of foreground voxels."""
    fg = np.argwhere(seg > 0)
    if len(fg) == 0:
        return "indeterminate"
    cx  = fg[:, 0].mean()
    mid = seg.shape[0] / 2.0
    tol = seg.shape[0] * 0.06    # 6% tolerance for "midline"
    if cx < mid - tol:   return "left"
    elif cx > mid + tol: return "right"
    else:                return "bilateral/midline"


def compute_ap_si_region(seg: np.ndarray):
    """Anterior-posterior and superior-inferior location from centroid."""
    fg = np.argwhere(seg > 0)
    if len(fg) == 0:
        return "indeterminate", "indeterminate"
    cy, cz = fg[:, 1].mean(), fg[:, 2].mean()
    H, _, D = seg.shape
    ap = "posterior" if cy < H / 3.0 else ("anterior" if cy > 2 * H / 3.0 else "central")
    si = "inferior"  if cz < D / 3.0 else ("superior" if cz > 2 * D / 3.0 else "mid-axial")
    return ap, si


def compute_volume_cm3(seg: np.ndarray, label_ids, voxel_spacing=(1.0, 1.0, 1.0)) -> float:
    voxel_vol_mm3 = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
    return float(np.isin(seg, label_ids).sum() * voxel_vol_mm3 / 1000.0)


def compute_subregion_fractions(seg: np.ndarray) -> dict:
    total = (seg > 0).sum()
    if total == 0:
        return {1: 0.0, 2: 0.0, 3: 0.0}
    return {lbl: float((seg == lbl).sum()) / total for lbl in (1, 2, 3)}


def compute_compactness(seg: np.ndarray) -> str:
    mask = seg > 0
    vol  = mask.sum()
    if vol == 0:
        return "indeterminate"
    struct   = ndimage.generate_binary_structure(3, 1)
    surface  = mask ^ ndimage.binary_erosion(mask, struct)
    ratio    = surface.sum() / (vol ** (2.0 / 3.0) + 1e-8)
    if ratio < 4.0:  return "compact and well-circumscribed"
    if ratio < 6.5:  return "moderately irregular"
    return "diffuse and irregularly shaped"


def compute_signal_descriptor(norm_vol: np.ndarray, region_mask: np.ndarray,
                               brain_mask: np.ndarray) -> str:
    if region_mask.sum() == 0:
        return "not present"
    region_mean = norm_vol[region_mask].mean()
    brain_mean  = norm_vol[brain_mask].mean() if brain_mask.sum() > 0 else 0.0
    diff = region_mean - brain_mean
    if diff > 0.5:  return "high signal"
    if diff < -0.5: return "low signal"
    return "mixed/heterogeneous signal"


def extract_case_facts(case_dict: dict) -> dict:
    """Extract all anatomical and signal facts needed to fill the report template."""
    flair_raw = np.asarray(nib.load(case_dict['flair']).dataobj)
    t1ce_raw  = np.asarray(nib.load(case_dict['t1ce']).dataobj)
    seg       = np.asarray(nib.load(case_dict['seg']).dataobj).astype(np.uint8)
    seg[seg == 4] = 3

    flair      = normalize_volume(flair_raw)
    t1ce       = normalize_volume(t1ce_raw)
    brain_mask = flair != 0

    hemisphere = compute_hemisphere(seg)
    ap, si     = compute_ap_si_region(seg)
    vol_total  = compute_volume_cm3(seg, [1, 2, 3])
    fracs      = compute_subregion_fractions(seg)
    shape_desc = compute_compactness(seg)

    edema_mask, necrotic_mask, enhancing_mask = seg == 2, seg == 1, seg == 3

    return {
        'patient_id':             case_dict['patient_id'],
        'hemisphere':             hemisphere,
        'ap_region':              ap,
        'si_region':              si,
        'volume_cm3':             round(vol_total, 1),
        'frac_necrotic':          round(fracs[1], 3),
        'frac_edema':             round(fracs[2], 3),
        'frac_enhancing':         round(fracs[3], 3),
        'shape_descriptor':       shape_desc,
        'edema_present':          bool(edema_mask.sum() > 0),
        'necrotic_present':       bool(necrotic_mask.sum() > 0),
        'enhancing_present':      bool(enhancing_mask.sum() > 0),
        'edema_signal_flair':     compute_signal_descriptor(flair, edema_mask, brain_mask),
        'necrotic_signal_t1ce':   compute_signal_descriptor(t1ce, necrotic_mask, brain_mask),
        'enhancing_signal_t1ce':  compute_signal_descriptor(t1ce, enhancing_mask, brain_mask),
    }


# ── English template ──────────────────────────────────────────────────────────

def fill_template_en(f: dict) -> str:
    """Fill the four-part English report template (paper Fig. 3)."""
    parts = [
        f"The lesion is located in the {f['hemisphere']} hemisphere, "
        f"{f['ap_region']}-{f['si_region']} region of the brain, "
        f"with a total tumor volume of approximately {f['volume_cm3']} cm3. "
        f"The overall shape is {f['shape_descriptor']}."
    ]
    if f['edema_present']:
        parts.append(
            f"Peritumoral edema is present, comprising about "
            f"{int(f['frac_edema']*100)}% of the tumor volume, showing "
            f"{f['edema_signal_flair']} on FLAIR."
        )
    else:
        parts.append("No significant peritumoral edema is observed.")

    if f['necrotic_present']:
        parts.append(
            f"A necrotic core is present, comprising about "
            f"{int(f['frac_necrotic']*100)}% of the tumor volume, showing "
            f"{f['necrotic_signal_t1ce']} on T1ce."
        )
    else:
        parts.append("No distinct necrotic core is observed.")

    if f['enhancing_present']:
        parts.append(
            f"Enhancing tumor tissue comprises about "
            f"{int(f['frac_enhancing']*100)}% of the tumor volume, showing "
            f"{f['enhancing_signal_t1ce']} on T1ce."
        )
    else:
        parts.append("No enhancing tumor component is observed.")
    return " ".join(parts)


# ── Bengali template ──────────────────────────────────────────────────────────

_HEMI_BN = {
    "left": "বাম", "right": "ডান", "bilateral/midline": "দ্বিপাক্ষিক/মধ্যরেখা",
    "indeterminate": "অনির্ধারিত"
}
_AP_BN = {
    "posterior": "পশ্চাৎ", "anterior": "পূর্ববর্তী", "central": "কেন্দ্রীয়",
    "indeterminate": "অনির্ধারিত"
}
_SI_BN = {
    "superior": "উপরিভাগ", "inferior": "নিম্নভাগ", "mid-axial": "মধ্য-অ্যাক্সিয়াল",
    "indeterminate": "অনির্ধারিত"
}
_SHAPE_BN = {
    "compact and well-circumscribed": "সুসংহত এবং সুস্পষ্ট সীমানাযুক্ত",
    "moderately irregular": "মাঝারিভাবে অনিয়মিত",
    "diffuse and irregularly shaped": "বিচ্ছুরিত এবং অনিয়মিত আকৃতির",
    "indeterminate": "অনির্ধারিত"
}
_SIG_BN = {
    "high signal": "উচ্চ সংকেত", "low signal": "কম সংকেত",
    "mixed/heterogeneous signal": "মিশ্র/ভিন্নধর্মী সংকেত", "not present": "অনুপস্থিত"
}


def fill_template_bn(f: dict) -> str:
    """Fill the four-part Bengali report template."""
    hemi  = _HEMI_BN.get(f['hemisphere'], f['hemisphere'])
    ap    = _AP_BN.get(f['ap_region'], f['ap_region'])
    si    = _SI_BN.get(f['si_region'], f['si_region'])
    shape = _SHAPE_BN.get(f['shape_descriptor'], f['shape_descriptor'])

    parts = [
        f"ক্ষতটি মস্তিষ্কের {hemi} গোলার্ধের {ap}-{si} অঞ্চলে অবস্থিত, "
        f"মোট টিউমারের আয়তন প্রায় {f['volume_cm3']} সেমি³। "
        f"সামগ্রিক আকৃতি {shape}।"
    ]
    if f['edema_present']:
        sig = _SIG_BN.get(f['edema_signal_flair'], f['edema_signal_flair'])
        parts.append(
            f"পেরিটিউমোরাল এডিমা বিদ্যমান, টিউমারের আয়তনের প্রায় "
            f"{int(f['frac_edema']*100)}% জুড়ে, FLAIR-এ {sig} দেখা যাচ্ছে।"
        )
    else:
        parts.append("কোনো উল্লেখযোগ্য পেরিটিউমোরাল এডিমা পরিলক্ষিত হয়নি।")

    if f['necrotic_present']:
        sig = _SIG_BN.get(f['necrotic_signal_t1ce'], f['necrotic_signal_t1ce'])
        parts.append(
            f"নেক্রোটিক কোর বিদ্যমান, টিউমারের আয়তনের প্রায় "
            f"{int(f['frac_necrotic']*100)}% জুড়ে, T1ce-এ {sig} দেখা যাচ্ছে।"
        )
    else:
        parts.append("কোনো স্বতন্ত্র নেক্রোটিক কোর পরিলক্ষিত হয়নি।")

    if f['enhancing_present']:
        sig = _SIG_BN.get(f['enhancing_signal_t1ce'], f['enhancing_signal_t1ce'])
        parts.append(
            f"এনহান্সিং টিউমার টিস্যু টিউমারের আয়তনের প্রায় "
            f"{int(f['frac_enhancing']*100)}% জুড়ে, T1ce-এ {sig} দেখা যাচ্ছে।"
        )
    else:
        parts.append("কোনো এনহান্সিং টিউমার উপাদান পরিলক্ষিত হয়নি।")
    return " ".join(parts)


# ── Batch report generation ───────────────────────────────────────────────────

def generate_all_reports(
    all_cases: List[dict],
    output_path: str = 'case_reports.json',
    verbose: bool = True
) -> Dict[str, dict]:
    """
    Generate bilingual case reports for all cases and save to JSON.

    Returns:
        dict mapping patient_id → {'facts': {...}, 'report_en': str, 'report_bn': str}
    """
    reports = {}
    for i, case in enumerate(all_cases):
        try:
            facts = extract_case_facts(case)
            pid   = facts['patient_id']
            reports[pid] = {
                'facts':     facts,
                'report_en': fill_template_en(facts),
                'report_bn': fill_template_bn(facts),
            }
        except Exception as e:
            if verbose:
                print(f'  [SKIP] {case.get("patient_id", "?")}: {e}')
            continue
        if verbose and (i + 1) % 100 == 0:
            print(f'  processed {i+1}/{len(all_cases)}')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f'Saved {len(reports)} reports → {output_path}')
    return reports
