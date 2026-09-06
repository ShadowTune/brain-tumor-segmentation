"""
BraTS dataset collection, deduplication, and stratified train/val/test split.

Key fixes over naive splitting:
  FIX 1 — Deduplication: BraTS2023 Part1 + Part2 share patient IDs.
           We deduplicate by patient_id BEFORE any split to prevent leakage.
  FIX 2 — Stratified split: ensures brats2020 / brats2023 ratio is identical
           in train, val, and test sets.
"""

import glob
import os
from typing import List, Tuple
from sklearn.model_selection import train_test_split


SEED = 42


def is_valid_nii(path: str, min_bytes: int = 1024) -> bool:
    """Check that a NIfTI file exists and is not empty/corrupted."""
    try:
        return os.path.isfile(path) and os.path.getsize(path) >= min_bytes
    except OSError:
        return False


def collect_brats2020_cases(data_root: str) -> List[dict]:
    """
    Scan BraTS2020 training directory and collect valid patient cases.
    Naming convention: BraTS20_Training_XXX / *_flair.nii, *_t1.nii, etc.
    """
    cases, skipped = [], 0
    for patient_dir in sorted(glob.glob(os.path.join(data_root, 'BraTS20_Training_*'))):
        flair = glob.glob(os.path.join(patient_dir, '*_flair.nii*'))
        t1    = glob.glob(os.path.join(patient_dir, '*_t1.nii*'))
        t1ce  = glob.glob(os.path.join(patient_dir, '*_t1ce.nii*'))
        t2    = glob.glob(os.path.join(patient_dir, '*_t2.nii*'))
        seg   = glob.glob(os.path.join(patient_dir, '*_seg.nii*'))
        raw   = [flair, t1, t1ce, t2, seg]
        if not all(raw):
            skipped += 1
            continue
        cands = [p[0] for p in raw]
        if not all(is_valid_nii(p) for p in cands):
            skipped += 1
            print(f'  [SKIP-corrupt] {os.path.basename(patient_dir)}')
            continue
        cases.append({
            'flair':      cands[0], 't1': cands[1],
            't1ce':       cands[2], 't2': cands[3], 'seg': cands[4],
            'source':     'brats2020',
            'patient_id': os.path.basename(patient_dir),
        })
    print(f'  BraTS2020 : {len(cases)} valid, {skipped} skipped')
    return cases


def collect_brats2023_cases(data_root: str) -> List[dict]:
    """
    Scan BraTS2023 directory and collect valid patient cases.
    Naming convention: BraTS-GLI-XXXXX / *-t2f.nii, *-t1n.nii, *-t1c.nii, *-t2w.nii, *-seg.nii
    """
    cases, skipped = [], 0
    for patient_dir in sorted(glob.glob(os.path.join(data_root, 'BraTS-GLI-*'))):
        t2f = glob.glob(os.path.join(patient_dir, '*-t2f.nii*'))
        t1n = glob.glob(os.path.join(patient_dir, '*-t1n.nii*'))
        t1c = glob.glob(os.path.join(patient_dir, '*-t1c.nii*'))
        t2w = glob.glob(os.path.join(patient_dir, '*-t2w.nii*'))
        seg = glob.glob(os.path.join(patient_dir, '*-seg.nii*'))
        raw = [t2f, t1n, t1c, t2w, seg]
        if not all(raw):
            skipped += 1
            continue
        cands = [p[0] for p in raw]
        if not all(is_valid_nii(p) for p in cands):
            skipped += 1
            print(f'  [SKIP-corrupt] {os.path.basename(patient_dir)}')
            continue
        cases.append({
            'flair':      cands[0], 't1': cands[1],
            't1ce':       cands[2], 't2': cands[3], 'seg': cands[4],
            'source':     'brats2023',
            'patient_id': os.path.basename(patient_dir),
        })
    tag = os.path.basename(data_root)
    print(f'  BraTS2023 ({tag}): {len(cases)} valid, {skipped} skipped')
    return cases


def collect_and_deduplicate(
    brats2020_path: str,
    brats2023_p1_path: str,
    brats2023_p2_path: str,
) -> List[dict]:
    """
    Collect cases from all three dataset roots and deduplicate by patient_id.

    BraTS2023 was released in two parts on Kaggle. Patient folders may overlap
    between Part1 and Part2. Deduplication happens here — before any split —
    to guarantee no patient appears in both training and test sets.

    Returns:
        Deduplicated list of case dicts
    """
    cases_2020   = collect_brats2020_cases(brats2020_path)
    cases_2023p1 = collect_brats2023_cases(brats2023_p1_path)
    cases_2023p2 = collect_brats2023_cases(brats2023_p2_path)

    seen_ids, all_cases, dup_count = set(), [], 0
    for c in cases_2020 + cases_2023p1 + cases_2023p2:
        pid = c['patient_id']
        if pid in seen_ids:
            dup_count += 1
            print(f'  [DUP-removed] {pid}')
            continue
        seen_ids.add(pid)
        all_cases.append(c)

    total_raw = len(cases_2020) + len(cases_2023p1) + len(cases_2023p2)
    print(f'\nRaw volumes collected : {total_raw}')
    print(f'Duplicates removed    : {dup_count}')
    print(f'Unique volumes        : {len(all_cases)}')

    src_counts = {}
    for c in all_cases:
        src_counts[c['source']] = src_counts.get(c['source'], 0) + 1
    for src, cnt in sorted(src_counts.items()):
        print(f'  {src}: {cnt}')

    return all_cases


def stratified_split(
    all_cases: List[dict],
    val_ratio:  float = 0.10,
    test_ratio: float = 0.10,
    seed: int = SEED,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Stratified train/val/test split by dataset source (brats2020 / brats2023).
    This ensures each split mirrors the overall source distribution.

    Returns:
        train_idx, val_idx, test_idx  — lists of indices into all_cases
    """
    indices      = list(range(len(all_cases)))
    strat_labels = [all_cases[i]['source'] for i in indices]
    temp_ratio   = val_ratio + test_ratio

    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, strat_labels,
        test_size=temp_ratio, random_state=seed, stratify=strat_labels
    )
    # Within the temp set, split val and test proportionally
    val_frac = val_ratio / temp_ratio
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - val_frac),
        random_state=seed, stratify=temp_labels
    )

    print(f'\nTrain / Val / Test : {len(train_idx)} / {len(val_idx)} / {len(test_idx)}')
    for split_name, split_idx in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
        sc = {}
        for i in split_idx:
            s = all_cases[i]['source']
            sc[s] = sc.get(s, 0) + 1
        breakdown = '  '.join(f'{k}:{v}' for k, v in sorted(sc.items()))
        print(f'  {split_name:5}: {breakdown}')

    return train_idx, val_idx, test_idx
