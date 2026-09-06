# ════════════════════════════════════════════════════════════════════
#  BilRAUNet — COMPLETE KAGGLE NOTEBOOK
#  Bilingual 3D Brain Tumor Segmentation (BraTS 2020 + 2023)
#
#  KAGGLE SETTINGS:
#      Settings -> Accelerator -> GPU T4 x2  (32GB total VRAM)
#      Settings -> Internet -> ON
#      Settings -> Persistence -> Files
# ════════════════════════════════════════════════════════════════════

# ── CELL 1: Setup ─────────────────────────────────────────────────
import os, sys, json, time
os.chdir('/kaggle/working/brain-tumor-segmentation')
os.system('pip install nibabel transformers sentencepiece monai scipy -q')

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("GPU count  :", torch.cuda.device_count())
print("Device     :", device)
total_vram = sum(torch.cuda.get_device_properties(i).total_memory
                 for i in range(torch.cuda.device_count())) / 1e9
print(f"Total VRAM : {total_vram:.1f} GB")

# ── CELL 2: Load config ───────────────────────────────────────────
import yaml
with open('configs/config.yaml') as f:
    cfg = yaml.safe_load(f)

# Edit these paths to match your Kaggle input datasets
cfg['paths']['brats2020']    = '/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
cfg['paths']['brats2023_p1'] = '/kaggle/input/brats2023-part-1'
cfg['paths']['brats2023_p2'] = '/kaggle/input/brats2023-part-2zip'
cfg['paths']['output_dir']   = '/kaggle/working/outputs'
cfg['stage'] = 2
os.makedirs(cfg['paths']['output_dir'], exist_ok=True)
print('Config loaded. Stage:', cfg['stage'])

# ── CELL 3: Collect and split datasets ───────────────────────────
from data.dataset_builder import collect_and_deduplicate, stratified_split
all_cases = collect_and_deduplicate(
    brats2020_path    = cfg['paths']['brats2020'],
    brats2023_p1_path = cfg['paths']['brats2023_p1'],
    brats2023_p2_path = cfg['paths']['brats2023_p2'],
)
train_idx, val_idx, test_idx = stratified_split(all_cases)
print(f'Split: {len(train_idx)} train | {len(val_idx)} val | {len(test_idx)} test')

# ── CELL 4: Load text encoder ────────────────────────────────────
from models.text_encoder import (load_text_encoder, precompute_prompt_embeddings,
                                  precompute_report_embeddings)
tokenizer, text_encoder_base, BASE_TEXT_DIM = load_text_encoder(device, stage=cfg['stage'])
cfg['model']['text_in_dim'] = BASE_TEXT_DIM

# ── CELL 5: Prompt embeddings ────────────────────────────────────
from data.prompts import ALL_PROMPTS
prompt_embeddings = precompute_prompt_embeddings(ALL_PROMPTS, tokenizer, text_encoder_base, device)
print('Prompt embeddings ready')

# ── CELL 6: Case report embeddings (optional) ────────────────────
case_report_embeddings = None
report_path = cfg['paths'].get('case_reports')
if report_path and os.path.isfile(report_path):
    with open(report_path) as f:
        reports = json.load(f)
    case_report_embeddings = precompute_report_embeddings(
        reports, tokenizer, text_encoder_base, device)
    print(f'Report embeddings: {len(case_report_embeddings)} patients')
else:
    print('No reports found - using prompt embeddings only')
    print('Generate reports with: from data.case_reports import generate_all_reports')

# ── CELL 7: Datasets + DataLoaders ──────────────────────────────
from torch.utils.data import DataLoader
from data.dataset import BraTS3DBilingualDataset
tc = cfg['training']; dc = cfg['data']
p_size = tuple(tc['patch_size'])

train_ds = BraTS3DBilingualDataset(all_cases, train_idx, prompt_embeddings,
    case_report_embeddings=case_report_embeddings,
    patch_size=p_size, augment=True, text_source=dc['text_source_train'])
val_ds   = BraTS3DBilingualDataset(all_cases, val_idx, prompt_embeddings,
    case_report_embeddings=case_report_embeddings,
    patch_size=p_size, augment=False, text_source=dc['text_source_val'])
test_ds  = BraTS3DBilingualDataset(all_cases, test_idx, prompt_embeddings,
    case_report_embeddings=case_report_embeddings,
    patch_size=p_size, augment=False, text_source=dc['text_source_test'])

train_loader = DataLoader(train_ds, batch_size=tc['batch_size'],
    shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
print(f'Loaders: {len(train_loader)} train | {len(val_loader)} val | {len(test_loader)} test')

# Sanity check
images, text_emb, targets, label_ids = next(iter(train_loader))
print(f'image:{images.shape}  text:{text_emb.shape}  target:{targets.shape}')

# ── CELL 8: Build model ──────────────────────────────────────────
from models.bilraunet import build_model
model = build_model(cfg, device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    print(f'DataParallel on {torch.cuda.device_count()} GPUs')

# Forward pass smoke test
with torch.no_grad():
    dummy = torch.zeros(1, 4, 64, 64, 64, device=device)
    dtxt  = torch.zeros(1, BASE_TEXT_DIM, device=device)
    model.train(); out = model(dummy, dtxt)
    main  = out[0] if isinstance(out, tuple) else out
    print(f'Train fwd OK: {main.shape}')
    model.eval();  out = model(dummy, dtxt)
    main  = out[0] if isinstance(out, tuple) else out
    print(f'Eval  fwd OK: {main.shape}')

# ── CELL 9: Train ────────────────────────────────────────────────
from training.train import train as run_training
history = run_training(
    model=model, text_encoder_base=text_encoder_base,
    train_loader=train_loader, val_loader=val_loader,
    cfg=cfg, device=device,
    output_dir=cfg['paths']['output_dir'],
    resume=True,
)
print('Training finished!')

# ── CELL 10: Plot training curves ────────────────────────────────
from visualization.plots import generate_all_plots
plots_dir = os.path.join(cfg['paths']['output_dir'], 'plots')
generate_all_plots(history_or_path=history, save_dir=plots_dir)
print(f'Plots saved to {plots_dir}/')

# ── CELL 11: Evaluate on test set ────────────────────────────────
from utils.checkpoint import load_checkpoint
from evaluation.evaluate import evaluate
best_ckpt = os.path.join(cfg['paths']['output_dir'], 'best_wt_model.pth')
if os.path.isfile(best_ckpt):
    load_checkpoint(best_ckpt, model, device=device)
    print(f'Loaded best WT model')
test_results = evaluate(
    model=model, text_encoder_base=text_encoder_base,
    test_loader=test_loader, device=device, cfg=cfg,
    use_tta=cfg['evaluation']['use_tta'],
    compute_hd95=cfg['evaluation']['compute_hd95'],
    output_dir=cfg['paths']['output_dir'],
)

# ── CELL 12: Final plots with test results ────────────────────────
generate_all_plots(
    history_or_path=history,
    test_results_or_path=test_results,
    save_dir=plots_dir,
)

# ── CELL 13: Inference demo on one test case ─────────────────────
from inference.predict import predict_volume
from visualization.plots import plot_N_slice_overlay
from data.preprocessing import load_volume, resize_3d
import numpy as np

sample_case = all_cases[test_idx[0]]
print(f'Running inference: {sample_case["patient_id"]}')
pred_seg, probs = predict_volume(
    case_dict=sample_case, model=model,
    text_encoder_base=text_encoder_base, tokenizer=tokenizer,
    device=device, patch_size=p_size,
    prompt='Segment all tumor regions including necrotic core, edema, and enhancing tumor',
    use_tta=True,
    save_path=os.path.join(cfg['paths']['output_dir'], 'sample_pred.nii.gz'),
)
orig_image, gt_seg = load_volume(sample_case)
orig_r, gt_r = resize_3d(orig_image, gt_seg, p_size)
plot_N_slice_overlay(
    image_np=orig_r,
    pred_seg=pred_seg.transpose(2,0,1),
    gt_seg=gt_r.transpose(2,0,1),
    save_dir=plots_dir,
    case_id=sample_case['patient_id'],
)

# ── CELL 14: Generate case reports (optional, ~2-3 hrs) ──────────
# from data.case_reports import generate_all_reports
# reports = generate_all_reports(
#     all_cases=all_cases,
#     output_path=os.path.join(cfg['paths']['output_dir'], 'case_reports.json'),
# )

# ── CELL 15: File summary ────────────────────────────────────────
print('\nOutput files:')
out_dir = cfg['paths']['output_dir']
for root, dirs, files in os.walk(out_dir):
    dirs.sort(); files.sort()
    level  = root.replace(out_dir, '').count(os.sep)
    indent = '  ' * level
    print(f'{indent}{os.path.basename(root)}/')
    for f in files:
        size = os.path.getsize(os.path.join(root, f)) / 1e6
        print(f'{indent}  {f}  ({size:.1f} MB)')
