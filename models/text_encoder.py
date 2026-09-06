"""
Multilingual text encoder — wraps intfloat/multilingual-e5-large.
Handles stage-aware freezing, mean-pooled normalized embeddings,
and pre-computation of prompt + case-report embeddings.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict


TEXT_ENCODER_NAME = 'intfloat/multilingual-e5-large'


def load_text_encoder(device, stage: int = 2):
    """
    Load the multilingual-e5-large encoder with stage-aware freezing.

    Stage 1: fully frozen (image-only pretraining — no text gradient)
    Stage 2: unfreeze last 4 transformer layers + pooler for fine-tuning

    Returns:
        tokenizer, text_encoder_base, BASE_TEXT_DIM
    """
    print(f'Loading tokenizer & model: {TEXT_ENCODER_NAME}')
    tokenizer         = AutoTokenizer.from_pretrained(TEXT_ENCODER_NAME)
    text_encoder_base = AutoModel.from_pretrained(TEXT_ENCODER_NAME).to(device)
    text_encoder_base.eval()

    BASE_TEXT_DIM = text_encoder_base.config.hidden_size
    print(f'Text encoder output dim: {BASE_TEXT_DIM}')

    # Freeze everything first
    for p in text_encoder_base.parameters():
        p.requires_grad = False

    if stage == 2:
        # Unfreeze last 4 transformer layers + pooler
        for name, p in text_encoder_base.named_parameters():
            if any(f'encoder.layer.{i}' in name for i in [20, 21, 22, 23]) \
                    or 'pooler' in name:
                p.requires_grad = True

    trainable = sum(p.numel() for p in text_encoder_base.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in text_encoder_base.parameters())
    print(f'Text encoder : {total:,} total  |  {trainable:,} trainable (Stage {stage})')

    return tokenizer, text_encoder_base, BASE_TEXT_DIM


def encode_text(
    prompts: List[str],
    tokenizer,
    text_encoder_base,
    device,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Encode a list of text prompts to L2-normalized embeddings.

    Uses mean pooling over token embeddings (weighted by attention mask).
    Prefix 'query: ' is prepended per multilingual-e5-large convention.

    Args:
        prompts: list of text strings
        ...
    Returns:
        Tensor of shape (N, BASE_TEXT_DIM) on CPU, float32, L2-normalized
    """
    all_embeddings = []
    ctx = torch.no_grad() if not text_encoder_base.training else torch.enable_grad()
    with ctx:
        for i in range(0, len(prompts), batch_size):
            batch = [f'query: {p}' for p in prompts[i: i + batch_size]]
            enc   = tokenizer(
                batch, padding=True, truncation=True,
                max_length=64, return_tensors='pt'
            ).to(device)
            out   = text_encoder_base(**enc)
            mask  = enc['attention_mask'].unsqueeze(-1).float()
            emb   = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            emb   = F.normalize(emb, dim=-1)
            all_embeddings.append(emb.cpu())
    return torch.cat(all_embeddings, dim=0)


def precompute_prompt_embeddings(
    prompts_dict: Dict,          # {label_id: {'en': [...], 'bn': [...]}}
    tokenizer,
    text_encoder_base,
    device,
) -> Dict:
    """
    Pre-compute embeddings for all class prompts (both languages).
    Returns dict: {label_id: {'en': Tensor(15, D), 'bn': Tensor(15, D)}}
    """
    print('Pre-computing prompt embeddings...')
    prompt_embeddings = {}
    for label_id, lang_prompts in prompts_dict.items():
        prompt_embeddings[label_id] = {}
        for lang, texts in lang_prompts.items():
            embs = encode_text(texts, tokenizer, text_encoder_base, device)
            prompt_embeddings[label_id][lang] = embs
            print(f'  Label {label_id} [{lang}]: {embs.shape}')
    return prompt_embeddings


def precompute_report_embeddings(
    reports: Dict,               # {patient_id: {'report_en': str, 'report_bn': str}}
    tokenizer,
    text_encoder_base,
    device,
) -> Dict:
    """
    Pre-compute embeddings for all patient case reports (both languages).
    Returns dict: {patient_id: {'en': Tensor(D), 'bn': Tensor(D)}}
    """
    print(f'Pre-computing report embeddings for {len(reports)} patients...')
    embeddings = {}
    for pid, r in reports.items():
        embeddings[pid] = {
            'en': encode_text([r['report_en']], tokenizer, text_encoder_base, device)[0],
            'bn': encode_text([r['report_bn']], tokenizer, text_encoder_base, device)[0],
        }
    print(f'Done. {len(embeddings)} patients embedded.')
    return embeddings
