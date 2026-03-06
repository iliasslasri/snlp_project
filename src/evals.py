"""
eval.py — Evaluate robustness of speech quantizers using UED.

Supports two modes:
  1. Baseline : HuBERT + k-means (E0) — the standard GSLM quantizer
  2. Robust   : HuBERT + trained RobustQuantizer (E1, E2, ...) checkpoint

For each model, UED is computed on all four augmentations from the paper
(Section 3.2.1): time_stretch, pitch_shift, reverberation, noise.

Usage:
  # Evaluate k-means baseline
  python eval.py

  # Evaluate a trained RobustQuantizer checkpoint
  python eval.py eval.checkpoint=checkpoints/quantizer/E1_best.pt

  # Evaluate on a specific augmentation only
  python eval.py eval.augmentation=time_stretch
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'speech_encoder', 'src'))

from speech_encoder import SpeechEncoder
from src.models import RobustQuantizer
from src.dataset import AudioDataset
from src.metrics import unit_edit_distance
from src.utils import set_seed, setup_logging
from src.robust_quantization import collate_fn_paired

# Augmentations to evaluate — matches Section 3.2.1 of the paper
ALL_AUGMENTATIONS = ["time_stretch", "pitch_shift", "reverberation", "noise"]


def load_baseline(cfg, device):
    """Load E0: HuBERT + k-means (frozen)."""
    logging.info(f"Loading baseline SpeechEncoder '{cfg.model.name}'...")
    encoder = SpeechEncoder.from_textlesslib(
        name=cfg.model.name,
        layer=cfg.model.layer,
        vocab_size=cfg.model.vocab_size,
        deduplicate=True,
        kind_kmeans=cfg.model.kind_kmeans,
    ).to(device)
    encoder.eval()
    return encoder


def load_robust_quantizer(cfg, checkpoint_path, device):
    """Load a trained RobustQuantizer from a checkpoint."""
    logging.info(f"Loading RobustQuantizer from '{checkpoint_path}'...")
    model = RobustQuantizer(
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.quantizer.hidden_dim,
        num_codes=cfg.model.vocab_size + 1,  # +1 for CTC blank
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def evaluate_baseline(E0, dataloader, augmentation, device):
    """Compute UED for the k-means baseline (E0) on one augmentation type.

    Both clean and augmented audio go through E0 (HuBERT + k-means).
    UED measures how much the discrete units change under augmentation.
    """
    orig_units_all = []
    aug_units_all  = []

    for clean_audio, clean_lens, aug_audio, aug_lens in dataloader:
        clean_audio = clean_audio.to(device)
        clean_lens  = clean_lens.to(device)
        aug_audio   = aug_audio.to(device)
        aug_lens    = aug_lens.to(device)

        # Units from clean audio
        clean_outputs = E0(clean_audio, lengths=clean_lens, formatted=True)
        for out in clean_outputs:
            orig_units_all.append(out['units'])

        # Units from augmented audio
        aug_outputs = E0(aug_audio, lengths=aug_lens, formatted=True)
        for out in aug_outputs:
            aug_units_all.append(out['units'])

    ued = unit_edit_distance(orig_units_all, aug_units_all)
    logging.info(f"[Baseline k-means | {augmentation}] UED = {ued:.2f}")
    return ued


@torch.no_grad()
def evaluate_robust_quantizer(E0, E_student, upstream_encoder, dataloader,
                               augmentation, device):
    """Compute UED for a trained RobustQuantizer on one augmentation type.

    Clean audio → E0 (k-means) → reference units
    Augmented audio → HuBERT → E_student (MLP) → argmax → predicted units
    """
    orig_units_all = []
    aug_units_all  = []

    for clean_audio, clean_lens, aug_audio, aug_lens in dataloader:
        clean_audio = clean_audio.to(device)
        clean_lens  = clean_lens.to(device)
        aug_audio   = aug_audio.to(device)
        aug_lens    = aug_lens.to(device)

        # Reference units from clean audio via E0
        clean_outputs = E0(clean_audio, lengths=clean_lens, formatted=True)
        for out in clean_outputs:
            orig_units_all.append(out['units'])

        # Predicted units from augmented audio via E_student
        aug_feats, _ = upstream_encoder(aug_audio, lengths=aug_lens)
        logits = E_student(aug_feats)          # [B, T, vocab+1]
        preds  = logits.argmax(dim=-1)         # [B, T]
        for i in range(preds.shape[0]):
            aug_units_all.append(preds[i].cpu().tolist())

    ued = unit_edit_distance(orig_units_all, aug_units_all)
    logging.info(f"[RobustQuantizer | {augmentation}] UED = {ued:.2f}")
    return ued


def build_dataloader(cfg, augmentation):
    """Build a dataloader with only one augmentation active at a time."""
    aug_config = {aug: (aug == augmentation) for aug in ALL_AUGMENTATIONS}
    dataset = AudioDataset(
        root=cfg.dataset.root,
        split=cfg.dataset.get("eval_split", cfg.dataset.valid_split),
        augment=True,
        config=aug_config,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        collate_fn=collate_fn_paired,
        num_workers=0,
    )


@hydra.main(version_base=None, config_path="configs", config_name="quantization")
def main(cfg: DictConfig):
    if cfg.get("seed"):
        set_seed(cfg.seed)

    setup_logging(cfg.training.checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Which augmentations to evaluate
    aug_to_eval = ALL_AUGMENTATIONS
    if cfg.get("eval") and cfg.eval.get("augmentation"):
        aug_to_eval = [cfg.eval.augmentation]

    # Load E0 (always needed as reference)
    E0 = load_baseline(cfg, device)
    upstream_encoder = E0.dense
    upstream_encoder.eval()

    results = {}

    checkpoint = cfg.get("eval", {}).get("checkpoint", None) if cfg.get("eval") else None

    if checkpoint:
        # --- Evaluate RobustQuantizer ---
        E_student = load_robust_quantizer(cfg, checkpoint, device)
        for aug in aug_to_eval:
            dataloader = build_dataloader(cfg, aug)
            ued = evaluate_robust_quantizer(
                E0, E_student, upstream_encoder, dataloader, aug, device
            )
            results[aug] = ued
        label = "RobustQuantizer"
    else:
        # --- Evaluate k-means baseline ---
        for aug in aug_to_eval:
            dataloader = build_dataloader(cfg, aug)
            ued = evaluate_baseline(E0, dataloader, aug, device)
            results[aug] = ued
        label = "Baseline (k-means)"

    # --- Summary ---
    logging.info(f"\n{'='*50}")
    logging.info(f"Results — {label} | vocab_size={cfg.model.vocab_size}")
    logging.info(f"{'='*50}")
    for aug, ued in results.items():
        logging.info(f"  {aug:<20} UED = {ued:.2f}")
    logging.info(f"  {'mean':<20} UED = {sum(results.values()) / len(results):.2f}")
    logging.info(f"{'='*50}\n")


if __name__ == "__main__":
    main()