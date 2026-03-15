"""Generate a per-augmentation UED comparison table for a trained E1 checkpoint.

This script evaluates the Unit Edit Distance (UED) for each of the four main
augmentation types (Time Stretch, Pitch Shift, Reverberation, Noise) and prints
a formatted Markdown / ASCII table suitable for inclusion in a paper or report.

Usage
-----
uv run python scripts/augmentation_report.py \\
    --checkpoint checkpoints/quantizer/round_0/E1_best.pt \\
    --data-root data/LibriSpeech \\
    --split test-clean \\
    --noise-dir noise_fullband \\
    --n-samples 500

Optional: compare multiple checkpoints (e.g. different training rounds)
uv run python scripts/augmentation_report.py \\
    --checkpoints \\
        checkpoints/quantizer/round_0/E1_best.pt \\
        checkpoints/quantizer/round_1/E1_best.pt \\
        checkpoints/quantizer/round_2/E1_best.pt \\
    --labels "E1 (round 0)" "E1 (round 1)" "E1 (round 2)" \\
    --data-root data/LibriSpeech \\
    --split test-clean \\
    --noise-dir noise_fullband

Optional: also evaluate a baseline (E0, KMeans)
    Add --include-baseline to include the E0 (HuBERT + KMeans) row.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "speech_encoder", "src"))

from omegaconf import OmegaConf

from speech_encoder import SpeechEncoder
from src.dataset import AudioDataset
from src.evaluation import UEDEvaluator, _build_single_aug_config, _paired_collate_fn
from src.metrics import batch_unit_edit_distance
from src.models import RobustQuantizer

logger = logging.getLogger(__name__)

AUGMENTATIONS = ["time_stretch", "pitch_shift", "reverberation", "noise"]
AUG_DISPLAY = {
    "time_stretch": "Time Stretch",
    "pitch_shift": "Pitch Shift",
    "reverberation": "Reverberation",
    "noise": "Noise",
}

FULL_AUG_CONFIG = OmegaConf.create(
    {
        "time_stretch": True,
        "pitch_shift": True,
        "reverberation": True,
        "noise": True,
        "echo": {"enabled": True, "volume_range": [0.1, 0.5], "duration_range": [0.1, 0.5]},
        "random_noise": {"enabled": True, "noise_std": 0.001},
        "pink_noise": {"enabled": True, "noise_std": 0.01},
        "lowpass_filter": {"enabled": True, "cutoff_freq": 5000},
        "highpass_filter": {"enabled": True, "cutoff_freq": 500},
        "bandpass_filter": {"enabled": True, "cutoff_freq_low": 300, "cutoff_freq_high": 8000},
        "smooth": {"enabled": True, "window_size_range": [2, 10]},
        "boost_audio": {"enabled": True, "amount": 20},
        "duck_audio": {"enabled": True, "amount": 20},
        "updownresample": {"enabled": True, "intermediate_freq": 32000},
    }
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-augmentation UED report for a trained E1 checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Single or multi-checkpoint mode
    ckpt_group = p.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a single E1 checkpoint (.pt).",
    )
    ckpt_group.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="Paths to multiple E1 checkpoints for side-by-side comparison.",
    )

    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Display labels for --checkpoints (must match the number of checkpoints).",
    )

    p.add_argument(
        "--include-baseline",
        action="store_true",
        help="Also evaluate the E0 baseline (HuBERT + KMeans, no E1).",
    )

    # Model hyper-parameters
    p.add_argument("--vocab-size", type=int, default=500)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--hubert-name", default="hubert-base-ls960")
    p.add_argument("--hubert-layer", type=int, default=9)

    # Data
    p.add_argument("--data-root", default="data/LibriSpeech")
    p.add_argument("--split", default="test-clean")
    p.add_argument("--noise-dir", default=None)
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)

    # Output
    p.add_argument("--output-json", default=None, help="Save raw results to this JSON file.")
    p.add_argument("--markdown", action="store_true", help="Print table in Markdown format.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING"])

    return p.parse_args()


# ---------------------------------------------------------------------------
# Baseline UED (E0: KMeans)
# ---------------------------------------------------------------------------


class BaselineUEDEvaluator:
    """Computes UED using E0 (frozen HuBERT + KMeans) as both clean and aug encoder.

    For the baseline, both clean and augmented audio go through E0.
    This measures how robust the original KMeans units are to augmentation.
    """

    def __init__(self, e0: SpeechEncoder, device: torch.device):
        self.e0 = e0
        self.device = device

    @torch.no_grad()
    def encode(self, waveforms: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> List[List[int]]:
        outputs = self.e0(waveforms, lengths=lengths, formatted=True)
        return [o["units"] for o in outputs]

    def evaluate_augmentation(
        self,
        root: str,
        split: str,
        augmentation_name: str,
        noise_dir: Optional[str] = None,
        max_length: Optional[int] = None,
        n_samples: Optional[int] = None,
        batch_size: int = 8,
        num_workers: int = 0,
    ) -> Dict[str, float]:
        from torch.utils.data import DataLoader, Subset

        aug_config = _build_single_aug_config(augmentation_name, FULL_AUG_CONFIG)
        dataset = AudioDataset(
            root=root,
            split=split,
            augment=True,
            config=aug_config,
            noise_dir=noise_dir,
            max_length=max_length,
        )
        if n_samples is not None and n_samples < len(dataset):
            indices = torch.randperm(len(dataset))[:n_samples].tolist()
            dataset = Subset(dataset, indices)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=_paired_collate_fn,
            num_workers=num_workers,
            shuffle=False,
        )

        self.e0.eval()
        all_ueds: List[float] = []

        for clean_audio, clean_lens, aug_audio, aug_lens in loader:
            clean_audio = clean_audio.to(self.device)
            clean_lens = clean_lens.to(self.device)
            aug_audio = aug_audio.to(self.device)
            aug_lens = aug_lens.to(self.device)

            clean_units = self.encode(clean_audio, clean_lens)
            aug_units = self.encode(aug_audio, aug_lens)
            _, scores = batch_unit_edit_distance(clean_units, aug_units)
            all_ueds.extend(scores)

        return {
            "mean_ued": float(np.mean(all_ueds)),
            "std_ued": float(np.std(all_ueds)),
        }


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def run_ued_for_checkpoint(
    checkpoint_path: str,
    label: str,
    E0: SpeechEncoder,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Dict]:
    """Evaluate UED for all augmentations for a single checkpoint."""
    logger.info(f"Loading E1 from {checkpoint_path} …")
    ckpt = torch.load(checkpoint_path, map_location=device)
    E1 = RobustQuantizer(
        input_dim=768,
        hidden_dim=args.hidden_dim,
        num_codes=args.vocab_size + 1,
    ).to(device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    E1.load_state_dict(state_dict)
    E1.eval()

    evaluator = UEDEvaluator(
        upstream_encoder=E0.dense,
        e1_model=E1,
        device=device,
        vocab_size=args.vocab_size,
    )

    row: Dict[str, Dict] = {}
    for aug_name in AUGMENTATIONS:
        try:
            metrics = evaluator.evaluate_augmentation(
                root=args.data_root,
                split=args.split,
                augmentation_name=aug_name,
                base_aug_config=FULL_AUG_CONFIG,
                noise_dir=args.noise_dir,
                max_length=160_000,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            row[aug_name] = metrics
            print(
                f"  [{label}] {AUG_DISPLAY[aug_name]:<16} "
                f"UED = {metrics['mean_ued']:.2f}% ± {metrics['std_ued']:.2f}%"
            )
        except Exception as exc:
            logger.warning(f"  [{label}] {aug_name} failed: {exc}")
            row[aug_name] = {"mean_ued": float("nan"), "std_ued": float("nan")}

    return row


# ---------------------------------------------------------------------------
# Table printers
# ---------------------------------------------------------------------------


def print_ascii_table(table: Dict[str, Dict[str, Dict]], augmentations: List[str]) -> None:
    """Print a plain-ASCII comparison table."""
    models = list(table.keys())
    col_w = max(len(m) for m in models) + 2
    aug_w = max(len(AUG_DISPLAY[a]) for a in augmentations) + 2

    header = f"{'Augmentation':<{aug_w}}" + "".join(f"{'Mean UED (%)':>{col_w}}" for _ in models)
    sub_hdr = f"{'':<{aug_w}}" + "".join(f"{m:>{col_w}}" for m in models)
    sep = "-" * len(header)

    print()
    print("  UED Comparison Table (lower is better)")
    print()
    print(f"  {sub_hdr}")
    print(f"  {sep}")

    for aug_key in augmentations:
        disp = AUG_DISPLAY.get(aug_key, aug_key)
        row_str = f"{disp:<{aug_w}}"
        for model in models:
            data = table[model].get(aug_key, {})
            val = data.get("mean_ued", float("nan"))
            sem = data.get("sem_ued", float("nan"))
            cell = f"{val:.2f}±{sem:.2f}" if not (val != val) else "N/A"
            row_str += f"{cell:>{col_w}}"
        print(f"  {row_str}")

    print(f"  {sep}")
    print()


def print_markdown_table(table: Dict[str, Dict[str, Dict]], augmentations: List[str]) -> None:
    """Print a Markdown-formatted comparison table."""
    models = list(table.keys())
    header_cols = ["Augmentation"] + [f"{m} (mean±sem)" for m in models]

    print()
    print("| " + " | ".join(header_cols) + " |")
    print("| " + " | ".join(["---"] * len(header_cols)) + " |")

    for aug_key in augmentations:
        disp = AUG_DISPLAY.get(aug_key, aug_key)
        cells = [disp]
        for model in models:
            data = table[model].get(aug_key, {})
            val = data.get("mean_ued", float("nan"))
            sem = data.get("sem_ued", float("nan"))
            cells.append(f"{val:.2f}±{sem:.2f}" if not (val != val) else "N/A")
        print("| " + " | ".join(cells) + " |")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.noise_dir is None:
        print(
            "WARNING: --noise-dir not set. The 'noise' augmentation will use Gaussian "
            "white noise (stationary) instead of DNS non-stationary noise. "
            "Results will NOT match the paper's 'noise' row."
        )

    # Resolve checkpoint list
    if args.checkpoint:
        checkpoints = [args.checkpoint]
        labels = ["E1"]
    else:
        checkpoints = args.checkpoints
        labels = args.labels or [f"E1 (ckpt {i})" for i in range(len(checkpoints))]
        if len(labels) != len(checkpoints):
            print("ERROR: --labels must have the same number of entries as --checkpoints.")
            sys.exit(1)

    # Load E0 once (shared)
    print(f"Loading HuBERT '{args.hubert_name}' layer {args.hubert_layer} …")
    E0 = SpeechEncoder.from_textlesslib(
        name=args.hubert_name,
        layer=args.hubert_layer,
        vocab_size=args.vocab_size,
        deduplicate=True,
        kind_kmeans="kmeans",
    ).to(device)
    E0.eval()

    # Evaluate
    table: Dict[str, Dict] = {}

    if args.include_baseline:
        print("\n[Baseline: E0 (HuBERT + KMeans)]")
        baseline_eval = BaselineUEDEvaluator(E0, device)
        baseline_row: Dict = {}
        for aug_name in AUGMENTATIONS:
            try:
                metrics = baseline_eval.evaluate_augmentation(
                    root=args.data_root,
                    split=args.split,
                    augmentation_name=aug_name,
                    noise_dir=args.noise_dir,
                    max_length=160_000,
                    n_samples=args.n_samples,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
                baseline_row[aug_name] = metrics
                print(
                    f"  [Baseline] {AUG_DISPLAY[aug_name]:<16} "
                    f"UED = {metrics['mean_ued']:.2f}% ± {metrics['std_ued']:.2f}%"
                )
            except Exception as exc:
                logger.warning(f"  [Baseline] {aug_name} failed: {exc}")
                baseline_row[aug_name] = {"mean_ued": float("nan"), "std_ued": float("nan")}
        table["Baseline (E0)"] = baseline_row

    for ckpt_path, label in zip(checkpoints, labels):
        print(f"\n[{label}]  {ckpt_path}")
        row = run_ued_for_checkpoint(ckpt_path, label, E0, args, device)
        table[label] = row

    # Print table
    if args.markdown:
        print_markdown_table(table, AUGMENTATIONS)
    else:
        print_ascii_table(table, AUGMENTATIONS)

    # Save JSON
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(table, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
