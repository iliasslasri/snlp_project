"""Evaluate UED across different quantizer vocabulary sizes (50/100/200/500).

Reproduces the quantizer-ablation table from the paper (Table 3 or equivalent),
where each column corresponds to a checkpoint trained with a different vocab size.

Usage
-----
# With separate checkpoints per vocab size:
uv run python scripts/ablation_report.py \
    --checkpoints \
        checkpoints/quantizer/E1_50.pt \
        checkpoints/quantizer/E1_100.pt \
        checkpoints/quantizer/E1_200.pt \
        checkpoints/quantizer/E1_500.pt \
    --vocab-sizes 50 100 200 500 \
    --data-root LibriSpeech \
    --split test-clean \
    --noise-dir noise_fullband \
    --n-samples 500 \
    --markdown

# With a single checkpoint (vocab size inferred from --vocab-sizes):
uv run python scripts/ablation_report.py \
    --checkpoints checkpoints/quantizer/E1_best.pt \
    --vocab-sizes 500 \
    --data-root LibriSpeech --split test-clean
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
from src.evaluation import UEDEvaluator, _build_single_aug_config
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
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UED ablation across quantizer vocabulary sizes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="One checkpoint per vocab size, in the same order as --vocab-sizes.",
    )
    p.add_argument(
        "--vocab-sizes",
        nargs="+",
        type=int,
        required=True,
        help="Vocabulary sizes matching each checkpoint (e.g. 50 100 200 500).",
    )
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--hubert-name", default="hubert-base-ls960")
    p.add_argument("--hubert-layer", type=int, default=9)
    p.add_argument("--data-root", default="LibriSpeech")
    p.add_argument("--split", default="test-clean")
    p.add_argument("--noise-dir", default=None)
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--markdown", action="store_true", help="Print Markdown table.")
    p.add_argument("--output-json", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


def run_ued_for_vocab(
    checkpoint_path: str,
    vocab_size: int,
    E0: SpeechEncoder,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Dict]:
    """Evaluate UED for all augmentations for one (checkpoint, vocab_size) pair."""
    label = f"K={vocab_size}"
    print(f"\n[{label}]  {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    E1 = RobustQuantizer(
        input_dim=768,
        hidden_dim=args.hidden_dim,
        num_codes=vocab_size + 1,
    ).to(device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    E1.load_state_dict(state_dict)
    E1.eval()

    evaluator = UEDEvaluator(
        upstream_encoder=E0.dense,
        e1_model=E1,
        device=device,
        vocab_size=vocab_size,
    )

    n_samples: Optional[int] = args.n_samples if args.n_samples > 0 else None
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
                n_samples=n_samples,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            row[aug_name] = metrics
            print(
                f"  {AUG_DISPLAY[aug_name]:<16} "
                f"UED = {metrics['mean_ued']:.2f}% ± {metrics['std_ued']:.2f}%"
            )
        except Exception as exc:
            logger.warning(f"  {aug_name} failed: {exc}")
            row[aug_name] = {"mean_ued": float("nan"), "std_ued": float("nan")}

    return row


def print_ascii_table(
    table: Dict[str, Dict[str, Dict]],
    augmentations: List[str],
    col_labels: List[str],
) -> None:
    col_w = max(max(len(c) for c in col_labels) + 2, 14)
    aug_w = max(len(AUG_DISPLAY[a]) for a in augmentations) + 2

    sub_hdr = f"{'Augmentation':<{aug_w}}" + "".join(f"{c:>{col_w}}" for c in col_labels)
    sep = "-" * len(sub_hdr)

    print()
    print("  UED Ablation — Vocabulary Size  (lower is better)")
    print()
    print(f"  {sub_hdr}")
    print(f"  {sep}")

    for aug_key in augmentations:
        disp = AUG_DISPLAY[aug_key]
        row_str = f"{disp:<{aug_w}}"
        for label in col_labels:
            data = table.get(label, {}).get(aug_key, {})
            val = data.get("mean_ued", float("nan"))
            sem = data.get("sem_ued", float("nan"))
            cell = f"{val:.2f}±{sem:.2f}" if val == val else "N/A"
            row_str += f"{cell:>{col_w}}"
        print(f"  {row_str}")

    print(f"  {sep}")
    print()


def print_markdown_table(
    table: Dict[str, Dict[str, Dict]],
    augmentations: List[str],
    col_labels: List[str],
) -> None:
    header_cols = ["Augmentation"] + col_labels
    print()
    print("| " + " | ".join(header_cols) + " |")
    print("| " + " | ".join(["---"] * len(header_cols)) + " |")
    for aug_key in augmentations:
        disp = AUG_DISPLAY[aug_key]
        cells = [disp]
        for label in col_labels:
            data = table.get(label, {}).get(aug_key, {})
            val = data.get("mean_ued", float("nan"))
            sem = data.get("sem_ued", float("nan"))
            cells.append(f"{val:.2f}±{sem:.2f}" if val == val else "N/A")
        print("| " + " | ".join(cells) + " |")
    print()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")

    if len(args.checkpoints) != len(args.vocab_sizes):
        print("ERROR: --checkpoints and --vocab-sizes must have the same number of entries.")
        sys.exit(1)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.noise_dir is None:
        print(
            "WARNING: --noise-dir not set. 'noise' augmentation will use Gaussian "
            "white noise instead of DNS non-stationary noise. "
            "Results will NOT match the paper's 'noise' row."
        )

    # Load E0 once with the largest vocab size (KMeans centroids differ per size,
    # but the HuBERT encoder is the same for all).
    # Each evaluator gets its own E1 but shares the same E0.dense feature extractor.
    print(f"Loading HuBERT '{args.hubert_name}' layer {args.hubert_layer} …")
    E0 = SpeechEncoder.from_textlesslib(
        name=args.hubert_name,
        layer=args.hubert_layer,
        vocab_size=args.vocab_sizes[0],  # KMeans centroids for the first entry; overridden per-run
        deduplicate=True,
        kind_kmeans="kmeans",
    ).to(device)
    E0.eval()

    col_labels = [f"K={v}" for v in args.vocab_sizes]
    table: Dict[str, Dict] = {}

    for ckpt_path, vocab_size in zip(args.checkpoints, args.vocab_sizes):
        label = f"K={vocab_size}"
        table[label] = run_ued_for_vocab(ckpt_path, vocab_size, E0, args, device)

    if args.markdown:
        print_markdown_table(table, AUGMENTATIONS, col_labels)
    else:
        print_ascii_table(table, AUGMENTATIONS, col_labels)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(table, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
