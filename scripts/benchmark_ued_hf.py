#!/usr/bin/env python3
"""
UED Benchmark across all HuggingFace checkpoints.

Downloads every E1_best.pt from iliasslasri/robust_speech_quantizer, then runs
UED (+ optional E0 baseline) on test-clean and test-other for each model type
(100 / 200 / 500 vocab). One benchmark table is printed per vocab size.

Result layout mirrors the HuggingFace repo:
    results/hf_benchmark/
    ├── 100_vocab_size/
    │   ├── _baseline/
    │   │   ├── ued_test-clean.json
    │   │   └── ued_test-other.json
    │   ├── round_0/
    │   │   ├── ued_test-clean.json
    │   │   └── ued_test-other.json
    │   ├── round_1/ ...
    │   └── summary.json          ← aggregated table for this vocab size
    ├── 200_vocab_size/ ...
    └── 500_vocab_size/ ...

Usage (Windows):
    uv run python scripts/benchmark_ued_hf.py ^
        --data-root LibriSpeech ^
        --noise-dir noise_fullband ^
        --n-samples 0

Options:
    --data-root     Path to LibriSpeech root (default: LibriSpeech)
    --noise-dir     DNS noise directory for the "noise" augmentation
    --n-samples     Samples per aug (0 = full split, default: 0)
    --splits        Splits to evaluate (default: test-clean test-other)
    --batch-size    Batch size for evaluation (default: 8)
    --hubert-layer  HuBERT layer to use (default: 6)
    --output-dir    Where to save results (default: results/hf_benchmark)
    --hf-cache-dir  Where to cache downloaded checkpoints (default: checkpoints/hf_cache)
    --skip-baseline Skip E0 baseline evaluation (saves time)
    --skip-existing Skip evaluations whose result JSON already exists
    --vocab-sizes   Which vocab sizes to benchmark (default: 100 200 500)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

# ── project imports ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "speech_encoder" / "src"))

from omegaconf import OmegaConf
from speech_encoder import SpeechEncoder
from src.evaluation import BaselineUEDEvaluator, UEDEvaluator
from src.models import RobustQuantizer

logger = logging.getLogger(__name__)

# ── HuggingFace repo ───────────────────────────────────────────────────────────
HF_REPO_ID = "iliasslasri/robust_speech_quantizer"


def _discover_hf_structure() -> dict[int, list[int]]:
    """
    Auto-discover vocab sizes and rounds from the HuggingFace repo.

    Scans for files matching ``{K}_vocab_size/round_{N}/E1_best.pt`` and
    returns {vocab_size: sorted_list_of_round_indices}.
    """
    import re
    from huggingface_hub import list_repo_files

    pattern = re.compile(r"^(\d+)_vocab_size/round_(\d+)/E1_best\.pt$")
    structure: dict[int, list[int]] = {}
    for filepath in list_repo_files(HF_REPO_ID, repo_type="model"):
        m = pattern.match(filepath)
        if m:
            vocab = int(m.group(1))
            rnd = int(m.group(2))
            structure.setdefault(vocab, []).append(rnd)
    # sort round lists
    return {v: sorted(r) for v, r in sorted(structure.items())}

PAPER_AUGMENTATIONS = ["time_stretch", "pitch_shift", "reverberation", "noise"]

AUG_DISPLAY = {
    "time_stretch":  "Time Stretch",
    "pitch_shift":   "Pitch Shift",
    "reverberation": "Reverberation",
    "noise":         "Noise",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _build_aug_config() -> object:
    return OmegaConf.create(
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


def _download_checkpoint(vocab_size: int, round_idx: int, cache_dir: Path) -> Path:
    """Download E1_best.pt for a given vocab_size / round from HuggingFace."""
    from huggingface_hub import hf_hub_download

    hf_path = f"{vocab_size}_vocab_size/round_{round_idx}/E1_best.pt"
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=hf_path,
        cache_dir=str(cache_dir),
        repo_type="model",
    )
    return Path(local_path)


def _load_hubert(hubert_name: str, hubert_layer: int, vocab_size: int, device: torch.device) -> SpeechEncoder:
    logger.info(f"Loading HuBERT '{hubert_name}' layer {hubert_layer} …")
    E0 = SpeechEncoder.from_textlesslib(
        name=hubert_name,
        layer=hubert_layer,
        vocab_size=vocab_size,
        deduplicate=True,
        kind_kmeans="kmeans",
    ).to(device)
    E0.eval()
    return E0


def _load_e1(checkpoint: Path, vocab_size: int, hidden_dim: int, device: torch.device) -> RobustQuantizer:
    logger.info(f"Loading E1 from {checkpoint} …")
    ckpt = torch.load(str(checkpoint), map_location=device)
    E1 = RobustQuantizer(
        input_dim=768,
        hidden_dim=hidden_dim,
        num_codes=vocab_size + 1,  # +1 for CTC blank
    ).to(device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    E1.load_state_dict(state_dict)
    E1.eval()
    return E1


def _run_ued_for_model(
    evaluator: UEDEvaluator | BaselineUEDEvaluator,
    data_root: str,
    split: str,
    noise_dir: str | None,
    n_samples: int | None,
    batch_size: int,
) -> dict:
    """Run UED for all 4 paper augmentations; returns dict keyed by aug name."""
    base_cfg = _build_aug_config()
    results: dict = {}
    for aug_name in PAPER_AUGMENTATIONS:
        logger.info(f"    [{aug_name}] …")
        try:
            metrics = evaluator.evaluate_augmentation(
                root=data_root,
                split=split,
                augmentation_name=aug_name,
                base_aug_config=base_cfg,
                noise_dir=noise_dir,
                max_length=160_000,
                n_samples=n_samples,
                batch_size=batch_size,
                num_workers=0,
            )
            results[aug_name] = metrics
            logger.info(
                f"      mean={metrics['mean_ued']:.2f}%  std={metrics['std_ued']:.2f}%  n={metrics['n']}"
            )
        except Exception as exc:
            logger.warning(f"      FAILED: {exc}")
            results[aug_name] = {"error": str(exc)}
    return results


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"  Saved → {path}")


def _print_table(vocab_size: int, splits: list[str], summary: dict) -> None:
    """Print a benchmark table for one vocab size.

    Each column shows  ``mean ± sem``  (both in %).
    """
    cell_width = 16   # "XX.XX ± XX.XX" fits in 15 chars + 1 padding
    aug_width  = 16
    print()
    print("=" * (aug_width + (cell_width + 2) * 6 + 4))
    print(f"  UED Benchmark — {vocab_size}_vocab_size  [lower is better, mean ± sem %]")
    print("=" * (aug_width + (cell_width + 2) * 6 + 4))

    for split in splits:
        # Collect columns: baseline + rounds (sorted)
        col_keys: list[str] = []
        if "_baseline" in summary and split in summary["_baseline"]:
            col_keys.append("_baseline")
        round_keys = sorted(
            [k for k in summary if k.startswith("round_") and split in summary[k]],
            key=lambda x: int(x.split("_")[1]),
        )
        col_keys.extend(round_keys)

        if not col_keys:
            continue

        sep_width = aug_width + (cell_width + 2) * len(col_keys) + 2

        # Header
        col_labels = [
            "Baseline" if k == "_baseline" else k.replace("_", " ").title()
            for k in col_keys
        ]
        header = f"  {'Augmentation':<{aug_width}}"
        for lbl in col_labels:
            header += f"  {lbl:^{cell_width}}"
        print(f"\n  Split: {split}\n")
        print(header)
        print("  " + "-" * sep_width)

        for aug_name in PAPER_AUGMENTATIONS:
            disp = AUG_DISPLAY[aug_name]
            row = f"  {disp:<{aug_width}}"
            for col_key in col_keys:
                entry = summary.get(col_key, {}).get(split, {}).get(aug_name, {})
                if "mean_ued" in entry and "sem_ued" in entry:
                    cell = f"{entry['mean_ued']:.2f} ± {entry['sem_ued']:.2f}"
                    row += f"  {cell:^{cell_width}}"
                elif "error" in entry:
                    row += f"  {'ERROR':^{cell_width}}"
                else:
                    row += f"  {'—':^{cell_width}}"
            print(row)

    print()


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UED benchmark across all HuggingFace checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default="LibriSpeech", help="LibriSpeech root directory.")
    p.add_argument("--noise-dir", default=None, help="DNS noise dir (for 'noise' aug).")
    p.add_argument(
        "--n-samples", type=int, default=0,
        help="Samples per augmentation (0 = full split).",
    )
    p.add_argument(
        "--splits", nargs="+", default=["test-clean", "test-other"],
        metavar="SPLIT",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hubert-name", default="hubert-base-ls960")
    p.add_argument("--hubert-layer", type=int, default=6)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument(
        "--vocab-sizes", type=int, nargs="+", default=[100, 200, 500],
        metavar="K",
    )
    p.add_argument("--output-dir", default="results/hf_benchmark")
    p.add_argument(
        "--hf-cache-dir", default="checkpoints/hf_cache",
        help="Local directory for caching downloaded HF checkpoints.",
    )
    p.add_argument("--skip-baseline", action="store_true", help="Skip E0 baseline evaluation.")
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip any evaluation whose result JSON already exists.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.hf_cache_dir)
    n_samples = args.n_samples if args.n_samples > 0 else None

    if args.noise_dir is None:
        logger.warning(
            "WARNING: --noise-dir not set. 'noise' augmentation will use Gaussian "
            "white noise instead of DNS non-stationary noise. Results will NOT match "
            "the paper for this augmentation."
        )

    # ── discover repo structure ────────────────────────────────────────────────
    logger.info("Discovering checkpoints on HuggingFace …")
    hf_structure = _discover_hf_structure()
    logger.info(
        "  Found: "
        + ", ".join(f"{k}_vocab_size → rounds {v}" for k, v in hf_structure.items())
    )

    # Filter to requested vocab sizes
    requested = set(args.vocab_sizes)
    vocab_sizes_to_run = [v for v in sorted(hf_structure) if v in requested]
    if not vocab_sizes_to_run:
        logger.error(
            f"None of the requested vocab sizes {args.vocab_sizes} were found in the repo. "
            f"Available: {list(hf_structure.keys())}"
        )
        return

    # ── outer loop: one bench per vocab size ──────────────────────────────────
    for vocab_size in vocab_sizes_to_run:
        rounds = hf_structure[vocab_size]
        vocab_dir = output_dir / f"{vocab_size}_vocab_size"
        summary: dict = {}  # round_key → {split → {aug → metrics}}

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Benchmarking vocab_size={vocab_size}  rounds={rounds}")
        logger.info("=" * 60)

        # Load HuBERT once per vocab size (shared across rounds & baseline)
        E0 = _load_hubert(args.hubert_name, args.hubert_layer, vocab_size, device)

        # ── baseline (E0) ────────────────────────────────────────────────────
        if not args.skip_baseline:
            baseline_evaluator = BaselineUEDEvaluator(E0, device)
            summary["_baseline"] = {}
            for split in args.splits:
                result_path = vocab_dir / "_baseline" / f"ued_{split}.json"
                if args.skip_existing and result_path.exists():
                    logger.info(f"  [baseline/{split}] Skipping (already exists).")
                    with open(result_path) as f:
                        summary["_baseline"][split] = json.load(f)
                    continue

                logger.info(f"  Evaluating baseline on {split} …")
                res = _run_ued_for_model(
                    baseline_evaluator, args.data_root, split,
                    args.noise_dir, n_samples, args.batch_size,
                )
                summary["_baseline"][split] = res
                _save_json(result_path, res)

        # ── E1 rounds ────────────────────────────────────────────────────────
        for round_idx in rounds:
            round_key = f"round_{round_idx}"
            summary[round_key] = {}

            # Check if we need to download or evaluate anything for this round
            all_exist = all(
                (vocab_dir / round_key / f"ued_{split}.json").exists()
                for split in args.splits
            )
            if args.skip_existing and all_exist:
                logger.info(f"  [{round_key}] All splits already evaluated, loading from disk.")
                for split in args.splits:
                    result_path = vocab_dir / round_key / f"ued_{split}.json"
                    with open(result_path) as f:
                        summary[round_key][split] = json.load(f)
                continue

            # Download checkpoint
            logger.info(f"  [{round_key}] Downloading checkpoint from HuggingFace …")
            try:
                ckpt_path = _download_checkpoint(vocab_size, round_idx, cache_dir)
                logger.info(f"  [{round_key}] Checkpoint: {ckpt_path}")
            except Exception as exc:
                logger.error(f"  [{round_key}] Download failed: {exc}")
                for split in args.splits:
                    summary[round_key][split] = {"error": f"download_failed: {exc}"}
                continue

            # Load E1
            try:
                E1 = _load_e1(ckpt_path, vocab_size, args.hidden_dim, device)
            except Exception as exc:
                logger.error(f"  [{round_key}] Failed to load E1: {exc}")
                for split in args.splits:
                    summary[round_key][split] = {"error": f"load_failed: {exc}"}
                continue

            evaluator = UEDEvaluator(
                upstream_encoder=E0.dense,
                e1_model=E1,
                device=device,
                vocab_size=vocab_size,
            )

            for split in args.splits:
                result_path = vocab_dir / round_key / f"ued_{split}.json"
                if args.skip_existing and result_path.exists():
                    logger.info(f"  [{round_key}/{split}] Skipping (already exists).")
                    with open(result_path) as f:
                        summary[round_key][split] = json.load(f)
                    continue

                logger.info(f"  [{round_key}] Evaluating on {split} …")
                res = _run_ued_for_model(
                    evaluator, args.data_root, split,
                    args.noise_dir, n_samples, args.batch_size,
                )
                summary[round_key][split] = res
                _save_json(result_path, res)

            # Free GPU memory before loading next checkpoint
            del E1, evaluator
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ── save aggregated summary for this vocab size ───────────────────────
        summary_path = vocab_dir / "summary.json"
        _save_json(summary_path, summary)

        # ── print benchmark table ─────────────────────────────────────────────
        _print_table(vocab_size, args.splits, summary)

        # Free HuBERT before next vocab size (different KMeans)
        del E0
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    logger.info("Benchmark complete.")
    logger.info(f"Results saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
