"""Evaluate a trained E1 checkpoint on GSLM metrics.

Supported metrics (select with --metrics):
    ued     – Unit Edit Distance per augmentation (requires --checkpoint)
    abx     – Phoneme discrimination, within/across speaker (requires Python 3.12+
              and --abx-items; use --abx-extract to generate feature files on the fly)
    swuggy  – Lexical evaluation (requires --lm-checkpoint + --swuggy-data)
    sblim   – Syntactic evaluation (requires --lm-checkpoint + --sblim-data)

Each metric can be run independently. By default all enabled metrics run.

Examples
--------
# UED only (fastest, the core robustness metric)
uv run python evaluate.py \\
    --metrics ued \\
    --checkpoint checkpoints/quantizer/round_0/E1_best.pt \\
    --data-root data/LibriSpeech --split test-clean \\
    --noise-dir noise_fullband --n-samples 500

# ABX only (no E1 checkpoint needed, only HuBERT features)
uv run python evaluate.py \\
    --metrics abx \\
    --abx-items data/librispeech-clean.item \\
    --abx-features path/to/features     # pre-extracted .pt files

# ABX with on-the-fly feature extraction
uv run python evaluate.py \\
    --metrics abx \\
    --abx-items data/librispeech-clean.item \\
    --abx-extract --abx-audio-dir data/LibriSpeech/test-clean \\
    --abx-features outputs/features/test-clean

# UED + ABX together
uv run python evaluate.py \\
    --metrics ued abx \\
    --checkpoint checkpoints/quantizer/round_0/E1_best.pt \\
    --data-root data/LibriSpeech --split test-clean \\
    --abx-items data/librispeech-clean.item \\
    --abx-features outputs/features/test-clean

# Full evaluation (all metrics)
uv run python evaluate.py \\
    --checkpoint checkpoints/quantizer/round_0/E1_best.pt \\
    --data-root data/LibriSpeech --split test-clean \\
    --noise-dir noise_fullband --n-samples 500 \\
    --abx-items data/librispeech-clean.item \\
    --abx-features outputs/features/test-clean \\
    --lm-checkpoint checkpoints/lm/ngram.pkl \\
    --swuggy-data data/swuggy.json \\
    --sblim-data data/sblim.json \\
    --output results/eval_round0.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "speech_encoder", "src"))

from omegaconf import OmegaConf

from speech_encoder import SpeechEncoder
from src.evaluation import ABXEvaluator, BaselineUEDEvaluator, LexicalEvaluator, SyntacticEvaluator, UEDEvaluator
from src.metrics import NgramLM
from src.models import RobustQuantizer

logger = logging.getLogger(__name__)

PAPER_AUGMENTATIONS = ["time_stretch", "pitch_shift", "reverberation", "noise"]
ALL_METRICS = ["ued", "abx", "swuggy", "sblim"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained E1 checkpoint on GSLM metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Metric selection
    p.add_argument(
        "--metrics",
        nargs="+",
        choices=ALL_METRICS,
        default=ALL_METRICS,
        metavar="METRIC",
        help=(
            "Which metrics to run. Any subset of: ued abx swuggy sblim. "
            "Default: all. Example: --metrics ued abx"
        ),
    )

    # Model (only needed for ued; abx only uses HuBERT, swuggy/sblim use lm-checkpoint)
    model = p.add_argument_group("Model (required for --metrics ued)")
    model.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the E1 checkpoint (.pt). Required for UED.",
    )
    model.add_argument("--vocab-size", type=int, default=500)
    model.add_argument("--hidden-dim", type=int, default=256)
    model.add_argument("--hubert-name", default="hubert-base-ls960")
    model.add_argument("--hubert-layer", type=int, default=6)

    # Data (for ued)
    data = p.add_argument_group("Data (for --metrics ued)")
    data.add_argument("--data-root", default="data/LibriSpeech")
    data.add_argument(
        "--split",
        default="test-clean",
        help="Dataset split, e.g. test-clean or test-other.",
    )
    data.add_argument("--noise-dir", default=None, help="DNS noise directory (for 'noise' aug).")
    data.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Samples per augmentation. Set to 0 for the full split.",
    )
    data.add_argument("--batch-size", type=int, default=8)
    data.add_argument("--num-workers", type=int, default=0)
    data.add_argument(
        "--include-baseline",
        action="store_true",
        help="Also evaluate the E0 baseline (HuBERT + KMeans, no E1) and show it as a reference row.",
    )

    # ABX
    abx = p.add_argument_group("ABX options (for --metrics abx, requires Python 3.12+)")
    abx.add_argument(
        "--abx-items",
        default=None,
        help="ZeroSpeech-format .item file with phoneme alignments.",
    )
    abx.add_argument(
        "--abx-features",
        default=None,
        help="Directory with per-utterance .pt HuBERT feature tensors.",
    )
    abx.add_argument(
        "--abx-extract",
        action="store_true",
        help=(
            "Extract HuBERT features from --abx-audio-dir and save to --abx-features "
            "before scoring. Skipped if features already exist."
        ),
    )
    abx.add_argument(
        "--abx-audio-dir",
        default=None,
        help="Audio directory to extract features from (used with --abx-extract).",
    )
    abx.add_argument(
        "--abx-distance",
        default="angular",
        choices=["angular", "cosine", "euclidean"],
        help="Distance metric for ABX. 'angular' is the ZeroSpeech 2021 default.",
    )

    # LM-based metrics
    lm = p.add_argument_group("LM-based metrics (for --metrics swuggy sblim)")
    lm.add_argument(
        "--lm-checkpoint",
        default=None,
        help="NgramLM pickle produced by NgramLM.save(). Required for swuggy/sblim.",
    )
    lm.add_argument(
        "--swuggy-data",
        default=None,
        help="sWUGGY JSON: list of {\"real\": [...], \"pseudo\": [...]}.",
    )
    lm.add_argument(
        "--sblim-data",
        default=None,
        help="sBLIMP JSON: list of {\"good\": [...], \"bad\": [...]}.",
    )

    # Output
    p.add_argument("--output", default=None, help="Save all results to this JSON file.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])

    return p.parse_args()


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _validate(args: argparse.Namespace) -> None:
    """Fail early with a clear message if required arguments are missing."""
    if "ued" in args.metrics and args.checkpoint is None:
        raise SystemExit(
            "ERROR: --metrics ued requires --checkpoint <path/to/E1_best.pt>"
        )
    if "abx" in args.metrics and args.abx_items is None:
        raise SystemExit(
            "ERROR: --metrics abx requires --abx-items <path/to/file.item>"
        )
    if "abx" in args.metrics and args.abx_features is None and not args.abx_extract:
        raise SystemExit(
            "ERROR: --metrics abx requires either --abx-features <dir> "
            "or --abx-extract with --abx-audio-dir <dir> and --abx-features <dir>"
        )
    if "swuggy" in args.metrics and (args.lm_checkpoint is None or args.swuggy_data is None):
        raise SystemExit(
            "ERROR: --metrics swuggy requires --lm-checkpoint and --swuggy-data"
        )
    if "sblim" in args.metrics and (args.lm_checkpoint is None or args.sblim_data is None):
        raise SystemExit(
            "ERROR: --metrics sblim requires --lm-checkpoint and --sblim-data"
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_hubert(args: argparse.Namespace, device: torch.device) -> SpeechEncoder:
    logger.info(f"Loading HuBERT '{args.hubert_name}' layer {args.hubert_layer} …")
    E0 = SpeechEncoder.from_textlesslib(
        name=args.hubert_name,
        layer=args.hubert_layer,
        vocab_size=args.vocab_size,
        deduplicate=True,
        kind_kmeans="kmeans",
    ).to(device)
    E0.eval()
    return E0


def _load_e1(args: argparse.Namespace, device: torch.device) -> RobustQuantizer:
    logger.info(f"Loading E1 from {args.checkpoint} …")
    ckpt = torch.load(args.checkpoint, map_location=device)
    E1 = RobustQuantizer(
        input_dim=768,
        hidden_dim=args.hidden_dim,
        num_codes=args.vocab_size + 1,
    ).to(device)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        epoch = ckpt.get("epoch", "?")
        loss = ckpt.get("best_loss", "?")
    else:
        state_dict = ckpt  # plain state_dict saved directly
        epoch, loss = "?", "?"
    E1.load_state_dict(state_dict)
    E1.eval()
    logger.info(f"  epoch={epoch}, best_loss={loss}")
    return E1


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


# ---------------------------------------------------------------------------
# Per-metric runners
# ---------------------------------------------------------------------------


def run_ued(args: argparse.Namespace, device: torch.device) -> dict:
    E0 = _load_hubert(args, device)
    E1 = _load_e1(args, device)
    evaluator = UEDEvaluator(
        upstream_encoder=E0.dense,
        e1_model=E1,
        device=device,
        vocab_size=args.vocab_size,
    )
    base_cfg = _build_aug_config()
    n_samples = args.n_samples if args.n_samples > 0 else None

    results: dict = {}

    if args.noise_dir is None:
        logger.warning(
            "WARNING: --noise-dir not set. The 'noise' augmentation will fall back to "
            "Gaussian white noise (stationary) instead of DNS non-stationary noise. "
            "UED results for 'noise' will NOT match the paper's numbers."
        )

    # Optional baseline row (E0: HuBERT + KMeans, no E1)
    if getattr(args, "include_baseline", False):
        baseline_evaluator = BaselineUEDEvaluator(E0, device)
        baseline_results: dict = {}
        for aug_name in PAPER_AUGMENTATIONS:
            logger.info(f"  UED baseline [{aug_name}] …")
            try:
                metrics = baseline_evaluator.evaluate_augmentation(
                    root=args.data_root,
                    split=args.split,
                    augmentation_name=aug_name,
                    base_aug_config=base_cfg,
                    noise_dir=args.noise_dir,
                    max_length=160_000,
                    n_samples=n_samples,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
                baseline_results[aug_name] = metrics
                logger.info(f"    mean={metrics['mean_ued']:.2f}%  std={metrics['std_ued']:.2f}%")
            except Exception as exc:
                logger.warning(f"    Failed: {exc}")
                baseline_results[aug_name] = {"error": str(exc)}
        results["_baseline"] = baseline_results

    for aug_name in PAPER_AUGMENTATIONS:
        logger.info(f"  UED [{aug_name}] …")
        try:
            metrics = evaluator.evaluate_augmentation(
                root=args.data_root,
                split=args.split,
                augmentation_name=aug_name,
                base_aug_config=base_cfg,
                noise_dir=args.noise_dir,
                max_length=160_000,
                n_samples=n_samples,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            results[aug_name] = metrics
            logger.info(
                f"    mean={metrics['mean_ued']:.2f}%  std={metrics['std_ued']:.2f}%"
            )
        except Exception as exc:
            logger.warning(f"    Failed: {exc}")
            results[aug_name] = {"error": str(exc)}
    return results


def run_abx(args: argparse.Namespace, device: torch.device) -> dict:
    # Only load HuBERT if we need to extract features
    if args.abx_extract:
        E0 = _load_hubert(args, device)
        evaluator = ABXEvaluator(upstream_encoder=E0.dense, device=device)
        audio_files = sorted(
            glob.glob(os.path.join(args.abx_audio_dir, "**", "*.flac"), recursive=True)
            + glob.glob(os.path.join(args.abx_audio_dir, "**", "*.wav"), recursive=True)
        )
        logger.info(f"  Extracting features for {len(audio_files)} files → {args.abx_features}")
        evaluator.extract_features(audio_files, args.abx_features)
    else:
        # Features already exist; still need the evaluator object for scoring
        # (upstream_encoder is unused when not extracting)
        E0 = _load_hubert(args, device)
        evaluator = ABXEvaluator(upstream_encoder=E0.dense, device=device)

    metrics = evaluator.evaluate(
        features_dir=args.abx_features,
        item_file=args.abx_items,
        distance=args.abx_distance,
    )
    logger.info(f"  Within-speaker ABX:  {metrics['within_speaker_abx']:.2f}%")
    logger.info(f"  Across-speaker ABX:  {metrics['across_speaker_abx']:.2f}%")
    return metrics


def run_swuggy(args: argparse.Namespace) -> dict:
    lm = NgramLM.load(args.lm_checkpoint)
    with open(args.swuggy_data) as f:
        pairs = json.load(f)
    real_seqs = [p["real"] for p in pairs]
    pseudo_seqs = [p["pseudo"] for p in pairs]
    result = LexicalEvaluator(lm).score(real_seqs, pseudo_seqs)
    logger.info(f"  sWUGGY: {result['swuggy']:.2f}%")
    return result


def run_sblim(args: argparse.Namespace) -> dict:
    lm = NgramLM.load(args.lm_checkpoint)
    with open(args.sblim_data) as f:
        pairs = json.load(f)
    good_seqs = [p["good"] for p in pairs]
    bad_seqs = [p["bad"] for p in pairs]
    result = SyntacticEvaluator(lm).score(good_seqs, bad_seqs)
    logger.info(f"  sBLIMP: {result['sblim']:.2f}%")
    return result


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

_AUG_DISPLAY = {
    "time_stretch": "Time Stretch",
    "pitch_shift": "Pitch Shift",
    "reverberation": "Reverberation",
    "noise": "Noise",
}


def _print_summary(results: dict, args: argparse.Namespace) -> None:
    sep = "=" * 62
    ckpt_label = args.checkpoint or "(no E1 checkpoint)"
    print(f"\n{sep}")
    print(f"  Evaluation — {ckpt_label}")
    print(f"  Split: {args.split}  |  Metrics: {', '.join(args.metrics)}")
    print(sep)

    if "ued" in results:
        ued = results["ued"]
        baseline = ued.get("_baseline")
        print("\n  Unit Edit Distance (UED)  [lower is better]\n")
        if baseline:
            print(f"  {'Augmentation':<20}  {'E0 mean':>8}  {'E0 ±sem':>8}  {'E1 mean':>8}  {'E1 ±sem':>8}")
            print("  " + "-" * 62)
            for aug_key in PAPER_AUGMENTATIONS:
                disp = _AUG_DISPLAY[aug_key]
                b = baseline.get(aug_key, {})
                e = ued.get(aug_key, {})
                if "mean_ued" in b:
                    b_str = f"{b['mean_ued']:>8.2f}  {b['sem_ued']:>8.2f}"
                else:
                    b_str = f"{'ERROR':>8}  {'':>8}"
                if "mean_ued" in e:
                    e_str = f"{e['mean_ued']:>8.2f}  {e['sem_ued']:>8.2f}"
                else:
                    e_str = f"{'ERROR':>8}  {'':>8}"
                print(f"  {disp:<20}  {b_str}  {e_str}")
        else:
            print(f"  {'Augmentation':<20}  {'Mean UED (%)':>12}  {'±SEM':>8}")
            print("  " + "-" * 44)
            for aug_key in PAPER_AUGMENTATIONS:
                disp = _AUG_DISPLAY[aug_key]
                data = ued.get(aug_key, {})
                if "error" in data:
                    print(f"  {disp:<20}  {'ERROR':>12}")
                elif data:
                    print(f"  {disp:<20}  {data['mean_ued']:>12.2f}  {data['sem_ued']:>8.2f}")

    if "abx" in results:
        data = results["abx"]
        if "error" not in data:
            print("\n  ABX Error Rate  [lower is better]\n")
            print(f"  {'Within-speaker':<30}  {data['within_speaker_abx']:>8.2f}%")
            print(f"  {'Across-speaker':<30}  {data['across_speaker_abx']:>8.2f}%")

    lm_items = {}
    if "swuggy" in results and "error" not in results["swuggy"]:
        lm_items["sWUGGY"] = results["swuggy"]["swuggy"]
    if "sblim" in results and "error" not in results["sblim"]:
        lm_items["sBLIMP"] = results["sblim"]["sblim"]
    if lm_items:
        print("\n  LM-based Metrics  [higher is better]\n")
        for name, val in lm_items.items():
            print(f"  {name:<30}  {val:>8.2f}%")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    _validate(args)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  |  Running metrics: {args.metrics}")

    results: dict = {}

    if "ued" in args.metrics:
        logger.info("=" * 50 + " UED")
        try:
            results["ued"] = run_ued(args, device)
        except Exception as exc:
            logger.error(f"UED evaluation failed: {exc}")
            results["ued"] = {"error": str(exc)}

    if "abx" in args.metrics:
        logger.info("=" * 50 + " ABX")
        try:
            results["abx"] = run_abx(args, device)
        except Exception as exc:
            logger.error(f"ABX evaluation failed: {exc}")
            results["abx"] = {"error": str(exc)}

    if "swuggy" in args.metrics:
        logger.info("=" * 50 + " sWUGGY")
        try:
            results["swuggy"] = run_swuggy(args)
        except Exception as exc:
            logger.error(f"sWUGGY evaluation failed: {exc}")
            results["swuggy"] = {"error": str(exc)}

    if "sblim" in args.metrics:
        logger.info("=" * 50 + " sBLIMP")
        try:
            results["sblim"] = run_sblim(args)
        except Exception as exc:
            logger.error(f"sBLIMP evaluation failed: {exc}")
            results["sblim"] = {"error": str(exc)}

    _print_summary(results, args)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()
